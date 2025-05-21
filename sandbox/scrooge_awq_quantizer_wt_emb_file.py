"""
todo: currently not working b/c the embedding files saved didn't have
todo: the embedding vs rotary embedding info!

Custom AWQ Quantizer (src/scrooge_awq_quantizer.py)
Splits calibration and quantization phases cleanly.

Built on top of awq.quantize.quantizer.AwqQuantizer.

START
  |
  v
Load model and tokenizer
  |
  v
Extract quantizable modules from model
  (self.modules = [Layer0, Layer1, ..., LayerN])
  |
  v
Load calibration samples
  (self.inps = batch of tokenized text)
  |
  v
┌───────────────────────────────────────────────┐
| For each module (layer) in self.modules:       |
|                                               |
|   1. Move layer to GPU                        |
|   2. Forward pass: module(self.inps)           |
|        (this is the actual INFERENCE step)     |
|   3. Capture input activations                |
|   4. Search best scaling factors (per channel)|
|   5. (Optional) Search best clipping thresholds|
|   6. clear_memory() after processing          |
└───────────────────────────────────────────────┘
  |
  v
(Optionally) Save calibration statistics (scales + clips) to disk
  |
  v
(Optional) Delete calibration inputs (self.inps) to save RAM
  |
  v
Apply Quantization:
┌───────────────────────────────────────────────┐
| For each module (layer) in self.modules:       |
|                                               |
|   1. Load layer onto GPU                      |
|   2. Apply precomputed scales and clips       |
|   3. Pseudo-quantize weights (int4 simulation)|
|   4. Replace Linear layer with WQLinear_GEMM  |
|      (or GEMV, Marlin, GEMVFast)               |
|   5. (Optional) Save quantized layer to disk  |
|   6. Move quantized module to CPU if needed   |
|   7. clear_memory()                           |
└───────────────────────────────────────────────┘
  |
  v
Mark model as quantized (self.is_quantized = True)
  |
  v
(Optional) Save full model to disk
  |
  v
END

"""

import logging
import os
import time
from typing import Any, Optional, List, Dict, Tuple, Union
from typing_extensions import override
from collections import defaultdict
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from awq.quantize.quantizer import AwqQuantizer
from awq.utils.module import (
    get_named_linears,
    exclude_layers_to_not_quantize,
    get_op_name,
    append_str_prefix,
)
from awq.quantize.scale import apply_scale, apply_clip
from awq.utils.utils import get_best_device, clear_memory
from awq.utils.calib_data import get_calib_dataset

# Project level
from utils.gpu_monitor import log_gpu_usage
from quantize.quantize_utils import safe_update

logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ScroogeAwqQuantizer(AwqQuantizer):
    """
    Inherits from autoawq's AWQ Quantizer class with explicit explicit
    calibration and quantization separation.

    """

    def __init__(
        self,
        model: Union[nn.Module, PreTrainedModel],
        tokenizer: Any,
        max_calib_samples: int = 96,  # ✅ lower from 512 to 128
        max_calib_seq_len: int = 1024,  # ✅ lower from 2048 to 512
        apply_clip: bool = True,
        n_parallel_calib_samples: int = 8,  # ! Need to set this low if GPU is small
    ):
        """
        * Need to set __init__ b/c we are "hijacking" the official autoawq model
        (directly trying to instantiate an AwqQuantizer subclass without having
        a full initialized context yet.)

        Initialize a lightweight AWQ quantizer for DeepSeek Scrooge-style calibration
        and quantization.

        Note:
            In the official AutoAWQ code, two model references are maintained internally:
            - `awq_model`: The AWQ wrapper object (adds quantization utilities)
            - `awq_model.model`: The inner HuggingFace model (for pure forward passes
            and layers)

            When creating ScroogeQuantizer, pass a HuggingFace model directly.
            Internally, it is assigned to both `self.awq_model` and `self.model`.

        Args:
            - model (nn.Module): The HuggingFace model to calibrate and quantize.
            - tokenizer (Any): The tokenizer compatible with the model.
            - max_calib_samples (int, optional): Maximum number of calibration samples.
            Defaults to 128 considering my GPU size.
            * Official default for best quality is 512.)
            - max_calib_seq_len (int, optional): Maximum sequence length for
            calibration samples.
            Defaults to 512 considering small GPU size (normally default to 2048.)
            - apply_clip (bool, optional): Whether to search and apply clipping.
            Defaults to True.
            - n_parallel_calib_samples (int, optional):
                Number of calibration samples processed in parallel per forward pass.
                Balances calibration speed and memory usage. Higher values are faster
                but use more GPU memory. Defaults to 64.
        Example:
            quantizer = ScroogeAwqQuantizer(
                model=full_precision_model,
                tokenizer=tokenizer,
            )
        """
        #! Do NOT call super().__init__() because AwqQuantizer expects a full config
        #! which we do not have

        # ✅ Core parameters
        self.awq_model: nn.Module = model  # For AWQ calibration logic
        self.model: nn.Module = (
            model.model
        )  # Core transformer model that has .layers (w/o the huggingface wrapper)
        self.tokenizer: Any = tokenizer
        self.max_calib_samples: int = max_calib_samples
        self.max_calib_seq_len: int = max_calib_seq_len
        self.apply_clip: bool = apply_clip
        self.n_parallel_calib_samples = n_parallel_calib_samples

        # ✅ Calibration state (initialized later)
        self.modules: Optional[List[nn.Module]] = None
        self.module_kwargs: Optional[Dict[str, Any]] = None
        self.inps: Optional[torch.Tensor] = None

        # ✅ Calibration results
        self.all_scales: Optional[Dict[str, torch.Tensor]] = None
        self.all_clips: Optional[Dict[str, torch.Tensor]] = None

        # ✅ Calibration dataset
        self.calib_data = None
        self.split = "validation"
        self.text_column = "text"
        self.dataset_name = "pileval"

        # ✅ Others:
        self.modules_to_not_convert = []  # empty = quantize everything

    @staticmethod
    def get_layers_for_scaling(
        module: nn.Module,
        input_feat: Dict[str, torch.Tensor],
        module_kwargs: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        * Note: Works with standard HF Transformers models only (not AWQ directly)

        Define quantization scaling groups inside a Transformer block (module).

        This method splits each Transformer block into logical groups of linear
        layers for quantization calibration. Each group shares input features and
        scaling behaviors.

        Model structure:
            Model
            └── List of modules (blocks) [block1, block2, block3, ..., blockN]
                └── Inside each block:
                    └── Groups:
                        - Attention input group (q_proj, k_proj, v_proj)
                        - Attention output group (o_proj)
                        - MLP input group (gate_proj, up_proj)
                        - MLP output group (down_proj)

        Args:
            - module (nn.Module): A Transformer block containing attention and MLP
            submodules.
            - input_feat (Dict[str, torch.Tensor]): Captured activation features
            for specific sublayers.
            - module_kwargs (Dict[str, Any]): Optional keyword arguments for module
            behavior.

        Returns:
            - List[Dict[str, Any]]: List of layer groups, each group defined as
            a dictionary
            - describing sublayers, previous operations, and input activations
            for calibration.
        """
        layers = []

        # Attention input scaling
        layers.append(
            dict(
                prev_op=module.input_layernorm,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )

        # Attention output scaling
        layers.append(
            dict(
                prev_op=module.self_attn.v_proj,
                layers=[module.self_attn.o_proj],
                inp=input_feat["self_attn.o_proj"],
            )
        )

        # MLP (Multi-Layer Perceptron) input scaling
        # (feed-forward neural network layers that follows the attention mechanism.)
        layers.append(
            dict(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                inp=input_feat["mlp.gate_proj"],
                module2inspect=module.mlp,
                kwargs=module_kwargs,
            )
        )

        # MLP output scaling
        layers.append(
            dict(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
            )
        )

        return layers

    @staticmethod
    def get_model_layers(model: nn.Module) -> List[nn.Module]:
        """
        Retrieve the list of Transformer blocks from the model.

        Specifically designed for DeepSeek R1 / Qwen2 architectures,
        where the model has a `.model.layers` attribute containing all
        Transformer block modules (e.g., QWenBlock instances).

        Args:
            - model (nn.Module): HuggingFace model containing nested Transformer
            layers.

        Returns:
            List[nn.Module]: List of Transformer block modules to be calibrated
            or quantized.
        """
        return model.layers

    @staticmethod
    def move_embed(model: nn.Module, device: Any) -> None:
        """
        Move model embeddings and rotary positional embeddings to the specified device.

        Specifically tailored for DeepSeek R1 / Qwen2 model architectures,
        where the model has `.model.embed_tokens` and `.model.rotary_emb` attributes.

        Args:
            - model (nn.Module): The model containing nested embedding and rotary
            embedding modules.
            - device (Any): The target device to move the tensors to
            (e.g., "cuda", "cpu", or a torch.device object).

        Note:
            Rotary embeddings inject position into the attention mechanism itself
            (by rotating Q and K), keeping embeddings and V pure.
            * Why: Rotary enables longer context lengths and better scaling with
            * cleaner attention internals.

        Returns:
            None
        """
        model.embed_tokens = model.embed_tokens.to(device)
        model.rotary_emb = model.rotary_emb.to(device)

    @override
    def init_quant(
        self, n_samples: int = 128, max_seq_len: int = 512
    ) -> tuple[List[nn.Module], Dict[str, Any], torch.Tensor]:
        """
        * Custom init_quant() that OVERRIDES the self.awq_model's (the official)
        * init_quant() method, which relies on loading a custom QWEN model specific
        * AWQ model

        Initializes quantization calibration by collecting input activations from
        the model.

        This function:
        - Fetches all Transformer blocks (modules) from the model.
        - Loads a calibration dataset and prepares input samples.
        - Moves the first module and model embeddings to the optimal device.
        - Captures input activations using a Catcher module for calibration.
        - Prepares model input kwargs (e.g., attention masks) for later processing.
        - Returns modules, layer-specific kwargs, and input activations.

        Args:
            - n_samples (int, optional): Number of calibration samples to use.
            Defaults to 128.
            - max_seq_len (int, optional): Maximum sequence length for calibration
            samples. Defaults to 512.

        Returns:
            tuple:
                - modules (List[nn.Module]): List of Transformer block modules
                (e.g., QWenBlocks).
                - layer_kwargs (Dict[str, Any]): Extra kwargs (such as attention masks)
                needed during inference.
                - inps (torch.Tensor): Captured input activations for the first module
                during calibration.
        """
        logger.info(
            f"🔍 [init_quant] Starting calibration with n_samples={n_samples}, max_seq_len={max_seq_len}"
        )

        # 1. Fetch transformer layers
        try:
            modules = self.get_model_layers(self.model)
            if not modules:
                raise ValueError(
                    "No modules (layers) found in model during init_quant()."
                )
            logger.info(f"✅ [init_quant] Retrieved {len(modules)} transformer blocks.")
        except Exception as e:
            logger.error(f"❌ [init_quant] Failed to fetch model layers: {e}")
            raise

        # 2. Load calibration samples (Use autoawq's default datase - pileval)
        try:
            samples = get_calib_dataset(
                data=self.calib_data or "pileval",  # default to pileval
                tokenizer=self.tokenizer,
                n_samples=n_samples,
                max_seq_len=max_seq_len,
                split=self.split,
                text_column=self.text_column,
            )
            logger.info(f"✅ [init_quant] Loaded calibration dataset: pileval")
        except Exception as e:
            logger.warning(f"⚠️ [init_quant] Failed to load pileval: {e}")
            logger.info(f"🔄 [init_quant] Falling back to c4...")

            try:
                samples = get_calib_dataset(
                    data="c4",  # fallback to c4
                    tokenizer=self.tokenizer,
                    n_samples=n_samples,
                    max_seq_len=max_seq_len,
                    split="validation",
                    text_column="text",
                )
                logger.info(
                    f"✅ [init_quant] Successfully loaded fallback calibration dataset: c4"
                )
            except Exception as e2:
                logger.error(
                    f"❌ [init_quant] Failed to load fallback c4 as well: {e2}"
                )
                raise RuntimeError(
                    "Calibration dataset loading failed for both pileval and c4."
                ) from e2

        samples = torch.cat(samples, dim=0)

        logger.info(
            f"✅ [init_quant] Loaded and tokenized {samples.shape[0]} calibration samples."
        )

        # Step 3: Prepare holders
        # Set holders to captured input activations from the embedding layer &
        # Additional inference inputs (e.g., attention mask, position ids)
        inps = []
        layer_kwargs = {}
        logger.info("Ins and layer kwargs holders created.")

        # 4. Move module[0] (first) and embeddings to the best device (usually GPU)

        # Note:
        # - modules[0] (first Transformer block) is temporarily moved to GPU for capturing
        # calibration inputs.
        # - After catching inputs, modules[0] is moved back to CPU to save memory.
        # - All other modules (modules[1:] onwards) were never moved to GPU;
        # they remain safely on CPU throughout.
        # - Therefore, no additional device moves are needed for modules[1:],
        # only modules[0] and embeddings.

        best_device = get_best_device()
        modules[0] = modules[0].to(best_device)
        self.move_embed(self.model, best_device)

        # 5. Catch the first input (using the Catcher trick)
        # (embeddings of the input text)
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                if len(args) > 0:
                    hidden_states = args[0]
                    del args
                else:
                    first_key = list(kwargs.keys())[0]
                    hidden_states = kwargs.pop(first_key)

                inps.append(hidden_states)
                layer_kwargs.update(kwargs)
                raise ValueError  # Early exit to avoid full forward

        modules[0] = Catcher(modules[0])

        try:
            self.model(samples.to(next(self.model.parameters()).device))
        except ValueError:
            logger.info(
                f"✅ [init_quant] Caught expected early exit after catching input activations."
            )
            # 🚀 Optimization: immediately free module 0
            modules[0] = modules[0].module  # unwrap Catcher
            modules[0] = modules[0].cpu()  # move to CPU immediately
            logger.info(
                "✅ [init_quant] First module moved back to CPU early after catching inputs."
            )
        except Exception as e:
            logger.error(f"❌ [init_quant] Unexpected error during forward pass: {e}")
            raise
        finally:
            # Only unwrap if still wrapped
            if isinstance(modules[0], Catcher):
                modules[0] = modules[0].module  # Restore original first module

        # 6. Prepare generation inputs
        try:
            layer_kwargs = self.model.prepare_inputs_for_generation(
                samples, **layer_kwargs
            )
            layer_kwargs.pop("input_ids", None)  # input_ids not needed
            logger.info(f"✅ [init_quant] Prepared model input kwargs for generation.")
        except Exception as e:
            logger.error(
                f"❌ [init_quant] Failed to prepare inputs for generation: {e}"
            )
            raise

        del samples
        inps = inps[0]

        # 7. Move modules back to CPU
        try:
            modules[0] = modules[0].cpu()
            self.move_embed(self.model, "cpu")
            clear_memory()
            logger.info(
                f"🧹 [init_quant] Modules and embeddings moved back to CPU, memory cleared."
            )
        except Exception as e:
            logger.error(f"❌ [init_quant] Failed during cleanup phase: {e}")
            raise

        # 8. Ensure attention_mask is on correct device
        if layer_kwargs.get("attention_mask") is not None:
            try:
                layer_kwargs["attention_mask"] = layer_kwargs["attention_mask"].to(
                    best_device
                )
                logger.info(f"✅ [init_quant] attention_mask moved to {best_device}")
            except Exception as e:
                logger.error(
                    f"❌ [init_quant] Failed to move attention_mask to device: {e}"
                )
                raise

        logger.info(f"🏁 [init_quant] Calibration data initialization complete.")

        log_gpu_usage("Before Clear")
        clear_memory()  # Free up memory for next texts
        log_gpu_usage("After Clear")

        return modules, layer_kwargs, inps

    def init_quant_modules_only(
        self, precomputed_embeds: torch.Tensor
    ) -> Tuple[List[nn.Module], Dict[str, Any], None]:
        """
        * Use this method if embeddings are precomputed and loaded in from file.

        Init modules + forward kwargs only — skip sampling and embedding.
        """
        modules = self.get_model_layers(self.model)

        # Prepare generation inputs just like init_quant would do
        try:
            layer_kwargs = self.model.prepare_inputs_for_generation(precomputed_embeds)
            layer_kwargs.pop("input_ids", None)  # don't need input_ids
        except Exception:
            layer_kwargs = {}

        return modules, layer_kwargs, None

    def use_precomputed_embeddings(
        self,
        embeddings: torch.Tensor,
        expected_num_samples: Optional[int] = None,
    ) -> None:
        """
        Inject precomputed input embeddings into the quantizer.

        This bypasses tokenization and embedding steps and prepares
        the quantizer to run calibration immediately.

        Args:
            embeddings (torch.Tensor): Precomputed input activations [n_samples, seq_len, hidden_dim].
            expected_num_samples (int, optional): Sanity check to verify input batch size.

        Sets:
            self.inps: ← embeddings
            self.modules: ← list of transformer blocks
            self.module_kwargs: ← any generation-related inputs
        """
        if expected_num_samples is not None:
            if embeddings.size(0) != expected_num_samples:
                raise ValueError(
                    f"Expected {expected_num_samples} samples, but got {embeddings.size(0)}"
                )

        logger.info(f"🔗 Injecting precomputed embeddings: {embeddings.shape}")

        # 🔥 Unwrap if necessary
        if hasattr(self.model, "model"):
            logger.info(
                f"Unwrapping nested model: {self.model.__class__} → {self.model.model.__class__}"
            )
            self.model = self.model.model

        self.inps = embeddings
        self.modules, self.module_kwargs, _ = self.init_quant_modules_only(embeddings)

        logger.info("✅ Quantizer is ready to calibrate using precomputed embeddings.")

    def calibrate(
        self,
        save_path: Optional[str] = None,
        clear_inps: bool = True,
        precomputed_embeds: Optional[torch.Tensor] = None,
    ) -> None:
        """
        * Calibrate the model layers for AWQ quantization.

        This method performs layer-by-layer calibration across two distinct
        phases:
        - Staging Phase: Describe what to calibrate (selecting Linear layers
        and grouping them).
        - Computing Phase: Search for optimal scaling factors (and optionally
        clipping thresholds).

        ----------
        Detailed Step-by-Step Process:

        1. **Initialization:**
            - Unwrap nested model if needed (self.model.model).
            - Move embedding layers (embed_tokens, rotary_emb) to GPU or
            best device.
            - Run init_quant() to prepare calibration samples and capture
            initial activations.

        2. **Module Calibration Loop:**
            For each Transformer block/module:

            2.1 **Inference Step (Activation Capture):**
                - Run the module's forward pass over calibration inputs.
                - Capture input activations into `input_feat`.
                - Structure: Dict mapping layer names → activation tensors
                (torch.Tensor).

            2.2 **Staging Phase (Describing What to Calibrate):**
                - Call `get_layers_for_scaling()` to generate `module_config`:
                    - Each entry describes:
                        - prev_op: Previous operation (e.g., LayerNorm)
                        - layers: List of Linear submodules
                        (e.g., [q_proj, k_proj, v_proj])
                        - inp: Input activation tensor
                        - module2inspect: Target module
                        - kwargs: Optional extra arguments
                - Merged projection layers (e.g., QKV combined) are automatically
                grouped here.

            2.3 **Computing Phase (Scale and Clip Search):**
                - Call `_search_best_scale()` for each group:
                    - Search per-channel scaling factors that minimize quantization
                    loss.
                - If `apply_clip=True`, call `_search_best_clip()`:
                    - Find optimal clipping thresholds to clamp activations.
                    - Improve quantization robustness against outliers.

            2.4 **Post-module Cleanup:**
                - After calibrating each module, call `clear_memory()` to free GPU RAM.

        3. **Finalization:**
            - Save calibration statistics (`all_scales`, `all_clips`) to `save_path`
            if provided.
            - Update object state: self.all_scales, self.all_clips.
            - Optionally clear calibration inputs (self.inps) to save RAM.

        ----------
        Parameters:
            save_path (Optional[str], default=None):
                Path to save calibration statistics (scales and clips).
            clear_inps (bool, default=True):
                Whether to delete captured input activations (self.inps) after calibration.
            precomputed_embeds (torch.Tensor, optional): Precomputed input embeddings.
                If provided, skips sampling and embedding steps.
        ----------
        Outputs:
            Updates self.all_scales: Dict[str, torch.Tensor]
            Updates self.all_clips: Dict[str, torch.Tensor] (only if clipping applied)

        ----------
        Important Notes:
        - Capturing input activations (input_feat) happens per module during init_quant().
        - Calibration uses a "describe first, compute later" model to separate concerns
        cleanly.
        - Supports both standard architectures and merged projections
        (e.g., fused c_attn layers).
        - clear_memory() is used aggressively between modules
        to manage GPU memory pressure.

        ----------
        Logging/Debugging:
        - Structured logs for every major phase (model moves, module calibration,
        memory cleanup).
        - Errors caught for model moving, calibration, and save operations.
        """
        # Start calibration timer
        start_time = time.time()

        if precomputed_embeds is not None:
            assert isinstance(precomputed_embeds, torch.Tensor)
            logger.info(
                f"🔗 [calibrate] Received precomputed embeddings of shape: {precomputed_embeds.shape}, \
device: {precomputed_embeds.device}"
            )
            self.use_precomputed_embeddings(precomputed_embeds)
        else:
            logger.info("🔄 No precomputed embeddings provided — generating.")
            # self.modules, self.module_kwargs, self.inps = self.init_quant(
            #     n_samples=self.max_calib_samples,
            #     max_seq_len=self.max_calib_seq_len,
            # )

            # * Move model embeddings manually to correct device
            device = get_best_device()

            if hasattr(self.model, "model"):
                logger.info(
                    f"🔍 [calibrate] Unwrapping nested model (self.model.model)."
                )
                self.model = self.model.model

            try:
                self.model.embed_tokens = self.model.embed_tokens.to(device)
                self.model.rotary_emb = self.model.rotary_emb.to(device)
                logger.info(
                    f"✅ [calibrate] Embedding layers moved to device: {device}"
                )
            except Exception as e:
                logger.error(f"❌ [calibrate] Failed to move embeddings to device: {e}")
                raise

            # Sanity check
            if not hasattr(self.model, "layers"):
                logger.error(f"❌ [calibrate] Model does not have '.layers' attribute!")
                raise AttributeError(
                    "Model does not have .layers! Check self.model assignment."
                )
            logger.info(f"✅ [calibrate] Model structure validated.")

            # Load sample dataset and input activations
            try:
                self.modules, self.module_kwargs, self.inps = self.init_quant(
                    n_samples=self.max_calib_samples,
                    max_seq_len=self.max_calib_seq_len,
                )
                logger.info(f"✅ [calibrate] Calibration inputs initialized.")
            except Exception as e:
                logger.error(f"❌ [calibrate] Failed during init_quant(): {e}")
                raise

        all_scales: Dict[str, torch.Tensor] = {}
        all_clips: Dict[str, torch.Tensor] = {}

        assert self.modules is not None, "self.modules must be set before calibration"

        for i, module in enumerate(self.modules):
            logger.info(f"🔍 [calibrate] Processing module {i}/{len(self.modules)}.")

            # 1. Move module to GPU
            device = get_best_device()
            module = module.to(device)
            self.modules[i] = module  # update the list
            logger.info(f"[calibrate] Module {i} moved to GPU.")

            # 2. Get input activations and move them to GPU
            named_linears = get_named_linears(module)
            named_linears = exclude_layers_to_not_quantize(
                named_linears, self.modules_to_not_convert
            )
            input_feat = self._get_input_feat(module, named_linears)
            logger.info(f"[calibrate] Finished getting input feature for module {i}.")

            # Inside _get_input_feat, must move input_feat to the right device
            # (move tensor one by one)
            # _get_input_feat does not directly use self.inps.
            # It creates a new input_feat by registering hooks on the module;
            # it captures new activations during a forward pass through the module!

            input_feat = {k: v.to(device) for k, v in input_feat.items()}
            logger.info(f"[calibrate] Input features for module {i} moved to CPU.")

            # 3. Staging: Describe layers to calibrate
            module_config = self.model.get_layers_for_scaling(
                module, input_feat, self.module_kwargs
            )

            for j, layer in enumerate(module_config):
                layer_name = layer.get("name", f"Layer_{j}")
                logger.info(
                    f"  🔹 [calibrate] Module {i}/{len(self.modules)} — Layer {j}/{len(module_config)}: {layer_name}"
                )
                self._search_best_scale(module, **layer)

            # 4. Compute best scaling factors
            # scales_list = [
            #     self._search_best_scale(module, **layer) for layer in module_config
            # ]
            # scales_list = append_str_prefix(
            #     scales_list, get_op_name(self.model, module) + "."
            # )
            # Use the loop format instead of list comprehension
            scales_list = []
            for j, layer in enumerate(module_config):
                layer_name = layer.get("name", f"Layer_{j}")
                logger.info(
                    f"  🔹 [calibrate] Module {i}/{len(self.modules)} — Layer {j}/{len(module_config)}: {layer_name}"
                )

                scale = self._search_best_scale(module, **layer)
                scales_list.append(scale)

            # Quick type checking b/c src code lacked clear typing
            try:
                safe_update(all_scales, scales_list, name="scales", strict=True)
            except Exception as e:
                logger.error(f"Calibration failed due to bad data: {e}")
                # Optionally re-raise or exit gracefully
                raise

            # 5. Optionally compute clipping
            if self.apply_clip:
                clip_list = self._search_best_clip(module, named_linears, input_feat)
                clip_list = append_str_prefix(
                    clip_list, get_op_name(self.model, module) + "."
                )
                # Quick type checking b/c src code lacked clear typing
                try:
                    safe_update(all_clips, clip_list, name="clips", strict=True)
                except Exception as e:
                    logger.error(f"Calibration failed due to bad data: {e}")
                    raise

            # 6. Move module back to CPU
            module = module.cpu()
            self.modules[i] = module

            # 7. Clear memory to free up GPU memory
            clear_memory()  # Free up memory between modules
            logger.info(f"🧹 [calibrate] Memory cleared after module {i}.")

        # Save calibration stats if needed
        if save_path:
            try:
                torch.save({"scales": all_scales, "clips": all_clips}, save_path)
                logger.info(
                    f"💾 [calibrate] Calibration statistics saved to {save_path}"
                )
            except Exception as e:
                logger.error(f"❌ [calibrate] Failed to save calibration stats: {e}")
                raise

        # Update object state
        self.all_scales = all_scales
        self.all_clips = all_clips

        # Clear calibration inputs if requested
        if clear_inps:
            try:
                del self.inps
                self.inps = None
                clear_memory()
                logger.info(f"✅ [calibrate] Calibration inputs cleared from memory.")
            except Exception as e:
                logger.warning(f"⚠️ [calibrate] Failed to clear calibration inputs: {e}")

        logger.info(f"🏁 [calibrate] Calibration completed successfully.")

        # End calibration timer
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(
            f"⏱️ [calibrate] Calibration time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}"
        )

    def apply_quantization(
        self, calib_stats_path: str = None, save_layers_dir: str = None
    ):
        """
        Apply quantization:
        - Load calibration stats (optional)
        - Apply scales and clip
        - Quantize weights
        - (Optional) Save each layer immediately after quantizing
        """
        if calib_stats_path:
            stats = torch.load(calib_stats_path)
            self.all_scales = stats.get("scales", {})
            self.all_clips = stats.get("clips", {})

        os.makedirs(save_layers_dir, exist_ok=True) if save_layers_dir else None

        for i in range(len(self.modules)):
            module = self.modules[i]
            named_linears = get_named_linears(module)
            named_linears = exclude_layers_to_not_quantize(
                named_linears, self.modules_to_not_convert
            )

            if hasattr(self, "all_scales") and self.all_scales:
                apply_scale(module, self.all_scales, input_feat_dict=None)

            if hasattr(self, "all_clips") and self.apply_clip and self.all_clips:
                apply_clip(module, self.all_clips)

            self._apply_quant(module, named_linears)

            if save_layers_dir:
                layer_name = f"model.layers.{i}"
                self._save_layer_quantized(
                    module, layer_name=layer_name, save_dir=save_layers_dir
                )

                # Optionally move CPU-side if you don't need inference here
                self.modules[i] = self.modules[i].cpu()

            clear_memory()

    def _save_layer_quantized(self, module, layer_name: str, save_dir: str):
        """
        Save quantized tensors of a single layer to disk.
        """
        layer_state_dict = {}

        named_linears = get_named_linears(module)

        for name, submodule in named_linears.items():
            if hasattr(submodule, "qweight"):
                full_name = f"{layer_name}.{name}"

                # Save qweight
                layer_state_dict[f"{full_name}.qweight"] = submodule.qweight.cpu()

                # Save scales
                layer_state_dict[f"{full_name}.scales"] = submodule.scales.cpu()

                # Save qzeros if exists
                if hasattr(submodule, "qzeros") and submodule.qzeros is not None:
                    layer_state_dict[f"{full_name}.qzeros"] = submodule.qzeros.cpu()

        save_path = os.path.join(save_dir, f"{layer_name.replace('.', '_')}.pt")
        torch.save(layer_state_dict, save_path)
