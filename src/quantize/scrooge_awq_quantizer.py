"""
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
from typing import Any, cast, Optional, List, Dict, Tuple, Union
from types import SimpleNamespace
from typing_extensions import override
from collections import defaultdict
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from huggingface_hub import snapshot_download, save_torch_state_dict

from awq.quantize.quantizer import AwqQuantizer
from awq.quantize.scale import apply_scale, apply_clip
from awq.utils.module import (
    get_named_linears,
    get_op_name,
    exclude_layers_to_not_quantize,
    append_str_prefix,
)
from awq.utils.utils import get_best_device, clear_memory
from awq.utils.calib_data import get_calib_dataset
from awq.modules.linear.gemm import WQLinear_GEMM

# Project level
from utils.memory_logger import cuda_memory_logger
from quantize.quantize_utils import (
    safe_update,
    unwrap_to_transformer,
    flatten_scales_or_clip_list,
    ScaleEntry,
    get_safe_parallel_sample_count,
)

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
        quant_config: Dict,
        processor: Optional[nn.Module] = None,
        max_calib_samples: int = 96,  # ✅ lower from 512 to 128
        max_calib_seq_len: int = 1024,  # ✅ lower from 2048 to 512
        apply_clip: bool = True,
        n_parallel_calib_samples: int = get_safe_parallel_sample_count(),
        # ! Need to set this low if GPU is small (to 4 or 2 for small VRAM)
        group_size: int = 128,
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
        self.model: nn.Module = model  # Do not unwrap to the QWEN model!
        self.tokenizer: Any = tokenizer
        self.quant_config = quant_config
        self.processor = processor
        self.max_calib_samples: int = max_calib_samples
        self.max_calib_seq_len: int = max_calib_seq_len
        self.apply_clip: bool = apply_clip
        self.n_parallel_calib_samples = n_parallel_calib_samples

        # ✅ Calibration state (initialized later)
        self.modules: Optional[List[nn.Module]] = None
        self.module_kwargs: Optional[Dict[str, Any]] = None
        self.inps: Optional[torch.Tensor] = None

        # ✅ Calibration results
        self.all_scales: List[Tuple[str, torch.Tensor]] = []
        self.all_clips: List[Tuple[str, torch.Tensor]] = []

        # ✅ Calibration dataset
        self.calib_data = None
        self.split = "validation"
        self.text_column = "text"
        self.dataset_name = "pileval"

        # ✅ Others:
        self.modules_to_not_convert = []  # empty = quantize everything
        self.group_size = group_size  # Standard default for AWQ
        self.max_chunk_memory = 64 * 1024 * 1024  # 64 MB
        # * standard default is 1024 MB (dialed it down due to smaller GPU size
        # amount of memory allowed when chunking/calibrating activations for scale search.
        self.duo_scaling = False
        self.zero_point = False  # Set to symmetric
        self.w_bit: int = 4

    def log_vram_usage(self, tag: str):
        """Log vram utilization"""
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        logger.info(
            f"[{tag}] VRAM - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB"
        )

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
        transformer = unwrap_to_transformer(model)

        logger.debug(f"Model structure after unwrap: {type(transformer)}")
        layers = getattr(transformer, "layers", None)

        if layers is None:
            logger.warning("No layers found in the model structure.")
            return []

        logger.info(f"Total layers identified: {len(layers)}")
        return layers

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
        try:
            transformer = unwrap_to_transformer(model)
            transformer.embed_tokens = transformer.embed_tokens.to(device)
            transformer.rotary_emb = transformer.rotary_emb.to(device)
        except AttributeError as e:
            raise AttributeError(
                "Could not find embed_tokens or rotary_emb inside model.model. Verify model structure."
            ) from e

    def unwrap_to_transformer(self, model: nn.Module) -> nn.Module:
        """
        Traverse nested model wrappers (e.g., AWQ → HuggingFace → Transformer).
        Returns the core transformer (e.g., Qwen2Model).
        """
        if hasattr(model, "model") and hasattr(model.model, "model"):
            return model.model.model
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model
        elif hasattr(model, "layers"):
            return model
        else:
            raise AttributeError(
                "Could not unwrap model to transformer (no .layers found)."
            )

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

        # 1. Unwrap to transformer and get layers
        try:
            transformer = self.unwrap_to_transformer(self.model)
            modules = transformer.layers
            if not modules:
                raise ValueError("No transformer blocks found.")
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
            samples = get_calib_dataset(
                data="c4",
                tokenizer=self.tokenizer,
                n_samples=n_samples,
                max_seq_len=max_seq_len,
                split="validation",
                text_column="text",
            )
            logger.info("✅ Loaded fallback calibration dataset: c4")

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

        # 6. Prepare generation inputs (one .model hop only)
        try:
            layer_kwargs = self.model.model.prepare_inputs_for_generation(
                samples, **layer_kwargs
            )
            layer_kwargs.pop("input_ids", None)  # input_ids not needed
            logger.info(f"✅ [init_quant] Prepared model input kwargs for generation.")
        except Exception as e:
            logger.error(
                f"❌ [init_quant] Failed to prepare inputs for generation: {e}"
            )
            raise

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

        cuda_memory_logger(tag="Before Clear")
        clear_memory()  # Free up memory for next texts
        cuda_memory_logger(tag="After Clear")

        # Check if all layers are moved
        for i, layer in enumerate(modules):
            device_types = {p.device.type for p in layer.parameters()}
            logger.info(f"[init_quant] Layer {i} devices: {device_types}")

        return modules, layer_kwargs, inps[0]

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

        start_time = time.time()
        logger.info("🚀 [calibrate] Starting AWQ calibration...")

        # Move model embedding layers to GPU
        device = get_best_device()

        # ✅ Don't unwrap!
        logger.info(
            f"✅ [calibrate] Model remains as {self.model.__class__} — wrapper preserved."
        )

        try:
            self.model.model.model.embed_tokens = (
                self.model.model.model.embed_tokens.to(device)
            )
            self.model.model.model.rotary_emb = self.model.model.model.rotary_emb.to(
                device
            )
            logger.info(f"✅ [calibrate] Embedding layers moved to device: {device}")
        except Exception as e:
            logger.error(
                f"❌ [calibrate] Failed to move embedding layers to device: {e}"
            )
            raise

        # Confirm model has layers
        if not (
            hasattr(self.model, "model")
            and hasattr(self.model.model, "model")
            and hasattr(self.model.model.model, "layers")
        ):
            raise AttributeError(
                "Model must have nested .model.model.layers (e.g. for DeepSeek or Qwen2)"
            )
        else:
            logger.info(
                "✅ [calibrate] Model structure validated — using self.model.model.model.layers"
            )

        # * Generate calibration inputs

        # Step 1: Generate calibration inputs via init_quant
        try:
            self.modules, self.module_kwargs, self.inps = self.init_quant(
                n_samples=self.max_calib_samples,
                max_seq_len=self.max_calib_seq_len,
            )
            logger.info("✅ [calibrate] Calibration inputs initialized.")
        except Exception as e:
            logger.error(f"❌ [calibrate] Failed during init_quant(): {e}")
            raise

        all_scales: List[Tuple[str, torch.Tensor]] = []
        all_clips: List[Tuple[str, torch.Tensor]] = []

        assert self.modules is not None, "self.modules must be set before calibration"

        # Step 2: Iterate over each module
        # * 🔁 Outer Loop
        for i, module in enumerate(self.modules):
            logger.info(f"🔍 [calibrate] Processing module {i}/{len(self.modules)}")

            module = module.to(device)
            self.modules[i] = module

            # Get input features
            named_linears = get_named_linears(module)
            named_linears = exclude_layers_to_not_quantize(
                named_linears, self.modules_to_not_convert
            )
            input_feat = self._get_input_feat(module, named_linears)
            input_feat = {k: v.to(device) for k, v in input_feat.items()}
            logger.info(
                f"[calibrate] Input features for module {i} captured and moved to device."
            )

            # todo: debug; delete later
            if input_feat["self_attn.k_proj"].std() < 1e-6:
                logger.warning(
                    f"[Flat k_proj activation] Layer {i}, std={input_feat["self_attn.k_proj"].std:.6f}"
                )
            if input_feat["self_attn.v_proj"].std() < 1e-6:
                logger.warning(
                    f"[Flat v_proj activation] Layer {i}, std={input_feat["self_attn.v_proj"].std:.6f}"
                )

            # Collect scale & clip data for the module
            module_scales = []
            module_clips = []

            # * ✅ Iterate to run actual search for scale!
            # Get config from model for what to calibrate
            module_config = self.model.get_layers_for_scaling(
                module, input_feat, self.module_kwargs
            )

            # Process each layer for scales
            # * 🔁 Inner Loop
            for j, layer in enumerate(module_config):

                # Logging
                layer_name = layer.get("name", f"Layer_{j}")
                logger.info(
                    f"  🔹 [calibrate] Module {i}/{len(self.modules)} — Layer {j}/{len(module_config)}: {layer_name}"
                )
                logger.info(f"🧪 Start _search_best_scale on module {i}, layer {j}")

                start_scale = time.time()

                # * Search for scale
                # ☑️ Returned format: ("name", ("subname1", "subname2"), tensor)
                # -> ("post_attention_layernorm", ("mlp.gate_proj", "mlp.up_proj"), tensor([...]))
                scale_data = self._search_best_scale(module, **layer)

                # Log the returned scale data
                logger.debug(
                    f"Scale data received from _search_best_scale: {scale_data} (type: {type(scale_data)})"
                )

                if scale_data is not None:
                    # Append ScaleEntry to module_scales
                    module_scales.append(scale_data)

                    end_scale = time.time()
                    logger.info(
                        f"Scale search completed in {end_scale - start_scale:.2f} seconds"
                    )

                else:
                    logger.warning(
                        f"Scale data is None for layer {layer_name}. Skipping."
                    )
                    continue

            # Flatten data
            flattened_scales = flatten_scales_or_clip_list(module_scales)
            logger.debug(
                f"Step 1 - Flattened scales format: {flattened_scales[:3]}"
            )  # Show first 3 for quick inspection

            # and added prefix (layer path)
            prefixed_scales = append_str_prefix(
                flattened_scales, get_op_name(self.model, self.modules[i]) + "."
            )
            logger.debug(
                f"Step 2 - Prefixed scales format: {prefixed_scales[:3]}"
            )  # Show first 3 for quick inspection

            # Extend to all_scals
            all_scales.extend(prefixed_scales)
            logger.debug(
                f"Step 3 - Updated all_scales length: {len(all_scales)} | Sample entries: {all_scales[-3:]}"
            )  # Show last 3 entries added

            logger.debug(
                f"[Module {i}] Cumulative scales count so far: {len(all_scales)}"
            )

            # Optional clipping (Process clips per module)
            # * clipping is @ the block level only!
            if self.apply_clip:
                try:
                    clip_list = self._search_best_clip(
                        module, named_linears, input_feat
                    )
                    module_clips.extend(clip_list)
                except Exception as e:
                    logger.error(f"❌ Failed during clip update for module {i}: {e}")

            # Add prefix to clips
            prefixed_clips = append_str_prefix(
                module_clips, get_op_name(self.model, self.modules[i]) + "."
            )
            logger.debug(f"[Module {i}] Prefixed clips: {prefixed_clips[:3]}")

            # Extend all_clips with current module_clips
            all_clips.extend(prefixed_clips)
            logger.debug(
                f"[Module {i}] Cumulative clips count so far: {len(all_clips)}"
            )

            # Free up memory
            self.modules[i] = module.cpu()
            clear_memory()

        # Safe update with enhanced logging
        try:
            # Log sizes and sample entries for both scales and clips
            logger.debug(f"Final scales count: {len(all_scales)}")
            logger.debug(f"Final clips count: {len(all_clips)}")

            # Display a few sample entries for verification
            sample_scales = all_scales[:3] if len(all_scales) > 3 else all_scales
            sample_clips = all_clips[:3] if len(all_clips) > 3 else all_clips

            logger.debug(f"Sample scales entries: {sample_scales}")
            logger.debug(f"Sample clips entries: {sample_clips}")

            # Update the scales and clips lists
            safe_update(self.all_scales, all_scales, name="scales", strict=False)
            safe_update(self.all_clips, all_clips, name="clips", strict=True)

        except Exception as e:
            logger.error(f"❌ Failed during scale/clips update: {e}")
            raise

        # Save calibration results
        if save_path:
            try:
                torch.save({"scales": all_scales, "clips": all_clips}, save_path)
                logger.info(
                    f"💾 [calibrate] Calibration statistics saved to {save_path}"
                )
            except Exception as e:
                logger.error(f"❌ [calibrate] Failed to save stats: {e}")
                raise

        if clear_inps:
            try:
                del self.inps
                self.inps = None
                clear_memory()
                logger.info("✅ [calibrate] Cleared cached inputs.")
            except Exception as e:
                logger.warning(f"⚠️ [calibrate] Failed to clear inputs: {e}")

        elapsed = time.time() - start_time
        logger.info(
            f"🏁 [calibrate] Calibration completed in {time.strftime('%H:%M:%S', time.gmtime(elapsed))}"
        )

    def load_calibration_stats(self, calib_stats_path: str) -> None:
        """
        Load calibration statistics (scales and clips) from a .pt file and assign them
        to internal state (`self.all_scales` and `self.all_clips`).

        Args:
            calib_stats_path (str): Path to the saved 'calib_stats.pt' file. Expected content:
                - "scales": Dict[str, torch.Tensor] - Mapping of operation names to scale tensors.
                - "clips": Dict[str, torch.Tensor] (optional) - Mapping of operation names to
                clip tensors.

        Raises:
            ValueError: If 'scales' entry is missing or empty.

        Example:
            >>> quantizer.load_calibration_stats("/path/to/calib_stats.pt")

        Logging:
            - Logs the number of scales and clips loaded.
            - Provides a brief preview of tensor shapes, mean, and standard deviation.
        """
        stats = torch.load(calib_stats_path, map_location="cpu")
        all_scales: Dict[str, torch.Tensor] = stats.get("scales") or {}
        all_clips: Dict[str, torch.Tensor] = stats.get("clips") or {}

        # Check/error handling
        if not isinstance(all_scales, dict) or not all(
            isinstance(v, torch.Tensor) for v in all_scales.values()
        ):
            raise ValueError(
                "Scales data is not in the expected format (Dict[str, Tensor])."
            )

        if all_clips and not all(
            isinstance(v, torch.Tensor) for v in all_clips.values()
        ):
            raise ValueError(
                "Clip data is not in the expected format (Dict[str, Tensor])."
            )

        num_scales = len(all_scales)
        num_clips = len(all_clips)

        if num_scales == 0:
            logger.error(
                "❌ No scale entries found — calibration may have failed or file is corrupted."
            )
            raise ValueError("Empty or missing 'scales' in calibration stats.")

        logger.info(
            f"📥 Loaded calibration stats: {num_scales} scales, {num_clips} clips"
        )

        # * Update the model in the class
        self.all_scales = all_scales
        self.all_clips = all_clips

        # Preview utility
        def _preview_tensor_dict(
            name: str, tensor_dict: Dict[str, torch.Tensor], limit: int = 5
        ):
            for k, v in list(tensor_dict.items())[:limit]:
                logger.debug(
                    f" - {name} {k}: shape={tuple(v.shape)}, mean={v.mean():.4f}, std={v.std():.4f}"
                )

        logger.debug("📊 Previewing loaded scales:")
        _preview_tensor_dict("Scale", self.all_scales)

        if self.all_clips:
            logger.debug("📊 Previewing loaded clips:")
            _preview_tensor_dict("Clip", self.all_clips)
        else:
            logger.debug("No clip data found.")

    def compute_zeros(
        self, weight: torch.Tensor, scales: torch.Tensor, group_size: int
    ) -> torch.Tensor:
        """
        Compute zero-points dynamically from weight and scales.

        Zero-points are calculated per group based on the mean of weights in each group.

        Args:
            weight (torch.Tensor): The weight tensor to be quantized. Shape: [out_features,
                in_features].
            scales (torch.Tensor): Precomputed scales for each group. Shape: [num_groups].
            group_size (int): Number of elements per group for quantization.

        Returns:
            torch.Tensor: Computed zero-points per group. Shape: [num_groups].

        Example:
            >>> weight = torch.randn(128, 256)
            >>> scales = torch.ones(128 // 32)
            >>> zeros = quantizer.compute_zeros(weight, scales, group_size=32)

        Notes:
            - The zero-point calculation is based on the mean of weights per group and
            adjusted by the scale.
            - The calculation formula is: zero = floor(-mean / scale + 0.5).
        """
        num_groups = weight.shape[1] // group_size
        zeros = torch.empty(num_groups, device=weight.device)

        for i in range(num_groups):
            group_weights = weight[:, i * group_size : (i + 1) * group_size]
            scale = scales[i]
            zeros[i] = torch.floor(
                -group_weights.mean() / scale + 0.5
            )  # Assign per group

        return zeros

    def quant_with_calib_scales(self) -> None:
        """
        Apply quantization to all Linear layers in the model using the calibration scales.

        This method applies quantization using precomputed scales and dynamically calculated
        zero-points. The process involves iterating over all modules, identifying Linear
        layers, and replacing them with quantized versions.

        Steps:
            1. Validates calibration scales are available.
            2. Get named layers [Block1, Block2, ..., Block24]
            3. Iterates through all modules in the model (get_linear) and
            Applies quantization to each Linear layer based on the calibration scales.
            4. Logs timing and status for each layer.

            * Note:
            * get_linear outputs grabs Lower-Level Extraction of Linear Layers:
                {
                "self_attn.q_proj": LinearLayer1,
                "self_attn.k_proj": LinearLayer2,
                "self_attn.v_proj": LinearLayer3,
                "self_attn.o_proj": LinearLayer4,
                "mlp.gate_proj": LinearLayer5,
                "mlp.up_proj": LinearLayer6,
                "mlp.down_proj": LinearLayer7,
                }
        Raises:
            RuntimeError: If `self.modules` is not set.
            ValueError: If calibration scales (`self.all_scales`) are not loaded.

        Logging:
            - Logs the start and completion of the quantization process.
            - Provides per-layer timing for quantization.
            - Logs skipped layers due to missing scales or missing `weight` attribute.

        Example:
            >>> quantizer.quant_with_calib_scales()

        Notes:
            - Uses WQLinear_GEMM as the quantized linear layer implementation.
            - Skipped layers are logged for further inspection.
        """
        start_time = time.time()
        layer_times = {}
        skipped_count = 0

        # Step 1: Check & validate
        # Check layer
        if not self.modules:
            logger.info("Modules not initialized. Fetching model layers...")
            self.modules = self.get_model_layers(self.model)
            logger.info(f"Modules initialized. Total layers: {len(self.modules)}")

        # Check scales loaded properly
        if not hasattr(self, "all_scales") or not self.all_scales:
            raise ValueError(
                "❌ `self.all_scales` is not populated. Run calibration first."
            )

        # Step 2: Iterate over modules and apply quantization
        for module in self.modules:
            named_linears = get_named_linears(module)

            for name, linear_layer in named_linears.items():
                # Full path to the layer
                op_name = get_op_name(self.model, linear_layer)

                # Ensure layer has a weight attribute
                if not hasattr(linear_layer, "weight"):
                    logger.warning(
                        f"Skipping {op_name}: Layer has no `weight` attribute."
                    )
                    skipped_count += 1
                    continue

                # Timer start
                layer_start_time = time.time()

                # Ensure scales are available for this layer
                if op_name not in self.all_scales:
                    logger.warning(f"Skipping {op_name}: No scale found.")
                    skipped_count += 1
                    continue

                scales = self.all_scales[op_name]

                # Compute zeros dynamically
                zeros = self.compute_zeros(linear_layer.weight, scales, self.group_size)

                # Create quantized layer
                quantized_layer = WQLinear_GEMM.from_linear(
                    linear=linear_layer,
                    w_bit=self.w_bit,
                    group_size=self.group_size,
                    scales=scales,
                    zeros=zeros,
                )

                # Traverse to the parent module
                tokens = op_name.split(".")
                parent = self.model
                for token in tokens[:-1]:
                    parent = getattr(parent, token)

                # Replace the layer in the parent module
                setattr(parent, tokens[-1], quantized_layer)

                # Timer end and log time
                elapsed_time = time.time() - layer_start_time
                layer_times[op_name] = elapsed_time
                logger.info(f"⏱️ Quantized {op_name} in {elapsed_time:.4f} seconds.")

        # Step 3: Overall timing
        total_time = time.time() - start_time
        logger.info(f"✅ Quantization completed in {total_time:.4f} seconds.")

        # Summary of skipped layers
        if skipped_count > 0:
            logger.info(
                f"🔍 Skipped {skipped_count} layers due to missing scales or missing `weight` attribute."
            )
        else:
            logger.info("✅ No layers were skipped during quantization.")

        # Detailed timing report
        for op_name, elapsed_time in layer_times.items():
            logger.debug(f"Layer {op_name}: {elapsed_time:.4f} seconds")

    def save_quantized_and_configs(
        self,
        save_dir: str,
        safetensors: bool = True,
        shard_size: str = "5GB",
    ) -> None:
        """
        Save quantized model and its configuration, including related files such as processor,
        configuration, and quantized weights.

        Args:
            save_dir (str): Directory to save the model.
            safetensors (bool): Whether to use safetensors format (`.safetensors`),
                otherwise `.bin`.
            shard_size (str): Maximum shard size for large models.

        Example:
            >>> quantizer = ScroogeAwqQuantizer(
            >>> model=model,
            >>> quant_config=quant_config,
            >>> processor=processor,
                )
            >>> quantizer.save_quantized_and_related(save_dir="/path/to/save", safetensors=True)

        Files Persisted:
            - `config.json`: Model configuration, including quantization parameters and model
            architecture.
            - `generation_config.json`: Configuration for generation (e.g., sampling parameters).
            - `pytorch_model.bin` / `model.safetensors`: The quantized model weights in the specified
            format.
            - `processor_config.json` (if applicable): Processor configuration for
            vision/multi-modal models.
            - `scales.pt`: Calibration scales for each layer, used for dequantization.
            - `qzeros.pt`: Zero points for each quantized layer, used for dequantization.
            - `calib_stats.pt` (optional): Calibration statistics, useful for inspection and debugging.

        Why Each File is Necessary:
            - `config.json`: Required to initialize the model structure and quantization configuration
            during loading.
            - `generation_config.json`: Enables text generation using consistent generation parameters.
            - `pytorch_model.bin` / `model.safetensors`: Contains the quantized weights and is essential
            for inference.
            - `processor_config.json`: Ensures input preprocessing remains consistent for multi-modal
            models.
            - `scales.pt`: Enables scaling of quantized weights to ensure accurate dequantization.
            - `qzeros.pt`: Provides zero points for each quantized layer, adjusting quantized weights
            to their original range.
            - `calib_stats.pt`: Optional, but useful for validating calibration data or re-quantization.

        Note:
            * Saving both the entire model (e.g., model.bin) and its state_dict is a standard practice.
            - model.bin: Use for deployment, inference, and model sharing/transfer.
            - state_dict: Use for fine-tuning pre-trained models, transfer learning, and updating
            specific model weights.
        """
        # Check paths
        save_dir = save_dir.rstrip("/")
        os.makedirs(save_dir, exist_ok=True)

        # Save model
        class EmptyModule(nn.Module):
            def __init__(self):
                super(EmptyModule, self).__init__()

            def forward(self, x):
                return x

        # Save model config and generation config
        self.model.config.quantization_config = self.quant_config
        self.model.generation_config.do_sample = True

        # Save model with empty state dict
        logger.info(f"Saving model configuration to {save_dir}")
        try:
            self.model.save_pretrained(save_dir, state_dict=EmptyModule().state_dict())
        except Exception as e:
            logger.error(f"Failed to save model configuration: {e}")

        # Save processor if applicable
        if self.processor is not None:
            logger.info("Saving processor for vision models...")
            self.processor.save_pretrained(save_dir)

        # Remove unnecessary empty state dict files
        default_paths = [
            f"{save_dir}/model.safetensors",
            f"{save_dir}/pytorch_model.bin",
        ]
        for path in default_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logger.debug(f"Removed empty state dict file: {path}")
            except Exception as e:
                logger.warning(f"Failed to remove {path}: {e}")

        # Save actual quantized state dict
        logger.info(f"Saving quantized model weights to {save_dir}")
        try:
            save_torch_state_dict(
                state_dict=self.model.state_dict(),
                save_directory=save_dir,
                max_shard_size=shard_size,
                safe_serialization=safetensors,
                force_contiguous=True,
                shared_tensors_to_discard=getattr(self.model, "_tied_weights_keys", []),
            )
            logger.info(f"Quantized model saved successfully to {save_dir}")
        except Exception as e:
            logger.error(f"Error while saving quantized model: {e}")
