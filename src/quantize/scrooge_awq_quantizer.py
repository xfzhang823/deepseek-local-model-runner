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

from pathlib import Path
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
from awq.utils.module import (
    get_named_linears,
    get_op_name,
    get_op_by_name,
    set_op_by_name,
    exclude_layers_to_not_quantize,
    append_str_prefix,
)
from awq.utils.utils import get_best_device, clear_memory
from awq.utils.calib_data import get_calib_dataset
from awq.modules.linear.gemm import WQLinear_GEMM

# Project level
from utils.gpu_monitor import log_gpu_usage
from quantize.quantize_utils import (
    safe_update,
    unwrap_to_transformer,
    flatten_scales_or_clip_list,
    ScaleEntry,
    get_safe_parallel_sample_count,
    persist_quantized_layer,
    load_quantized_layers_into_model,
)
from quantize.scrooge_scale import (
    apply_scale,
    apply_clip,
    normalize_scales_across_group,
)  # import apply_scale from custom scale module

logger = logging.getLogger(__name__)
resource_logger = logging.getLogger("resource")

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
        save_dir: Optional[str] = None,
        save_per_layer: bool = True,
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
        self.n_parallel_calib_samples: int = n_parallel_calib_samples
        self.group_size: int = group_size  # Standard default for AWQ
        self.save_dir: Optional[str] = save_dir
        self.save_per_layer: bool = save_per_layer

        # Ensure save directory is created only if provided
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

        # Paths to store layer-wise quantized models if `save_per_layer` is True
        self.layer_paths: List[str] = [] if save_per_layer else []

        # ✅ Layer management
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

        logger.info(
            f"Initialized ScroogeAwqQuantizer with save_per_layer={self.save_per_layer}"
        )

        # ✅ Others:
        self.modules_to_not_convert = []  # empty = quantize everything
        self.max_chunk_memory = 64 * 1024 * 1024  # 64 MB
        # * standard default is 1024 MB (too large for most consumer laptop GPUs)
        # amount of memory allowed when chunking/calibrating activations for scale search.
        self.duo_scaling = False
        self.zero_point = False  # Set to symmetric
        self.w_bit: int = 4

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

        log_gpu_usage(
            "[init_quant] After moving layer 0 + embeddings to CPU"
        )  # vram logging
        clear_memory()  # Free up memory for next texts
        log_gpu_usage(
            "[init_quant] After torch.cuda.empty_cache() + gc.collect()"
        )  # vram logging

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
            embeddings (torch.Tensor): Precomputed input activations [n_samples, seq_len,
                hidden_dim].
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

    def init_calibration(self) -> None:
        """
        Initialize the calibration data including model layers, input activations, and kwargs.
        """
        logger.info("Initializing calibration inputs...")

        try:
            # Prepare calibration inputs using init_quant
            self.modules, self.module_kwargs, self.inps = self.init_quant(
                n_samples=self.max_calib_samples, max_seq_len=self.max_calib_seq_len
            )
            logger.info("Calibration inputs initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize calibration data: {e}")
            raise

    @override
    @torch.no_grad()
    def _search_best_scale(
        self,
        module: nn.Module,
        prev_op: nn.Module,
        layer: nn.Linear,
        inp: torch.Tensor,
        module2inspect: Optional[nn.Module] = None,
        kwargs: Optional[dict] = None,
    ) -> Tuple[str, Tuple[str], torch.Tensor]:
        """
        * Override official method to search per layer only!

        AWQ-style group-wise scale search for a single Linear layer.

        This method normalizes weights per group and computes scale values
        that balance quantized weight distribution and input activation dynamics.

        Args:
            module: Parent container (e.g., TransformerBlock).
            prev_op: Previous op (LayerNorm, GELU, etc.) used for applying scale.
            layer: The single Linear layer to calibrate.
            inp: Input tensor to the layer (typically float16).
            module2inspect: Defaults to `layer`, only used for forward().
            kwargs: Additional kwargs to pass to forward pass (e.g., attention mask).

        Returns:
            Tuple of:
            - Previous op name (str)
            - Target layer name (tuple of one str)
            - Computed scale tensor: [out_features] (float16 or bfloat16)

        * Mathematical Steps:
        - Grouped Weight Normalization:
            w_grouped = weight.view(O, G, group_size)
            w_scaled = |w_grouped| / max(|w_grouped|)  → [0, 1] within each group
            w_mean = mean(w_scaled, dim=-1) → [out_features, num_groups]
        - Input Mean (chunked, per group):
            x_mean = mean(abs(x[:, group]), dim=0).mean() → one value per group
        - Final:
            best_scales[:, g] = _compute_best_scale(x_group, w_mean[:, g], x_mean[g], ...)

        * WORKFLOW:
        1. Weight Grouping:
            - Split weights into groups of `group_size` along the input dimension.
            - Normalize each group:  w_scaled = |W| / max(|W|_group)
            - Compute mean(|W|_group) per output → `w_mean ∈ [O, G]`

        2. Input Mean Calculation:
            - Compute per-group input activation mean → `x_mean ∈ [G]`
            - Broadcast to all output channels → `x_mean_broadcasted ∈ [O, G]`

        3. Forward Pass:
            - Forward the full input through the inspected layer to obtain `fp16_output ∈ [B, O]`

        4. Optimize Scale:
            - For each group g:
                - Pass x_group and corresponding w_group into _compute_best_scale()
                - Get one scale per output channel: `s ∈ [O]`
                - Store into `scales[:, g]`

        ┌────────────────────────────────────────────────────┐
        │        Input: layer.weight ∈ [O, I]                │
        └────────────────────────────────────────────────────┘
                                │
                                ▼
        ┌────────────────────────────────────────────────────┐
        │  Step 1: Group and normalize weights               │
        │  w_scaled ∈ [O, G, group_size]                     │
        │  w_mean ∈ [O, G] ← mean(|w_grouped|)               │
        └────────────────────────────────────────────────────┘
                                │
                                ▼
        ┌────────────────────────────────────────────────────┐
        │  Step 2: Compute input means per group             │
        │  x_mean ∈ [G] → broadcast → [O, G]                 │
        └────────────────────────────────────────────────────┘
                                │
                                ▼
        ┌────────────────────────────────────────────────────┐
        │  Step 3: Forward pass (original fp16 layer)        │
        │  fp16_output ∈ [B, O]                              │
        └────────────────────────────────────────────────────┘
                                │
                                ▼
        ┌────────────────────────────────────────────────────┐
        │  Step 4: Optimize scales per group                 │
        │  scales[:, g] = _compute_best_scale(...) ∈ [O]     │
        └────────────────────────────────────────────────────┘

        """
        layer_name = get_op_name(module, layer)
        module2inspect = module2inspect or layer
        kwargs = kwargs or {}

        # --- STEP 1: Compute grouped weight mean ---
        try:
            weight = layer.weight.detach()  # create a copy of the weights
            out_features, in_features = weight.shape  # get dim (for reshaping)

            # if self.group_size <= 0:
            #     raise ValueError("group_size must be > 0 for group quantization.")

            if in_features % self.group_size != 0:
                raise ValueError(
                    f"[{layer_name}] in_features={in_features} not divisible by group_size={self.group_size}"
                )

            num_groups = in_features // self.group_size
            weight_grouped = weight.view(
                out_features, num_groups, self.group_size
            )  # * Reshape weights

            # Logging for debug
            logger.info(f"weight shape = {weight.shape}")
            logger.info(f"num_groups: {num_groups}")
            logger.info(f"[{layer_name}] weight_grouped shape: {weight_grouped.shape}")
            logger.info(
                f"[{layer_name}] weight_grouped preview [:3, 0, :5]: {weight_grouped[:3, 0, :5].tolist()}"
            )

            # todo: commented out; delete after debugging
            # # Group-wise normalized weights
            # w_max = weight_grouped.abs().amax(dim=-1, keepdim=True) + 1e-6
            # w_scaled = weight_grouped.abs() / w_max
            # w_mean = w_scaled.mean(
            #     dim=-1
            # )  #! shape: [O, G] (in Torch, row: output/col: input)
            # # w_mean: Average weight magnitude per output group (all groups)

            # logger.debug(f"[{layer_name}] w_mean.shape (group-wise) = {w_mean.shape}")

            # Clear weight_grouped to save memory
            clear_memory(weight_grouped)

        except Exception as e:
            logger.error(f"[{layer_name}] Error during weight scale computation: {e}")
            raise

        # --- STEP 2: Compute input mean (per input channel), chunked to avoid OOM
        try:
            inp_flat = inp.cpu().abs().view(-1, inp.shape[-1])
            num_elements = inp_flat.size(0)
            num_channels = inp_flat.size(1)
            chunk_size = min(
                self.max_chunk_memory // (inp_flat.element_size() * 2 * num_channels),
                num_elements,
            )

            x_sum = torch.zeros(num_channels, dtype=torch.float32)
            for i in range(0, num_elements, chunk_size):
                end = min(i + chunk_size, num_elements)
                x_sum += inp_flat[i:end].to(torch.float32).sum(dim=0)

                # todo: delete later
                # chunk_sum = inp_flat[i:end].to(torch.float32).sum(dim=0)
                # x_sum += chunk_sum.to("cpu")  # device-safe addition

            # todo: commented out b/c doing it in the loop; delete later
            # x_mean = (x_sum / num_elements).to(inp.dtype)
            # clear_memory()

            # # Project x_mean to output channels → then regroup
            # x_mean = x_mean.to(weight.device)  # Move to cpu for calculation

            # Project input activation mean into output channels
            x_mean_flat = (
                (x_sum / num_elements).to(inp.dtype).to(weight.device)
            )  # [in_features]
            x_mean_grouped = x_mean_flat.view(num_groups, self.group_size).mean(
                dim=1
            )  # [num_groups]
            x_mean_broadcasted = x_mean_grouped.expand(
                out_features, -1
            ).contiguous()  # [O, G]

            logger.info(
                f"[{layer_name}] x_mean_grouped shape = {x_mean_broadcasted.shape}"
            )
            logger.info(
                f"[{layer_name}] weight_grouped preview [:3, 0, :3]: {x_mean_broadcasted[:3, :3].tolist()}"
            )

            # todo: commented out; delete later
            # x_mean_flat = inp.cpu().abs().view(-1, inp.shape[-1])  # -> [*, in_features]
            # x_mean = x_mean_flat.mean(dim=0)  # -> [in_features]
            # x_mean_grouped_raw = x_mean.view(num_groups, self.group_size).mean(
            #     dim=1
            # )  # -> like [12]
            # x_mean_grouped = x_mean_grouped_raw.expand(
            #     out_features, -1
            # ).contiguous()  # -> regroup to 2-dim, like [1536, 12]
            # # x_mean_grouped: Approximate average input signal per output group (all groups)

            # logger.debug(
            #     f"[{layer_name}] x_mean.shape (group-wise) = {x_mean_grouped.shape}"
            # )

            clear_memory(x_sum)

        except Exception as e:
            logger.error(f"[{layer_name}] Error during input mean computation: {e}")
            raise

        log_gpu_usage("[fp16_output] BEFORE forward pass")  # ⬅️ VRAM logging

        # --- STEP 3: Forward pass for FP16 output ---
        try:
            module_kwargs = self._sanitize_kwargs(kwargs, module2inspect)
            fp16_output = self._module_forward(inp, module2inspect, module_kwargs)

            log_gpu_usage("[fp16_output] AFTER forward pass")  # ⬅️ VRAM logging

            fp16_output = fp16_output.clip(
                torch.finfo(fp16_output.dtype).min,
                torch.finfo(fp16_output.dtype).max,
            )
            fp16_output = fp16_output.to("cpu")
            torch.cuda.empty_cache()

            # VRAM logging
            log_gpu_usage("[DEBUG] After fp16_output moved + cleared")  # ⬅️ VRAM logging
            resource_logger.debug(
                f"Is model still on GPU? {[n.device for n in module2inspect.parameters()]}"
            )
            resource_logger.debug(f"Is inp still on GPU? {inp.device}")

        except Exception as e:
            logger.error(f"[{layer_name}] Forward pass failed: {e}")
            raise

        # --- STEP 4: Optimize scale using loss search ---
        # Compute best scale via grid search

        # Prepare the final scale tensor
        scales = torch.zeros(
            (layer.out_features, num_groups), dtype=torch.float16
        )  # setup holder
        self.duo_scaling = True  # ! include both w_mean and x_mean

        # iterate over each group
        for g in range(num_groups):
            start, end = g * self.group_size, (g + 1) * self.group_size

            # Slice inputs and weights for the group
            x_per_group = inp[
                :, :, start:end
            ]  # ☑️ slicing in_features[B, sequence, group_size]

            # todo: debug; delete later
            if torch.isnan(x_per_group).any():
                logger.error(
                    f"[{layer_name}] Group {g} input (x_per_group) contains NaNs!"
                )
                raise ValueError("NaNs in input tensor during calibration.")
            logger.info(f"x_per_group shape == {x_per_group.shape}")

            w_per_group = layer.weight[:, start:end]  # i.e., [1536, 128]

            # Slice x_mean and compute group mean per input feature group
            # x_mean = x_group.abs().mean(dim=0)  # [B, group_size]
            # x_mean_grouped = x_mean.mean().expand(layer.out_features)  # [O, group_size]

            # Group-wise input activation average, broadcast per output channel
            x_per_group_mean = (
                x_per_group.abs().mean(dim=0).mean().expand(out_features)
            )  # [O]
            w_per_group_max = w_per_group.abs().amax(dim=1, keepdim=True) + 1e-6
            w_per_group_scaled = w_per_group.abs() / w_per_group_max
            w_per_group_mean = w_per_group_scaled.mean(dim=1)  # [O]

            # Log VRAM utilization
            log_gpu_usage(f"[{layer_name}] Group {g}: before scale search")

            # Call your per-group scale search
            best_scale_per_group = self._compute_best_scale_groupwise(
                x=inp,  # full input
                w_mean=w_per_group_mean,  # per group only
                x_mean=x_per_group_mean,  # per group only
                module2inspect=layer,  # full layer (layer for pass forward)
                linears2scale=[layer],  # full layer (layer to calib scales)
                fp16_output=fp16_output,  # full layer output
                group_idx=g,
                group_size=self.group_size,
                kwargs=module_kwargs,
            )

            # Log VRAM utilization
            log_gpu_usage(f"[{layer_name}] Group {g}: after scale search")

            scales[:, g] = best_scale_per_group
            logger.debug(
                f"[{layer_name}] Group {g}: shape={best_scale_per_group.shape}, "
                f"mean={best_scale_per_group.mean():.4f}, std={best_scale_per_group.std():.4f}"
            )
            logger.debug(
                f"[{layer_name}] Group {g} scale preview: {best_scale_per_group[:5].tolist()}"
            )

        # Logging and shape check
        logger.debug(
            f"[{get_op_name(module, layer)}] best_scales shape: {scales.shape}"
        )
        logger.debug(
            f"[{get_op_name(module, layer)}] best_scales preview:\n{scales[:3, :]}"
        )

        assert scales.shape == (out_features, num_groups), (
            f"[{layer_name}] Expected best_scales shape {[out_features, num_groups]}, "
            f"got {scales.shape}"
        )

        del inp, fp16_output
        clear_memory()

        return get_op_name(module, prev_op), (layer_name,), scales

    def _compute_best_scale_groupwise(
        self,
        x: torch.Tensor,
        w_mean: torch.Tensor,
        x_mean: torch.Tensor,
        module2inspect: torch.nn.Module,
        linears2scale: List[nn.Linear],
        fp16_output: torch.Tensor,
        group_idx: int,  # ✔️ Added to autoawq code
        group_size: int,  # ✔️ Added to autoawq code
        kwargs: Dict = {},
    ):
        """
        * Replace official autoawq's _compute_best_scale method to accomodate
        * per group calculation efficiently.

        Compute loss and select best scales

        L(s) = || Q(W * s) (s^-1 * X) - W * X ||
        Q: weight quantization function | pseudo_quantize_tensor(W * s)
        X: inputs from calib dataset    | X
        W: original weights in FP16     | layer
        s: per channel scaling factor   | s^-1 * X

        Grid search to find the best quantization scale for a specific input group.

        This method evaluates candidate scales for a single input group (specified
        by `group_idx`) by temporarily modifying the corresponding slice of weight
        matrices in `linears2scale`. The modified layer(s) are then used in a forward
        pass to compute the quantized output, which is compared against the original
        FP16 output to compute reconstruction loss.

        Args:
            x (torch.Tensor): Full input tensor to the layer (shape: [B, in_features]).
            w_mean (torch.Tensor): Per-output-channel mean of normalized weights
                for this group (shape: [out_features]).
            x_mean (torch.Tensor): Scalar input mean for this group, broadcasted
                per output channel (shape: [out_features]).
            module2inspect (nn.Module): The module to forward for computing
                the quantized output.
                Typically a single Linear layer, but can also be a higher-level container.
            linears2scale (List[nn.Linear]): List of Linear layers in which the scale
                should be applied.
                Usually contains a single layer.
            fp16_output (torch.Tensor): Original output of the unquantized full layer
                (shape: [B, out_features]), used as the target for loss comparison.
            group_idx (int): Index of the input group currently being quantized.
            group_size (int): Number of input dimensions in each group.
            kwargs (Dict, optional): Additional keyword arguments passed to the module's
                forward method (e.g., attention masks). Defaults to empty.

        Returns:
            torch.Tensor: Optimal scale vector for this group, per output channel
            (shape: [out_features]).
        """
        start = group_idx * group_size
        end = (group_idx + 1) * group_size

        n_grid = 20
        history = []
        best_ratio = -1
        best_scales = None
        best_error = float("inf")

        org_sd = {k: v.cpu() for k, v in module2inspect.state_dict().items()}

        device = x.device
        x_mean = x_mean.view(-1).to(device)
        w_mean = w_mean.view(-1).to(device)

        assert not torch.isnan(w_mean).any(), "w_mean contains NaNs"  # extra guard
        assert not torch.isnan(x_mean).any(), "x_mean contains NaNs"  # extra guard

        for ratio in range(n_grid):
            # create new scales
            ratio = ratio / n_grid

            # NOTE: s^-1 * x is fused here, according to paper
            if self.duo_scaling:
                scales = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp(
                    min=1e-4
                )
            else:
                scales = x_mean.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()
            scales_view = scales.view(-1, 1).to(
                device
            )  # ☑️ updated: flip dim to broadcast across num of groups input dims

            # avoid scaling values that overflow
            scales[torch.isinf(scales)] = 1
            scales[torch.isnan(scales)] = 1

            # Q(W * s)
            for fc in linears2scale:
                fc.weight[:, start:end].mul_(scales_view)  # ☑️ updated to slice weights
                fc.weight.data[:, start:end] = (
                    self.pseudo_quantize_tensor(fc.weight.data[:, start:end])[0]
                    / scales_view
                )  # ☑️ updated to slice weights
                logger.debug(f"Group {group_idx}: scale shape = {scales_view.shape}")

            # W * X
            int_w_output = self._module_forward(x, module2inspect, kwargs)
            int_w_output = int_w_output.clip(
                torch.finfo(int_w_output.dtype).min, torch.finfo(int_w_output.dtype).max
            )  # clamp
            int_w_output = int_w_output.to(
                "cpu"
            )  # ☑️ Added to bring to same device as fp16_output

            # compute mean squared error (L2 norm)
            loss = self._compute_loss(fp16_output, int_w_output, device)

            history.append(loss)
            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                best_scales = scales.clone()
                logger.debug(
                    f"[Group {group_idx}] New best ratio = {ratio:.3f}, loss = {loss:.6f}"
                )
            module2inspect.load_state_dict(org_sd)

        if best_ratio == -1:
            logger.error(f"No valid scale found for group. Loss history: {history}")
            raise Exception

        assert best_scales is not None  # Add extra guard
        assert torch.isnan(best_scales).sum() == 0, best_scales

        return best_scales.detach().cpu()

    @override
    @torch.no_grad()
    def _search_best_clip(
        self,
        layer: nn.Module,
        named_linears: Dict[str, nn.Linear],
        input_feat: torch.Tensor,
    ) -> Tuple[str, Optional[float]]:
        """
        AWQ-style clipping threshold search for a single Linear layer.

        This computes the max absolute activation value (clipping range)
        that minimizes quantization loss, unless the layer is excluded
        from clipping by name.

        Args:
            layer (nn.Module): Linear layer being calibrated.
            named_linears (Dict[str, nn.Linear]): Mapping of layer names for lookups.
            input_feat (torch.Tensor): Input activation for the given layer.

        ---
        🧠 Workflow:

            Input:
                - weight ∈ [out_features, in_features]
                - input_feat ∈ [batch_size, in_features]

            ┌────────────────────────────────────────────┐
            │ Step 1: Layer Name Filter                  │
            │  - Skip known layers like q_proj, k_proj   │
            └────────────────────────────────────────────┘
                        │
                        ▼
            ┌────────────────────────────────────────────┐
            │ Step 2: Move to Device                     │
            │  - Prepare layer for forward operation     │
            └────────────────────────────────────────────┘
                        │
                        ▼
            ┌────────────────────────────────────────────┐
            │ Step 3: Compute Clipping Threshold         │
            │  - Use input_feat and weight               │
            │  - Output: max_val_tensor ∈ scalar (tensor)│
            └────────────────────────────────────────────┘
                        │
                        ▼
            ┌────────────────────────────────────────────┐
            │ Step 4: Convert and Return                 │
            │  - Return (layer_name, float threshold)    │
            └────────────────────────────────────────────┘

        ---
        Returns:
            Tuple[str, Optional[float]]:
                - Layer name as string
                - Float clipping threshold or None (if skipped)
        """
        layer_name = get_op_name(self.model, layer)

        # Heuristically skip clipping for attention Q/K heads
        avoid_clipping_tags = ["q_", "k_", "query", "key", "Wqkv"]
        if any(tag in layer_name for tag in avoid_clipping_tags):
            logger.info(
                f"🛑 Skipping clipping for {layer_name} (matches exclusion list)."
            )
            return layer_name, None

        try:
            layer = layer.to(get_best_device())
            logger.debug(
                f"[{layer_name}] moved to {layer.weight.device} for clipping calc."
            )

            max_val_tensor = self._compute_best_clip(layer.weight, input_feat)
            max_val: float = max_val_tensor.item()

            logger.info(f"[{layer_name}] best clip value = {max_val:.4f}")
            return layer_name, max_val

        except Exception as e:
            logger.error(f"❌ Error computing clip for {layer_name}: {e}")
            return layer_name, None

        finally:
            layer.cpu()
            logger.debug(f"[{layer_name}] moved back to CPU after clip search.")

    @torch.no_grad()
    def compute_zeros(
        self,
        weight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
    ) -> torch.Tensor:
        """
        * Custom function replace official code's pseudo_quantize_tensor method's
        * compute zeros functionality

        Compute zero-points for symmetric group-wise quantization.

        Each group of weights is assigned a zero-point based on the mean value
        of the weights in that group. This helps shift the quantized representation
        toward a centered range, reducing quantization error.

        ---
        🧠 Formula:
            zero = floor(-mean(weight_group) / scale + 0.5)

        ---
        📉 Workflow

            weight ∈ [out_features, in_features]
                            │
                            ▼
            ┌──────────────────────────────┐
            │ Group along input dim        │
            │ → groups = in_features // G  │
            └──────────────────────────────┘
                            │
                            ▼
            ┌────────────────────────────────┐
            │ For each group i:              │
            │   W_i = weight[:, i*G:(i+1)*G] │
            │   μ_i = mean(W_i)              │
            │   z_i = floor(-μ_i / s_i + 0.5)│
            └────────────────────────────────┘
                            │
                            ▼
            Output: zeros ∈ [num_groups]

        ---
        Args:
            weight (torch.Tensor): The full linear layer weight [O, I]
            scales (torch.Tensor): Precomputed scales per group [num_groups]
            group_size (int): Number of input dims per quant group

        Returns:
            torch.Tensor: Zero-points per group [num_groups]
        """
        out_features, in_features = weight.shape
        if in_features % group_size != 0:
            raise ValueError(
                f"Invalid group size: in_features={in_features} is not divisible by group_size={group_size}"
            )

        num_groups = in_features // group_size
        zeros = torch.empty(num_groups, device=weight.device)

        for i in range(num_groups):
            group_weights = weight[:, i * group_size : (i + 1) * group_size]
            group_mean = group_weights.mean()
            scale = scales[i]
            zero_point = torch.floor(-group_mean / scale + 0.5)
            zeros[i] = zero_point

            logger.debug(
                f"[Group {i}] mean={group_mean.item():.6f}, scale={scale.item():.6f}, zero={zero_point.item():.2f}"
            )

        logger.info(f"✅ Computed zero-points: shape={zeros.shape}")
        return zeros

    def calibrate_and_quantize(
        self, save_dir_path: Optional[str | Path] = None
    ) -> None:
        """
        Runs full Scrooge-style calibration and quantization in a single pass.

        This method processes the model block by block, performing all necessary steps:
            1. Captures input activations per block.
            2. Computes optimal per-channel scales (per layer).
            3. Normalizes group-wise scales to share a consistent quantization range.
            4. Applies those scales to weights (and optionally to activations).
            5. Computes zero-points and (optionally) clipping thresholds.
            6. Replaces original Linear layers with quantized WQLinear_GEMM modules.
            7. Saves quantized weights and configs if `save_dir_path` is specified.

        Key Features:
            - Activations are captured and discarded per layer to minimize VRAM usage.
            - Input activations (`self.inps`) are automatically cleared after use.
            - Scales and clips are tracked internally in `self.all_scales` and
            `self.all_clips`.

        Args:
            save_dir_path (Optional[str | Path]):
                If provided, saves quantized model weights and metadata to this path.

        Raises:
            RuntimeError: If `self.modules` is not initialized.
            ValueError: If no calibration scales are generated.

        Logging:
            - Logs per-layer progress and timing.
            - Logs number of scale and clip entries computed.
            - Logs total elapsed time for the full pipeline.
        """
        start_time = time.time()
        logger.info("🚀 [calibrate] Starting full quantization pass")

        self.init_calibration()
        self.model = self.model.to(device)

        if self.modules is None or self.inps is None:
            raise RuntimeError("Calibration data (modules or inputs) not initialized.")

        total_layers_quantized = 0
        for idx, module in enumerate(self.modules):  # * ☑️ Outer loop
            logger.info(
                f"\n🔍 [Block {idx}/{len(self.modules)}] Processing {module.__class__.__name__}"
            )
            module = module.to(device)
            self.modules[idx] = module

            named_linears = exclude_layers_to_not_quantize(
                get_named_linears(module), self.modules_to_not_convert
            )
            name_to_layer = {v: k for k, v in named_linears.items()}

            input_feat = self._get_input_feat(module, named_linears)
            input_feat = {k: v.to(device) for k, v in input_feat.items()}
            logger.debug(f"[Block {idx}] Input features captured.")

            module_config = self.awq_model.get_layers_for_scaling(
                module, input_feat, self.module_kwargs
            )
            logger.debug(f"[Block {idx}] Groups = {len(module_config)}")

            for group in module_config:
                prev_op = group["prev_op"]
                layers = group["layers"]
                group_name = prev_op.__class__.__name__

                logger.info(f"\n⚙️  [Group: {group_name}] {len(layers)} layers")

                # 1. Compute scales per layer
                scales_dict = {}
                for layer in layers:
                    layer_name = name_to_layer.get(layer)
                    if layer_name is None:
                        raise ValueError(f"Layer not found in name mapping: {layer}")
                    logger.info(f"🔎 [scale] {layer_name}")

                    op_name, _, best_scales = self._search_best_scale(
                        module=module,
                        prev_op=prev_op,
                        layer=layer,
                        inp=input_feat[layer_name],
                    )

                    best_scales = best_scales.to(device)
                    scales_dict[layer_name] = best_scales
                    logger.debug(
                        f"[raw scales] {layer_name} → shape: {best_scales.shape}, preview: {best_scales[:5].tolist()}"
                    )

                # 2. Normalize scales across the group
                normalized_scales = normalize_scales_across_group(
                    layers=layers,
                    name_to_layer=name_to_layer,
                    scales_dict=scales_dict,
                )
                for name, scale_tensor in list(normalized_scales.items()):
                    logger.debug(
                        f"[normalized scales] {name} → shape: {scale_tensor.shape}, preview: {scale_tensor[:5].tolist()}"
                    )

                # 3. Apply scale, compute zeros, clip, quantize
                for layer in layers:
                    layer_name = name_to_layer[layer]

                    # Apply scales
                    scales = normalized_scales[layer_name]
                    logger.info(f"🧪 [apply] {layer_name}")

                    apply_scale(
                        module, [(get_op_name(module, prev_op), (layer_name,), scales)]
                    )

                    # Apply clips
                    if self.apply_clip:
                        # Optional clipping
                        input_feat_tensor = input_feat.get(layer_name) or torch.empty(0)
                        clip_name, clip_value = self._search_best_clip(
                            layer=layer,
                            named_linears=named_linears,
                            input_feat=input_feat_tensor,
                        )
                        if clip_value is not None:
                            apply_clip(module, clip_name, clip_value)
                            logger.info(f"[clip] {clip_name} ← {clip_value:.4f}")
                        else:
                            logger.debug(f"[clip] {clip_name} skipped")

                    # Compute zeros
                    zeros = self.compute_zeros(layer.weight, scales, self.group_size)
                    logger.debug(
                        f"[zeros] {layer_name} → {zeros.shape}, first: {zeros[:5].tolist()}"
                    )

                    # Replace sub-layer with quantized GEMM layer
                    quantized_layer = WQLinear_GEMM.from_linear(
                        linear=layer,
                        w_bit=self.w_bit,
                        group_size=self.group_size,
                        scales=scales,
                        zeros=zeros,
                    )

                    set_op_by_name(
                        layer=module, name=layer_name, new_module=quantized_layer
                    )
                    logger.info(f"✅ Quantized {layer_name} → WQLinear_GEMM")

                    if save_dir_path:
                        persist_quantized_layer(
                            quantized_layer, save_dir=str(save_dir_path)
                        )
                        logger.debug(f"💾 Saved {layer_name} to {save_dir_path}")

                # Count loops
                total_layers_quantized += len(layers)

        logger.info(f"🔢 Total quantized layers: {total_layers_quantized}")
        elapsed = time.time() - start_time
        logger.info(f"\n🏁 Finished calibration + quantization in {elapsed:.2f} sec")

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

    def quantize_and_save_layer(
        self,
        layer: nn.Module,
        layer_name: str,
        scales: torch.Tensor,
        zeros: torch.Tensor,
    ) -> nn.Module:
        """
        Process a single layer: quantize and optionally save to disk.

        Args:
            layer (nn.Module): The original linear layer to be quantized.
            layer_name (str): The name identifier for the layer.
            scales (torch.Tensor): Precomputed scales for the layer.
            zeros (torch.Tensor): Precomputed zero points for the layer.
        """
        # Quantize the layer
        quantized_layer = self.quantize_layer(layer, scales, zeros)

        # Save the entire layer with all data (scales, qweight, qzeros)
        if self.save_per_layer and self.save_dir:
            save_path = os.path.join(self.save_dir, f"{layer_name}_quant.pt")
            torch.save(quantized_layer.state_dict(), save_path)
            self.layer_paths.append(save_path)
            logger.info(f"Saved quantized layer {layer_name} to {save_path}")

        return quantized_layer

    def quantize_layer(
        self, layer: nn.Module, scales: torch.Tensor, zeros: torch.Tensor
    ) -> nn.Module:
        """
        Convert a given linear layer to its quantized version using the WQLinear_GEMM class.

        Args:
            layer (nn.Module): The original linear layer to be quantized.
            scales (torch.Tensor): Precomputed scales for the layer.
            zeros (torch.Tensor): Precomputed zero points for the layer.

        Returns:
            nn.Module: The quantized version of the input layer.
        """
        # Convert to quantized linear layer
        q_linear = WQLinear_GEMM.from_linear(
            linear=layer,
            w_bit=self.w_bit,
            group_size=self.group_size,
            scales=scales,
            zeros=zeros,
        )
        return q_linear

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
