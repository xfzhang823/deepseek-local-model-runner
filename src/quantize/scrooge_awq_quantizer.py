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
| For each module (layer) in
elf.modules:       |
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
import gc
import json
from typing import Any, cast, Optional, List, Dict, Tuple, Union
from typing_extensions import override
from collections import defaultdict
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from huggingface_hub import snapshot_download, save_torch_state_dict

# autoawq modules
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
from awq.utils.calib_data import get_calib_dataset  # Official version - w/o sent aware

from awq.modules.linear.gemm import WQLinear_GEMM

# Project level modules
from utils.gpu_monitor import log_gpu_usage
from utils.offload_to_cpu import offload_tensor_to_cpu
from utils.check_existing import are_block_sublayers_quantized
from utils.calib_data import get_calib_dataset_sent_aware
from quantize.quantize_utils import (
    unwrap_to_transformer,
    flatten_scales_or_clip_list,
    get_safe_parallel_sample_count,
    get_scale_for_zero_point,
    persist_quantized_layer,
    load_quantized_layers_into_model,
    move_rope_to_device,
    move_module_to_device,
    clear_up_module_memory,
    forward_with_memory_chunking,
    tensor_preview,
)
from quantize.scrooge_scale import (
    apply_scale_all_groups,
    apply_clip,
    normalize_scales_across_groups,
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
        save_per_layer: bool = True,
        version: str = "gemm",
        modules_to_not_convert: List[str] = [],  #! include ALL layers!
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
        self.save_per_layer: bool = save_per_layer
        self.version: str = version

        # Paths to store layer-wise quantized models if `save_per_layer` is True
        self.layer_paths: List[str] = [] if save_per_layer else []

        # ✅ Layer management
        self.modules: Optional[List[nn.Module]] = None
        self.module_kwargs: Optional[Dict[str, Any]] = None
        self.inps: Optional[torch.Tensor] = None

        # * Calibration device (gpu/cpu) (calibration the most computationally expensive)
        self.full_gpu = False  # set to False
        self.hybrid_mode = True  # default to hybrid

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
        self.modules_to_not_convert: list[str] = (
            modules_to_not_convert  # ☑️ include ALL layers!
        )
        self.max_chunk_memory = 64 * 1024 * 1024  # 64 MB

        # * standard default is 1024 MB (too large for most consumer laptop GPUs)
        # amount of memory allowed when chunking/calibrating activations for scale search.
        self.duo_scaling = True  #! ALWAYS set to True
        self.zero_point = True  #! Always Set to True (asymmetric)
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

            if hasattr(transformer.rotary_emb, "cos_cached"):
                transformer.rotary_emb.cos_cached = (
                    transformer.rotary_emb.cos_cached.to(device)
                )
            if hasattr(transformer.rotary_emb, "sin_cached"):
                transformer.rotary_emb.sin_cached = (
                    transformer.rotary_emb.sin_cached.to(device)
                )
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

        #! Commented out: use sent aware only if dataset is large enough (and big VRAM to handle it)
        # # 2. Load calibration samples (Use autoawq's default datase - pileval)
        # try:
        #     samples = get_calib_dataset_sent_aware(
        #         data=self.calib_data or "pileval",  # default to pileval
        #         tokenizer=self.tokenizer,
        #         n_samples=n_samples,
        #         max_seq_len=max_seq_len,
        #         split=self.split,
        #         text_column=self.text_column,
        #     )
        #     logger.info(f"✅ [init_quant] Loaded calibration dataset: pileval")
        # except Exception as e:
        #     logger.warning(f"⚠️ [init_quant] Failed to load pileval: {e}")
        #     logger.info(f"🔄 [init_quant] Falling back to c4...")
        #     samples = get_calib_dataset_sent_aware(
        #         data="c4",
        #         tokenizer=self.tokenizer,
        #         n_samples=n_samples,
        #         max_seq_len=max_seq_len,
        #         split="validation",
        #         text_column="text",
        #     )
        #     logger.info("✅ Loaded fallback calibration dataset: c4")

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
            log_gpu_usage("[init_quant] just before catcher forward.")  # log VRAM
            self.model(samples.to(next(self.model.parameters()).device))
            log_gpu_usage("[init_quant] AFTER catcher forward")  # log VRAM

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

        # * 🚨 Explicit cleanup of memory-holding calibration tensors
        if "samples" in locals():
            del samples

        # commented out: very small/no need to remove this
        # if "attention_mask" in layer_kwargs:
        #     del layer_kwargs["attention_mask"]

        gc.collect()  # triggers immediate cleanup

        clear_memory()  # Free up memory for next texts
        log_gpu_usage(
            "[init_quant] After torch.cuda.empty_cache() + gc.collect()"
        )  # vram logging

        # Check if all layers are moved
        for i, layer in enumerate(modules):
            device_types = {p.device.type for p in layer.parameters()}
            logger.info(f"[init_quant] Layer {i} devices: {device_types}")

        return modules, layer_kwargs, inps[0]

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
    ) -> torch.Tensor:
        """
        * Override official autoawq method to search per layer only
        * (logic fully preserved)

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
        Steps:
        1. Grouped Weight Normalization:
            - Group weight matrix into [O, G, group_size]
            - Normalize each group by its max absolute value
            - Compute mean across input dim: w_mean ∈ [O]

        2. Input Mean Calculation (per group):
            - Compute input mean per group: x_mean ∈ [G]
            - Broadcast to each output channel: x_mean ∈ [O]

        3. Forward Pass:
            - Run full-precision forward to get fp16_output ∈ [B, O]

        4. Optimize Scale:
            - For each group g:
                - Call _compute_best_scale_groupwise(...)
                - Output: scalar scale ∈ ℝ
                - Store into scales[g]

        ┌────────────────────────────────────────────────────┐
        │        Input: layer.weight ∈ [O, I]                │
        └────────────────────────────────────────────────────┘
                        │
                        ▼
        ┌────────────────────────────────────────────────────┐
        │  Step 1: Group weights by input dim                │
        │  weight_grouped ∈ [O, G, group_size]               │
        │  w_group_mean ∈ [group_size] per group             │
        └────────────────────────────────────────────────────┘
                        │
                        ▼
        ┌────────────────────────────────────────────────────┐
        │  Step 2: Compute x_mean per group ∈ [group_size]   │
        │  (one per group, no expansion to [O])              │
        └────────────────────────────────────────────────────┘
                        │
                        ▼
        ┌────────────────────────────────────────────────────┐
        │  Step 3: Forward pass                              │
        │  fp16_output ∈ [B, O]                              │
        └────────────────────────────────────────────────────┘
                        │
                        ▼
        ┌────────────────────────────────────────────────────┐
        │  Step 4: Optimize scalar scale per group           │
        │  scales ∈ [G] (1 value per group)                  │
        └────────────────────────────────────────────────────┘


        * Note 1: Why AWQ uses abs() in calcuating means of activation (x_mean)
        Compute group-wise scale signal balancing input strength (x) and weight structure (w).
        Scales are derived by combining input magnitude (x_mean) and normalized weight mean (w_mean),
        capturing group influence and entropy.
        Larger x or flatter w (low entropy) lead to larger scales, enabling coarser quantization
        where precision matters less.

        # ! Note 2: Do not confuse these activation scales with the final quantization scales
        # ! that are stored in the quantized model!
        #
        # The "scales" computed here are **activation (preconditioning) scales**. They are used
        # only to modify (precondition) the floating-point weights before quantization. These
        # activation scales are *not* the same as the quantization scales embedded in the final
        # model or used at inference time.
        #
        # The workflow:
        #   1. Activation scales are searched and used to rescale (precondition) the weights.
        #   2. The rescaled weights are then quantized (qweight, zero points, quantization scales).
        #   3. The quantization scales and zero points (derived from the preconditioned weights)
        #      are what get stored and used in the quantized model.
        #
        # In summary: The activation scales "condition" the weights to improve quantization.
        # They do not become the final quantization parameters themselves.
        """
        layer_name = get_op_name(module, layer)
        module2inspect = module2inspect or layer
        kwargs = kwargs or {}

        # 🟩 Setup dual device config
        if self.full_gpu:
            forward_device = calibration_device = torch.device("cuda")  # ✅ All on GPU
        elif self.hybrid_mode:
            forward_device = torch.device("cuda")
            calibration_device = torch.device("cpu")  # ⚖️ Forward on GPU, stats on CPU
        else:
            forward_device = calibration_device = torch.device(
                "cpu"
            )  # 🐢 Fallback: full CPU

        logger.info(
            f"💡 Quant forward_device: {forward_device}, calibration_device: {calibration_device}"
        )

        # 🟩 Ensure layer is on forward device
        layer.to(forward_device)

        # * --- STEP 1: Load in weights and prelimenary checks ---
        try:
            weight = layer.weight.detach()  # create a copy of the weights
            out_features, in_features = weight.shape  # get dim (for reshaping)

            # Check if layer dim is divisible by group size
            if in_features % self.group_size != 0:
                raise ValueError(
                    f"[{layer_name}] in_features={in_features} not divisible by group_size={self.group_size}"
                )

            num_groups = in_features // self.group_size

            # Logging for debug
            logger.info(
                f"Original weight shape [out_feature, in_feature] = {weight.shape}"
            )
            logger.info(f"num_groups: {num_groups}")

        except Exception as e:
            logger.error(f"[{layer_name}] Error during weight scale computation: {e}")
            raise

        # * --- STEP 2: Reshape weights to [O, G, group_size]
        try:
            # Imitating a slighly older version of the autoawq code (logic still the same)
            weight_grouped = weight.view(
                out_features, num_groups, self.group_size
            )  # [O, G, group_size]

            # Normalize weights per group
            w_max = weight_grouped.abs().amax(dim=2, keepdim=True) + 1e-6  # [O, G, 1]
            w_scaled = weight_grouped.abs() / w_max  # [O, G, group_size]

            # Reshape back to original shape
            w_scaled = w_scaled.view(out_features, in_features)  # [O, I]

            # Compute mean across input channels in each group
            w_mean = w_scaled.mean(dim=0)  # [I]

            # Transpose
            w_mean = w_mean.view(-1, self.group_size)  # [G, group_size]

        except Exception as e:
            logger.error(f"[{layer_name}] Error during w_mean computation: {e}")
            raise

        # * --- STEP 3: Compute input mean (per input channel), chunked to avoid OOM
        try:
            # ✅ run on calibration_device
            inp_flat = (
                inp.to(calibration_device).abs().view(-1, inp.shape[-1])
            )  # Flatten B and T -> [B*T, I]
            num_elements = inp_flat.size(0)  # rows (B*T) (all tokens)
            num_channels = inp_flat.size(1)  # column (input channel)
            chunk_size = min(
                self.max_chunk_memory // (inp_flat.element_size() * 2 * num_channels),
                num_elements,
            )

            x_sum = torch.zeros(
                num_channels, dtype=torch.float32, device=calibration_device
            )  # setup holder; calibrate on cpu
            for i in range(
                0, num_elements, chunk_size
            ):  # Compute x_sum (for averaging)
                end = min(i + chunk_size, num_elements)
                x_sum += inp_flat[i:end].to(torch.float32).sum(dim=0)  # [I]

            # Final mean over all tokens
            x_mean_flat = (
                (x_sum / num_elements).to(inp.dtype).to(forward_device)
            )  # [I (in_features)]; forward on gpu

            # Reshape into [G, group_size]
            x_mean = x_mean_flat.view(num_groups, self.group_size)  # [G, group_size]

            clear_memory(x_sum)

        except Exception as e:
            logger.error(f"[{layer_name}] Error during input mean computation: {e}")
            raise

        log_gpu_usage("[fp16_output] BEFORE forward pass")  # ⬅️ VRAM logging

        # * --- STEP 3: Forward pass for FP16 output ---
        try:
            module_kwargs = self._sanitize_kwargs(kwargs, module2inspect)
            fp16_output = forward_with_memory_chunking(
                module=module2inspect.to(
                    forward_device
                ),  # move module to forward device (gpu)
                inp=inp,
                module_kwargs=module_kwargs,
                max_chunk_memory=self.max_chunk_memory,
            )

            log_gpu_usage("[fp16_output] AFTER forward pass")  # ⬅️ VRAM logging

            fp16_output = fp16_output.clip(
                torch.finfo(fp16_output.dtype).min,
                torch.finfo(fp16_output.dtype).max,
            )
            fp16_output = fp16_output.to(
                calibration_device
            )  # to calibration device (cpu)
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

        # * --- STEP 4: Search best scale per group → returns scalar per group ---
        # * --- Compute best scale via grid search ---

        # setup holder
        scales = torch.zeros(
            num_groups, self.group_size, dtype=torch.float16, device=forward_device
        )  # [G, group_size]

        # iterate over each group (using loop for memory safety)
        for g in range(num_groups):
            # w and x for each group
            w_mean_group = w_mean[g]  # [group_size]
            x_mean_group = x_mean[g]  # [group_size]

            log_gpu_usage(
                f"[{layer_name}] Group {g}: before scale search"
            )  # ☑️ # Log VRAM

            logger.debug(f"w_mean_group shape == {w_mean_group.shape}")
            logger.debug(f"x_mean_group shape == {x_mean_group.shape}")

            # todo: debug; delete later
            logger.debug(
                f"[Group {g}] w_mean preview: {w_mean_group[:5].to(torch.float32).cpu().numpy().round(4)}"
            )
            logger.debug(
                f"[Group {g}] x_mean preview: {x_mean_group[:5].to(torch.float32).cpu().numpy().round(4)}"
            )
            # first 2 groups, first 5 dims

            # Call your per-group scale search
            best_scale_per_group = self._compute_best_scale_groupwise(
                x=inp,  # full input
                w_mean=w_mean_group,  # per group only
                x_mean=x_mean_group,  # per group only
                module2inspect=layer,  # full layer (layer for pass forward)
                linears2scale=[layer],  # full layer (layer to calib scales)
                fp16_output=fp16_output,  # full layer output
                group_idx=g,
                group_size=self.group_size,
                kwargs=module_kwargs,
            )

            log_gpu_usage(
                f"[{layer_name}] Group {g}: after scale search"
            )  # ☑️ VRAM logging

            # todo: debug; delete later
            logger.debug(
                f"[{layer_name}] Group {g} scale shape == {best_scale_per_group.shape}"
            )
            logger.debug(
                f"[{layer_name}] Group {g} scale preview: {best_scale_per_group[:5].tolist()}"
            )

            scales[g, :] = best_scale_per_group  # ✅ [G, group_size]

        # Logging and shape check
        logger.debug(
            f"[{get_op_name(module, layer)}] best_scales shape: {scales.shape}"
        )
        logger.debug(
            f"[{get_op_name(module, layer)}] best_scales preview: {scales[:3, :3].tolist()}"
        )

        logger.debug(f"scales.shape == {scales.shape}")

        del inp, fp16_output
        clear_memory()

        # return get_op_name(module, prev_op), (layer_name,), scales # Comment out for now
        return scales

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
        * group iteration (to save memory)

        Efficient per-group scale search using AutoAWQ-style duo scaling.

        This method replaces AutoAWQ's original `_compute_best_scale` to perform
        quantization calibration for a specific input group using grid search over
        scaling ratios.

        It minimizes the reconstruction loss between the quantized and original
        outputs by testing candidate scale values derived from a fused formula:

            s = (x_mean^r / w_mean^(1 - r))                     [Duo scaling]
            L(s) = || Q(W * s) * (s^-1 * X) - W * X ||

        Where:
            - Q: weight quantization function (`pseudo_quantize_tensor`)
            - W: original FP16 weights
            - X: calibration inputs
            - s: candidate scaling factor (scalar per group, or vector in full version)

        The method iteratively applies scale candidates, quantizes the weights,
        forwards the input through the modified layer(s), and compares the result
        with the original output to find the best-performing scale.

        Args:
            x (torch.Tensor): Full input tensor to the layer [B, in_features].
            w_mean (torch.Tensor): Per-output-channel mean of normalized weights
                for the group [out_features].
            x_mean (torch.Tensor): Per-output-channel mean of input activations
                for the group [out_features].
            module2inspect (nn.Module): Layer/module to perform forward pass.
            linears2scale (List[nn.Linear]): Linear layers to apply the scale to.
            fp16_output (torch.Tensor): Ground truth output from the full-precision layer
                [B, out_features].
            group_idx (int): Index of the current input group (0-based).
            group_size (int): Size of each input group (must divide in_features evenly).
            kwargs (Dict): Optional forward arguments for `module2inspect`.

        Returns:
            torch.Tensor: Best scale (scalar) for this group [1-element tensor].
        """

        start = group_idx * group_size
        end = (group_idx + 1) * group_size

        history = []
        best_ratio = -1
        best_scale = None
        best_error = float("inf")

        # Set n grid and tolerance depending on tensor size
        if fp16_output.shape[-1] >= 8192:  # For big MLPs like gate_proj/down_proj
            n_grid = 10
            early_stop_tolerance = 1.001
        else:
            n_grid = 20
            early_stop_tolerance = 1.05

        # Cache original weights directly instead of full state_dict
        # * Process: Save original, Apply scale and quantize, Measure loss, Repeat w/t other scales
        original_weights = {
            fc: fc.weight[:, group_idx * group_size : (group_idx + 1) * group_size]
            .detach()
            .clone()
            for fc in linears2scale
        }

        # Move to the right device
        device = x.device
        x_mean = x_mean.view(-1).to(device)
        w_mean = w_mean.view(-1).to(device)
        fp16_output = fp16_output.to(device)

        assert x_mean.shape == w_mean.shape, "x_mean and w_mean must match group size"
        assert not torch.isnan(x_mean).any(), "x_mean contains NaNs"
        assert not torch.isnan(w_mean).any(), "w_mean contains NaNs"

        w_slice = torch.cat([original_weights[fc] for fc in linears2scale], dim=0)
        std_anchor = w_slice.std().item()
        std_anchor = max(std_anchor, 1e-5)  # avoid zero or tiny

        ratios = torch.linspace(0.0, 1.0, n_grid)  # covers full α range

        for ratio_idx, ratio in enumerate(ratios):

            # AutoAWQ's fused duo-scaling formula: s = (x_mean^r / w_mean^(1-r))^(1/norm)
            if self.duo_scaling:
                scale_tensor = (
                    x_mean.pow(ratio) / (w_mean.pow(1.0 - ratio) + 1e-4)
                ).clamp(min=1e-4)
            else:
                scale_tensor = x_mean.pow(ratio).clamp(min=1e-4)

            # Normalize the scale tensor to prevent dynamic range blowup
            scale_tensor = (
                scale_tensor / (scale_tensor.max() * scale_tensor.min()).sqrt()
            )  # [group_size]

            # Ensure no NaN or Inf in scales
            scale_tensor[torch.isnan(scale_tensor)] = 1.0
            scale_tensor[torch.isinf(scale_tensor)] = 1.0

            # Broadcast to be a 2D tensor
            scale_tensor = scale_tensor.view(1, -1)  # Shape: [1, group_size]

            # Apply pseudo-quantization for this scale
            for fc in linears2scale:
                with torch.no_grad():
                    fc.weight[:, start:end].copy_(
                        original_weights[fc]
                    )  # Restore original
                    fc.weight[:, start:end].copy_(
                        self.pseudo_quantize_tensor(
                            fc.weight[:, start:end] / scale_tensor
                        )[0]
                        * scale_tensor
                    )  # quantize–then–dequantize and write new scaled weights

                logger.debug(
                    f"[{fc.__class__.__name__}] Group {group_idx} | "
                    f"scale preview: {scale_tensor[:5].to(torch.float32).cpu().numpy().round(4).tolist()}"
                )

            int_w_output = self._module_forward(x, module2inspect, kwargs).clamp(
                torch.finfo(fp16_output.dtype).min, torch.finfo(fp16_output.dtype).max
            )

            loss = self._compute_loss(fp16_output, int_w_output, device)
            history.append(loss)

            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                best_scale = scale_tensor.clone()
                logger.debug(
                    f"[Group {group_idx}] New best scale (preview): {scale_tensor[:2].tolist()}, loss = {loss:.6f}"
                )

            elif loss > best_error * early_stop_tolerance:
                logger.debug(
                    f"[Group {group_idx}] Early exit at ratio {ratio:.3f}, loss = {loss:.6f}"
                )
                break

        if best_ratio == -1:
            logger.error(f"No valid scale found for group. Loss history: {history}")
            raise Exception

        assert best_scale is not None  # Add extra guard

        return best_scale.view(-1).to(dtype=torch.float16, device=device)

    @torch.no_grad()
    def _search_best_scale_per_channel(
        self,
        module: nn.Module,
        prev_op: nn.Module,
        layer: nn.Linear,
        inp: torch.Tensor,
        module2inspect: Optional[nn.Module] = None,
        kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        * Full calibration!

        AWQ-style group-wise *per-channel* scale search for a single Linear layer.

        This method normalizes weights per group and computes a scale value for each
        output channel within each group, resulting in a scale tensor of shape [O, G].

        It balances quantized weight distribution against input activation dynamics,
        using grid search for per-output scale tuning within each input group.

        Args:
            module (nn.Module): Parent container (e.g., TransformerBlock).
            prev_op (nn.Module): Previous op (e.g., LayerNorm, GELU) used for
                applying scale.
            layer (nn.Linear): Target Linear layer to calibrate.
            inp (torch.Tensor): Input tensor to the layer (typically float16).
            module2inspect (Optional[nn.Module]): Module used during the forward
                pass (defaults to `layer`).
            kwargs (Optional[dict]): Additional kwargs for the forward pass
                (e.g., attention mask).

        Returns:
            Scale tensor of shape [out_features, num_groups] (float16 or bfloat16)

        Steps:
        1. **Group and normalize weights**
            - Reshape weight to [O, G, group_size]
            - Normalize by max per group
            - Compute mean absolute weight per output channel → w_mean ∈ [O, G]

        2. **Compute input means**
            - Input activation x ∈ [B, S, I]
            *- Compute mean per input group per channel → x_mean ∈ [O, G]

        3. **Forward pass**
            - Get reference fp16 output ∈ [B, O]

        4. **Scale optimization**
            - For each group:
                - Slice x, w, and x_mean
                - Call `_compute_best_scale_groupwise_per_channel`
                - Receive per-output scale → ∈ [O]
                - Store into column `g` of `scales ∈ [O, G]`

        Output Summary:
            Produces per-output per-group scale tensor (2D) compatible with GEMM quantization.
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
                f"[{layer_name}] weight_grouped preview [:2, :2, :2]: {weight_grouped[:2, :2, :2].tolist()}"
            )

            clear_memory(weight_grouped)

        except Exception as e:
            logger.error(f"[{layer_name}] Error during weight scale computation: {e}")
            raise

        # * --- STEP 2: Compute input mean (per input channel), chunked to avoid OOM
        try:
            # Flatten batch and sequence
            inp_flat = inp.cpu().abs().view(-1, inp.shape[-1])  # [B*S, I]

            num_elements = inp_flat.size(0)  # B*S
            num_channels = inp_flat.size(1)  # I
            chunk_size = min(
                self.max_chunk_memory // (inp_flat.element_size() * 2 * num_channels),
                num_elements,
            )

            x_sum = torch.zeros(num_channels, dtype=torch.float32)  # [I]
            for i in range(0, num_elements, chunk_size):
                end = min(i + chunk_size, num_elements)
                x_sum += inp_flat[i:end].to(torch.float32).sum(dim=0)

            x_mean_flat = (
                (x_sum / num_elements).to(inp.dtype).to(weight.device)
            )  # [I] ([in_features])
            x_mean_grouped = x_mean_flat.view(
                num_groups, self.group_size
            )  # [G, group_size]
            x_mean_per_channel = x_mean_grouped.unsqueeze(0).expand(
                out_features, -1, -1
            )  # [O, G, C (channel per group/group size)] Store this and slice by group later inside the loop.

            logger.info(
                f"[{layer_name}] x_mean_grouped per_channel_shape = {x_mean_per_channel.shape}"
            )
            logger.info(
                f"[{layer_name}] x_mean_per_channel preview: {x_mean_per_channel[:2, :2].tolist()}"
            )

            clear_memory(x_sum)

        except Exception as e:
            logger.error(f"[{layer_name}] Error during input mean computation: {e}")
            raise

        log_gpu_usage("[fp16_output] BEFORE forward pass")  # ⬅️ VRAM logging

        # * --- STEP 3: Forward pass for FP16 output ---
        try:
            module_kwargs = self._sanitize_kwargs(kwargs, module2inspect)
            fp16_output = self._module_forward(
                inp, module2inspect, module_kwargs
            )  # [B*S, O]

            log_gpu_usage("[fp16_output] AFTER forward pass")  # ⬅️ VRAM logging

            fp16_output = fp16_output.clip(
                torch.finfo(fp16_output.dtype).min,
                torch.finfo(fp16_output.dtype).max,
            )

            logger.debug(f"fp16_output shape == {fp16_output.shape}")

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

            # # Slice inputs and weights for the group
            # x_per_group = inp[:, start:end]  # ☑️ [B*S, group_size]

            # Extract precomputed input mean for group g
            # Shape: [out_features, group_size]
            x_mean_per_channel_per_group = x_mean_per_channel[:, g, :]
            logger.info(
                f"x_mean_per_channel_per_group shape == {x_mean_per_channel_per_group.shape}"
            )

            # Extract current weight group
            w_per_group = layer.weight[
                :, start:end
            ]  # shape: [O, group_size], i.e., [1536, 128]

            # # Expand to match number of output channels (per output channel scaling)
            # x_per_group_mean = x_per_group_mean.unsqueeze(0).expand(
            #     layer.weight.shape[0], -1
            # )  # shape: [O, group_size]

            # Normalize weight magnitude
            w_per_group_max = (
                w_per_group.abs().amax(dim=1, keepdim=True) + 1e-6
            )  # [O, 1]
            w_per_group_scaled = w_per_group.abs() / w_per_group_max  # [O, group_size]

            # Log VRAM utilization
            log_gpu_usage(f"[{layer_name}] Group {g}: before scale search")

            # Call your per-group scale search
            best_scale_per_group = self._compute_best_scale_groupwise_per_channel(
                x=inp,  # full input
                w_mean_group=w_per_group_scaled,  # [O, C] or [O, group_size]
                x_mean_group=x_mean_per_channel_per_group,  # [O, C] [O, group_size]
                module2inspect=layer,  # full layer (layer for pass forward)
                linears2scale=[layer],  # full layer (layer to calib scales)
                fp16_output=fp16_output,  # full layer output
                group_idx=g,
                group_size=self.group_size,
                kwargs=module_kwargs,
            )
            logger.info(f"best_scale_per_group shape == {best_scale_per_group.shape}")

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
            f"[{get_op_name(module, layer)}] best_scales preview:\n{scales[:3, :3]}"
        )

        assert scales.shape == (out_features, num_groups), (
            f"[{layer_name}] Expected best_scales shape {[out_features, num_groups]}, "
            f"got {scales.shape}"
        )

        del inp, fp16_output
        clear_memory()

        return scales

    def _compute_best_scale_groupwise_per_channel(
        self,
        x: torch.Tensor,
        w_mean_group: torch.Tensor,
        x_mean_group: torch.Tensor,
        module2inspect: torch.nn.Module,
        linears2scale: List[nn.Linear],
        fp16_output: torch.Tensor,
        group_idx: int,  # ✔️ Added to autoawq code
        group_size: int,  # ✔️ Added to autoawq code
        kwargs: Dict = {},
    ):
        """
        * Full calibration.

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
            x_mean (torch.Tensor): Per-channel input activation mean for this group,
                broadcasted across output channels. Shape: [out_features, group_size].
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
        x_mean_group = x_mean_group.to(
            device
        )  # Shape: [O, group_size] = [out_features, C]
        w_mean_group = w_mean_group.to(device)  # Shape: [O, group_size]

        # todo: debug; delete later
        logger.info(
            f"[group {group_idx}] x_group_mean shape: {x_mean_group.shape}"
        )  # Expecting [O, C]
        logger.info(
            f"[group {group_idx}] w_group shape: {w_mean_group.shape}"
        )  # Expecting [O, C] or [O, 1]

        assert not torch.isnan(
            w_mean_group
        ).any(), "w_mean contains NaNs"  # extra guard
        assert not torch.isnan(
            x_mean_group
        ).any(), "x_mean contains NaNs"  # extra guard

        for ratio in range(n_grid):
            # create new scales
            ratio = ratio / n_grid

            # NOTE: s^-1 * x is fused here, according to paper
            if self.duo_scaling:  # use both weights and activation
                scales = (
                    x_mean_group.pow(ratio) / (w_mean_group.pow(1 - ratio) + 1e-4)
                ).clamp(min=1e-4)
            else:  # activation only
                scales = x_mean_group.pow(ratio).clamp(min=1e-4).view(-1)

            scales = (scales / (scales.max() * scales.min()).sqrt()).to(
                device
            )  # normalizaion/scale; [O, C or group_size]

            # avoid scaling values that overflow
            scales[torch.isinf(scales)] = 1
            scales[torch.isnan(scales)] = 1

            # ✅ Quantize weights: Q(W * s)
            for fc in linears2scale:
                fc.weight[:, start:end].mul_(scales)  # ☑️ updated to slice weights
                fc.weight.data[:, start:end] = (
                    self.pseudo_quantize_tensor(fc.weight.data[:, start:end])[0]
                    / scales
                )  # ☑️ updated to slice weights
                logger.debug(f"Group {group_idx}: scale shape = {scales.shape}")

            # ✅ Feed forward to compute activation: W * X
            int_w_output = self._module_forward(x, module2inspect, kwargs)
            int_w_output = int_w_output.clip(
                torch.finfo(int_w_output.dtype).min, torch.finfo(int_w_output.dtype).max
            ).to(
                "cpu"
            )  # ☑️ clamp; Added to bring to same device as fp16_output

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

            if max_val_tensor.numel() != 1:
                max_val_tensor = max_val_tensor.max()

            max_val: float = max_val_tensor.item()

            logger.info(f"[{layer_name}] best clip value = {max_val:.4f}")
            return layer_name, max_val

        except Exception as e:
            logger.error(f"❌ Error computing clip for {layer_name}: {e}")
            return layer_name, None

        finally:
            layer.cpu()
            logger.debug(f"[{layer_name}] moved back to CPU after clip search.")

    # todo: WIP (need to make it full per output/per input channel, or delete)
    @torch.no_grad()
    def compute_zeros_per_channel(
        self,
        weight: torch.Tensor,  # shape: [O, I] — output × input
        scales: torch.Tensor,  # shape: [O, G] — one scale per output and group
        group_size: int,
    ) -> torch.Tensor:
        """
        * Compute per-output-channel, per-group asymmetric zero-points for quantization.
        ! Asymmetric requires per group, per channel computation!

        This function calculates a separate zero-point for each output channel (row) and
        input group (column slice) using the asymmetric formula:
            zero = round(-min(w) / scale)

        This aligns with the quantization format used by AWQ-style inference kernels, which
        apply weight group-wise but retain per-channel scaling.

        Args:
            weight (torch.Tensor): Unquantized weight matrix of shape [O, I],
                                where O = output channels and I = input features.
            scales (torch.Tensor): Scaling tensor of shape [O, G] (float16 or float32),
                                where G = I // group_size.
            group_size (int): Number of input features per group. Must divide I evenly.

        Returns:
            torch.Tensor: Zero-point tensor of shape [G, O], dtype=torch.int32.
                        Each [g, o] entry contains the zero point for group g, output o.

        Notes:
            - The output is transposed to [G, O] to match kernel expectations (e.g., GEMM).
            - Assumes asymmetric quantization where zeros are non-zero integers.
            - Improves range fit over symmetric quantization (which assumes zero-point = 0).
        """
        O, I = weight.shape
        if I % group_size != 0:
            raise ValueError(
                f"Input dim I={I} must be divisible by group_size={group_size}"
            )

        G = I // group_size
        zeros = torch.empty(G, O, dtype=torch.int32, device=weight.device)  # [G, O]

        for g in range(G):
            start, end = g * group_size, (g + 1) * group_size
            w_group = weight[:, start:end]  # shape: [O, group_size]
            min_vals = w_group.min(dim=1).values  # shape: [O]
            scale_g = scales[:, g].to(torch.float32)  # shape: [O]
            zero_vals = (-min_vals / scale_g).round().to(torch.int32)  # shape: [O]
            zeros[g] = zero_vals

            if g == 0:
                logger.debug(
                    f"[compute_zeros] Group {g}: "
                    f"scale_g[:5]={scale_g[:5].tolist()}, "
                    f"min_vals[:5]={min_vals[:5].tolist()}, "
                    f"zero_vals[:5]={zero_vals[:5].tolist()}"
                )

        logger.info(
            f"✅ Computed int32 zero-points: shape={zeros.shape}, dtype={zeros.dtype}"
        )
        logger.debug(f"[compute_zeros] First group zeros: {zeros[0, :5].tolist()}")
        logger.debug(f"[compute_zeros] min={zeros.amin()}, max={zeros.amax()}")

        return zeros

    def calibrate_and_quantize(
        self, save_dir_path: Optional[str | Path] = None, use_full_calib: bool = False
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
            use_full_calib (bool):
                If True, performs full calibration with per-output-channel per-group
                scale search (resulting in a [O, G] scale tensor). This improves quantization
                accuracy but requires more computation.
                If False, uses a lighter calibration mode with
                one scale per group ([G]), which is faster but less precise.

        Raises:
            RuntimeError: If `self.modules` is not initialized.
            ValueError: If no calibration scales are generated.

        Logging:
            - Logs per-layer progress and timing.
            - Logs number of scale and clip entries computed.
            - Logs total elapsed time for the full pipeline.
        """
        start_time = time.time()

        # Setup search scale and compute zeros functions based on use_full_calib

        # todo: debut; delete later
        logger.info(f"use_full_calib: {use_full_calib}")

        search_best_scale_fn = (
            self._search_best_scale_per_channel
            if use_full_calib
            else self._search_best_scale
        )
        mode_str = (
            "full (per-channel per-group)"
            if use_full_calib
            else "light (per-group only)"
        )
        logger.info(
            f"🚀 [calibrate] Starting calibration & quantization — using {mode_str} scale search."
        )

        # Set device (gpu/cup)
        device = get_best_device()  # pylint: disable=global-variable-undefined

        # * Init calibration
        self.init_calibration()
        self.model = self.model.to(device)
        logger.info("Model set up for quantization.")

        if self.modules is None or self.inps is None:
            raise RuntimeError("Calibration data (modules or inputs) not initialized.")

        # * Calibrate and quantize
        total_layers_quantized = 0
        for idx, module in enumerate(self.modules):  # * ☑️ Outer loop

            # Check if the block exists already
            if save_dir_path is not None:
                # Dynamically get and filter sublayers based on the same exclusion logic
                named_linears = exclude_layers_to_not_quantize(
                    get_named_linears(module), self.modules_to_not_convert
                )

                # todo: debug; delete later
                logger.info(f"Layers from named_linear: {named_linears}")

                logger.info(f"🔍 Checking for quantized layers in: {save_dir_path}")
                if are_block_sublayers_quantized(
                    block_module=module,
                    block_idx=idx,
                    save_dir=str(save_dir_path),
                    modules_to_not_convert=self.modules_to_not_convert,
                ):
                    logger.info(
                        f"⏭️ [Block {idx}] Skipping — all sublayers already quantized."
                    )
                    continue

            logger.info(
                f"\n🔍 [Block {idx}/{len(self.modules)}] Processing {module.__class__.__name__}"
            )
            log_gpu_usage(
                f"[calibrate] Block {idx} - before moving module to GPU"
            )  # ☑️ log VRAM

            self.move_embed(self.model, device)  # Move shared embeddings back to GPU

            log_gpu_usage("[calibrate] After moving model to GPU")  # ⬅️ log VRAM

            named_linears = exclude_layers_to_not_quantize(
                get_named_linears(module), self.modules_to_not_convert
            )
            name_to_layer = {v: k for k, v in named_linears.items()}

            # todo: debug; delete later
            logger.info(f"named_linear: {named_linears}")

            # 💡 Input features captured while module is on CPU
            input_feat = self._get_input_feat(module, named_linears)
            # input_feat = {k: v.to(device) for k, v in input_feat.items()}

            # todo: debug; delete later
            logger.info("Input features captured: %s", list(input_feat.keys()))
            if "self_attn.o_proj" in input_feat:
                logger.info(
                    "o_proj input shape: %s", input_feat["self_attn.o_proj"].shape
                )
            else:
                logger.info("self_attn.o_proj input not captured!")

            logger.debug(f"[Block {idx}] Input features captured.")

            # 🚀 Move everything needed to GPU
            move_module_to_device(module=module, input_feat=input_feat, device="cuda")

            # Defensive check: reset or filter tensors
            if self.module_kwargs is not None:
                for k, v in self.module_kwargs.items():
                    if isinstance(v, torch.Tensor):
                        self.module_kwargs[k] = v.to(device)

                # module_config = self.awq_model.get_layers_for_scaling(
                #     module, input_feat, self.module_kwargs
                # )
                module_config = self.get_layers_for_scaling(
                    module, input_feat, self.module_kwargs
                )  # Call custom get_layers_for_scaling method
                logger.debug(f"[Block {idx}] Groups = {len(module_config)}")

            for group in module_config:
                prev_op = group["prev_op"]
                layers = group["layers"]
                group_name = prev_op.__class__.__name__

                # Resolve actual layer names (needed for attention check)
                layer_names = [
                    str(name_to_layer[layer])
                    for layer in layers
                    if name_to_layer.get(layer)
                ]

                logger.info(f"\n⚙️  [Group: {group_name}] {len(layers)} layers")

                # todo: debug; delete later
                logger.info(f"[Group: {group_name}] {layer_names})")

                # # Move RoPE back to GPU if needed
                # if is_attention_layer([str(group_name)]) or is_attention_layer(
                #     layer_names
                # ):
                #     move_rope_to_device(self.model, device)

                # * 1. Compute scales per layer
                scales_dict = {}
                for layer in layers:
                    layer_name = name_to_layer.get(layer)
                    if layer_name is None:
                        raise ValueError(f"Layer not found in name mapping: {layer}")
                    logger.info(
                        f"About to quantize: {layer_name}, input_feat shape: {input_feat[layer_name].shape}"
                    )
                    logger.info(f"🔎 [scale] for {layer_name}")

                    log_gpu_usage(
                        f"[calibrate] Block {idx} - before scale search ({layer_name})"
                    )  # ⬅️ log VRAM

                    best_scales = search_best_scale_fn(
                        module=module,
                        prev_op=prev_op,
                        layer=layer,
                        inp=input_feat[layer_name],
                    )  # [G, group_size]

                    best_scales = best_scales.to(device)
                    scales_dict[layer_name] = best_scales

                    log_gpu_usage(
                        f"[calibrate] Block {idx} - after scale search ({layer_name})"
                    )  # ⬅️ log VRAM

                    logger.debug(
                        f"[raw scales] {layer_name} → shape: {best_scales.shape}, preview: {best_scales[:3, :3].tolist()}"
                    )

                # * 2. Apply scale, compute zeros, clip, quantize, save, clear VRAM
                for layer in layers:

                    # * Save a copy of original weights for WQLinear_GEMM.from_linear method
                    original_weight = layer.weight.detach().clone()
                    logger.info("Created copy of original weight.")

                    layer_name = name_to_layer[layer]

                    # Apply scales
                    scales = scales_dict[layer_name]  # [G, group_size]
                    logger.info(f"🧪 [apply] scale to {layer_name}")

                    apply_scale_all_groups(layer=layer, scales=scales)
                    logger.info(f"Scales applied to {layer_name}")

                    # todo: debug; delete later; Check if weights changed
                    delta = (
                        (layer.weight - original_weight.to(layer.weight.device))
                        .abs()
                        .mean()
                        .item()
                    )
                    logger.info(
                        f"[{layer_name}] mean delta from original weight after scaling: {delta:.6f}"
                    )

                    # todo: extra logging to investigate 0 z-points; delete later
                    weight = layer.weight.data
                    group_size = self.group_size  # <-- use self here
                    G = weight.shape[1] // group_size
                    mean_per_group_original = [
                        original_weight[:, g * group_size : (g + 1) * group_size]
                        .mean()
                        .item()
                        for g in range(G)
                    ]
                    logger.info(
                        f"[{layer_name}] weight mean per group before scale: {mean_per_group_original[:5]}"
                    )
                    logger.info(
                        f"[{layer_name}] weight std before scale: {original_weight.std().item():.6f}"
                    )
                    logger.info(
                        f"[{layer_name}] weight mean before scale: {original_weight.mean().item():.6f}"
                    )

                    # todo: extra logging to investigate 0 z-points; delete later
                    mean_per_group = [
                        weight[:, g * group_size : (g + 1) * group_size].mean().item()
                        for g in range(G)
                    ]
                    logger.info(
                        f"[{layer_name}] weight mean per group after scale: {mean_per_group[:5]}"
                    )
                    logger.info(
                        f"[{layer_name}] weight std after scale: {weight.std().item():.6f}"
                    )
                    logger.info(
                        f"[{layer_name}] weight mean after scale: {weight.mean().item():.6f}"
                    )

                    # move scales to CPU to free up VRAM
                    scales = offload_tensor_to_cpu(scales)

                    # Apply clips
                    if self.apply_clip:
                        # Optional clipping
                        input_feat_tensor = input_feat[layer_name]
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

                    # Compute zeros and scales (quantization scales)
                    if use_full_calib:  # todo: full calib still under construction
                        weight = layer.weight.data
                        scale_for_zp = get_scale_for_zero_point(
                            weight, self.group_size, self.w_bit
                        )
                        zeros = self.compute_zeros_per_channel(
                            weight, scale_for_zp, self.group_size
                        )
                    else:
                        w = layer.weight.data  # [O, I]
                        layer.weight.data, quant_scales, zeros = (
                            self.pseudo_quantize_tensor(w)
                        )

                    # todo: debug; delete later
                    logger.debug(
                        f"[quant_scales] {layer_name} → {quant_scales.shape}, first group first 5: {quant_scales[0, :5].tolist()}"
                    )

                    # todo: debug; delete later
                    if zeros is not None:
                        logger.debug(
                            f"[zeros] {layer_name} → {zeros.shape}, first group first 5: {zeros[0, :5].tolist()}"
                        )

                    # Move metadata to CPU
                    quant_scales = offload_tensor_to_cpu(quant_scales)
                    if zeros is not None:
                        zeros = offload_tensor_to_cpu(
                            zeros
                        )  # ✅ Move to CPU after zero computation

                    # ! Clear VRAM cache
                    del original_weight
                    gc.collect()
                    torch.cuda.empty_cache()
                    logger.info("Original weights deleted and VRAM memory cleared.")

                    # * ✅ Quantize & replace sub-layer with quantized GEMM layer

                    # ☑️ Load all to GPU before GEMM creation
                    logger.info(
                        f"{layer_name} loading weights, quantization scales, zero points back to GPU."
                    )
                    device = (
                        layer.weight.device
                        if layer.weight.is_cuda
                        else get_best_device()
                    )
                    quant_scales = quant_scales.to(device)
                    if zeros is not None:
                        zeros = zeros.to(device)
                    layer = layer.to(device)

                    # Transpose scales and zeros
                    if self.version == "gemm":
                        quant_scales = quant_scales.t().contiguous()
                        if zeros is not None:
                            zeros = zeros.t().contiguous()

                    # ✅ Sanity check
                    G = layer.weight.shape[1] // self.group_size
                    O = layer.weight.shape[0]

                    assert quant_scales.shape == (
                        G,
                        O,
                    ), f"Expected scales [G, O], got {scales.shape}"
                    if zeros is not None:
                        assert zeros.shape == (
                            G,
                            O,
                        ), f"Expected zeros [G, O], got {zeros.shape}"

                    # todo: debug; delete later
                    # 🔍 Quantization debug peek
                    logger.debug("Checking before WQLinear_GEMM:")
                    logger.debug(
                        f"[{layer_name}] weight: shape={layer.weight.shape}, max={layer.weight.abs().max():.6f}, min={layer.weight.min():.6f}"
                    )
                    logger.debug(
                        f"[{layer_name}] scales: shape={quant_scales.shape}, max={quant_scales.max():.6f}, min={quant_scales.min():.6f}, mean={scales.mean():.6f}"
                    )
                    logger.debug(
                        f"[{layer_name}] zeros: shape={zeros.shape}, max={zeros.max():.2f}, min={zeros.min():.2f}, mean={zeros.float().mean():.2f}"
                    )

                    # Optional: print actual small slice
                    if (
                        quant_scales.ndim == 2
                        and quant_scales.size(0) >= 5
                        and quant_scales.size(1) >= 5
                    ):
                        logger.debug(
                            f"[{layer_name}] scale[:3, :5] = {quant_scales[:3, :5].tolist()}"
                        )
                    else:
                        logger.debug(
                            f"[{layer_name}] scale preview: {quant_scales.tolist()}"
                        )

                    # Zero-points
                    if zeros.ndim == 2 and zeros.size(0) >= 5 and zeros.size(1) >= 5:
                        logger.debug(
                            f"[{layer_name}] zero[:2, :2] = {zeros[:2, :2].tolist()}"
                        )
                    else:
                        logger.debug(f"[{layer_name}] zero preview: {zeros.tolist()}")

                    # Weights
                    if (
                        layer.weight.ndim == 2
                        and layer.weight.size(0) >= 5
                        and layer.weight.size(1) >= 5
                    ):
                        logger.debug(
                            f"[{layer_name}] weight[:5, :5] = {layer.weight[:5, :5].tolist()}"
                        )
                    else:
                        logger.debug(
                            f"[{layer_name}] weight preview: {layer.weight.tolist()}"
                        )
                    # todo: debug; delete above after

                    # Quantization and packing
                    quantized_layer = WQLinear_GEMM.from_linear(
                        linear=layer,  # ✅ original unscaled weights [O, I]
                        w_bit=self.w_bit,  # ✅ 4 (or 8), matches qmax and packing logic
                        group_size=self.group_size,  # ✅ used for slicing input dim (I must be divisible)
                        scales=quant_scales,  # ✅ quantization scales [G, O]
                        zeros=zeros,  # ✅ zero points [G, O]
                    )

                    # todo: debug; delete later
                    # 🔍 Debug: after creating WQLinear_GEMM
                    logger.debug("Checking after WQLinear_GEMM:")
                    logger.debug(
                        f"[{layer_name}] qweight shape = {quantized_layer.qweight.shape}, dtype = {quantized_layer.qweight.dtype}"
                    )
                    logger.debug(
                        f"[{layer_name}] qzeros shape = {quantized_layer.qzeros.shape}, dtype = {quantized_layer.qzeros.dtype}"
                    )
                    logger.debug(
                        f"[{layer_name}] scales shape = {quantized_layer.scales.shape}, dtype = {quantized_layer.scales.dtype}"
                    )

                    # qweight: [out_features, in_features // (32 // w_bit)]
                    if (
                        quantized_layer.qweight.ndim == 2
                        and quantized_layer.qweight.size(0) >= 5
                        and quantized_layer.qweight.size(1) >= 5
                    ):
                        logger.debug(
                            f"[{layer_name}] qweight[:5, :5] = {quantized_layer.qweight[:5, :5].tolist()}"
                        )
                    else:
                        logger.debug(
                            f"[{layer_name}] qweight preview: {quantized_layer.qweight.tolist()}"
                        )

                    # qzeros: same shape as qweight typically
                    if (
                        quantized_layer.qzeros.ndim == 2
                        and quantized_layer.qzeros.size(0) >= 5
                        and quantized_layer.qzeros.size(1) >= 5
                    ):
                        logger.debug(
                            f"[{layer_name}] qzeros[:5, :5] = {quantized_layer.qzeros[:5, :5].tolist()}"
                        )
                    else:
                        logger.debug(
                            f"[{layer_name}] qzeros[:3, :3] = {quantized_layer.qzeros[:3, :3].tolist()}"
                        )

                    # scales: usually 1D [num_groups]
                    if (
                        quantized_layer.scales.ndim == 1
                        and quantized_layer.scales.numel() >= 5
                    ):
                        logger.debug(
                            f"[{layer_name}] scales[:5] = {quantized_layer.scales[:5].tolist()}"
                        )
                    else:
                        logger.debug(
                            f"[{layer_name}] scales[:3, :3] = {quantized_layer.scales[:3, :3].tolist()}"
                        )

                    set_op_by_name(
                        layer=module, name=layer_name, new_module=quantized_layer
                    )
                    logger.info(f"✅ Quantized {layer_name} → WQLinear_GEMM")
                    log_gpu_usage(
                        f"[calibrate] Block {idx} - after replacing {layer_name}"
                    )  # ⬅️ log VRAM

                    # Persist layer to disk
                    module_name = get_op_name(self.model, module)

                    if not save_dir_path:
                        raise ValueError(
                            "❌ save_dir_path must be provided to persist quantized layers."
                        )

                    save_success = persist_quantized_layer(
                        quant_layer=quantized_layer,
                        save_dir=str(save_dir_path),
                        module_name=module_name,
                        sub_layer_name=layer_name,
                    )

                    if not save_success:
                        logger.warning(
                            f"⚠️ Skipped saving {module_name}.{layer_name} due to previous error."
                        )

                    # ✅ CLEANUP VRAM memory
                    # clear input features
                    if layer_name in input_feat:
                        del input_feat[layer_name]

                    # ✅ FREE TRANSIENT TENSORS FROM VRAM (scales, zeros, etc.)
                    del scales, zeros
                    gc.collect()
                    torch.cuda.empty_cache()

                    # ✅ NOW safe to move entire module/layer to CPU
                    layer.cpu()

                    log_gpu_usage(
                        f"[free_input] Freed input_feat for {layer_name}"
                    )  # ⬅️ log VRAM

                # Count loops
                total_layers_quantized += len(layers)

            log_gpu_usage(
                f"[calibrate] Block {idx} - end of block before memory cleanup"
            )  # ⬅️ log VRAM

            clear_up_module_memory(
                module=module,
                input_feat=input_feat,
                device="cpu",
            )

            log_gpu_usage(
                f"[calibrate] Block {idx} - end of block after memory cleanup"
            )  # ⬅️ log VRAM

            logger.info(
                f"[Block {idx}] Quantized {len(layers)} layers across {len(module_config)} groups."
            )

        logger.info(f"🔢 Total quantized layers: {total_layers_quantized}")
        elapsed = time.time() - start_time
        logger.info(f"\n🏁 Finished calibration + quantization in {elapsed:.2f} sec")

    def verify_saved_layers(self, load_dir_path: str, strict: bool = True) -> None:
        """
        Verify that all expected per-layer quantized files (layer_0.pt, layer_1.pt, ...)
        exist in the specified directory.

        Args:
            load_dir_path (str): Directory to check.
            strict (bool): If True, raise error if any file is missing. If False, log warnings.

        Raises:
            FileNotFoundError: If any expected layer file is missing (only if `strict=True`).
        """
        num_layers = len(self.model.model.layers)
        missing = []

        logger.info(
            f"🔍 Verifying presence of {num_layers} layer files in: {load_dir_path}"
        )
        for i in range(num_layers):
            layer_file = os.path.join(load_dir_path, f"layer_{i}.pt")
            if not os.path.isfile(layer_file):
                missing.append(f"layer_{i}.pt")

        if missing:
            msg = f"❌ Missing {len(missing)} layer files: {missing}"
            if strict:
                raise FileNotFoundError(msg)
            else:
                logger.warning(msg)
        else:
            logger.info("✅ All expected layer files found.")

    def load_saved_layer_weights(self, load_dir_path: str) -> None:
        """
        Loads quantized sublayer weights into the model.
        Automatically replaces nn.Linear layers with WQLinear_GEMM if needed.
        Logs a summary of replacements, loads, and missing files.
        """
        if not os.path.isdir(load_dir_path):
            raise FileNotFoundError(f"❌ Directory does not exist: {load_dir_path}")

        logger.info(
            f"📥 Loading quantized sublayers from {load_dir_path} using model structure."
        )

        loaded_count = 0
        replaced_layers = []
        reused_layers = []
        missing_layers = []
        failed_layers = []

        if self.modules is None or self.inps is None:
            raise RuntimeError(
                "❌ ScroogeQuantizer was not initialized properly — self.modules or self.inps is None."
            )

        for idx, module in enumerate(self.modules):
            named_linears = exclude_layers_to_not_quantize(
                get_named_linears(module), self.modules_to_not_convert
            )

            for name, layer in named_linears.items():
                filename = f"model.layers.{idx}.{name}.pt"
                filepath = os.path.join(load_dir_path, filename)

                if not os.path.exists(filepath):
                    logger.warning(f"⚠️ Missing: {filename}")
                    missing_layers.append(filename)
                    continue

                try:
                    state_dict = torch.load(filepath, map_location="cpu")

                    if not isinstance(layer, WQLinear_GEMM):
                        logger.info(
                            f"🔁 Replacing model.layers.{idx}.{name} with WQLinear_GEMM"
                        )
                        quant_layer = WQLinear_GEMM(
                            w_bit=self.quant_config.get("w_bit", 4),
                            group_size=self.quant_config.get("q_group_size", 128),
                            in_features=layer.in_features,
                            out_features=layer.out_features,
                            bias="bias" in state_dict,
                            dev="cpu",
                            training=False,
                        )
                        set_op_by_name(module, name, quant_layer)
                        layer = quant_layer
                        replaced_layers.append(filename)
                    else:
                        reused_layers.append(filename)

                    layer.load_state_dict(state_dict)
                    logger.debug(f"✅ Loaded {filename} into model.layers.{idx}.{name}")
                    loaded_count += 1

                except Exception as e:
                    logger.warning(f"⚠️ Failed to load {filename}: {e}")
                    failed_layers.append(filename)

        # 🧾 Summary
        logger.info("\n📦 Load Summary:")
        logger.info(f"   ✅ Loaded and applied: {loaded_count} layers")
        logger.info(f"   🔁 Replaced layers    : {len(replaced_layers)}")
        logger.info(f"   ♻️  Already quantized : {len(reused_layers)}")
        if missing_layers:
            logger.warning(f"   ⚠️ Missing files      : {len(missing_layers)}")
        if failed_layers:
            logger.warning(f"   ❌ Failed to load     : {len(failed_layers)}")

        logger.debug(f"   🔁 Replaced: {replaced_layers}")
        logger.debug(f"   ♻️  Reused   : {reused_layers}")
        logger.debug(f"   ⚠️ Missing : {missing_layers}")
        logger.debug(f"   ❌ Failed  : {failed_layers}")

    def save_quant_config(self, save_dir: str):
        """
        Saves quantization configuration inside 'config.json' using the format expected
        by AutoAWQ.

        Args:
            save_dir (str): Directory where 'config.json' will be written or updated.

        * Note:
            This saves the quantization config in the standard Hugging Face-compatible format:
            - Uses 'bits' instead of 'w_bit'
            - Uses 'group_size' instead of 'q_group_size'

            AutoAWQ's model loading functions (e.g., `from_quantized`) will automatically map
            these fields to its internal `AwqConfig` structure using `from_transformers_dict`.

            This ensures compatibility with standard AutoAWQ model loading workflows.
        """
        config_path = Path(save_dir) / "config.json"

        # Load existing config if present
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = {}

        # Inject quantization_config field using HuggingFace-style names

        # HuggingFace-compatible key mapping
        key_map = {
            "w_bit": "bits",
            "q_group_size": "group_size",
            "zero_point": "zero_point",
            "version": "version",
            "quant_method": "quant_method",
        }

        # Dynamically map and lowercase version
        quant_config_hf = {
            hf_key: (
                self.quant_config.get(orig_key)
                if orig_key != "version"
                else self.quant_config.get("version", "GEMM").lower()
            )
            for orig_key, hf_key in key_map.items()
        }

        # Include excluded layers if present
        if hasattr(self, "modules_to_not_convert"):
            quant_config_hf["modules_to_not_convert"] = self.modules_to_not_convert

        # Inject into config.json
        config["quantization_config"] = quant_config_hf

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        logger.info(
            f"📝 Saved updated config.json with quantization_config → {config_path}"
        )

    def save_quant_model(
        self,
        save_dir: str,
        safetensors: bool = True,
        shard_size: str = "5GB",
    ) -> None:
        """
        Save the quantized model and its configuration (not calibration files or processor).

        This function persists:
            - the quantized model weights (in `.safetensors` or `.bin`)
            - model architecture and generation configuration (in `config.json` and
            `generation_config.json`)

        Args:
            save_dir (str): Directory to save the model files.
            safetensors (bool): Whether to use `.safetensors` format instead of `.bin`.
            shard_size (str): Max shard size for splitting large model files.

        Files Written:
            - `config.json`: Hugging Face model configuration including quantization config.
            - `generation_config.json`: Sampling/generation parameters for inference.
            - `model.safetensors` or `pytorch_model.bin`: Serialized quantized model weights.

        * Files Not Saved (see other functions):
            - `scales.pt`, `qzeros.pt`, `calib_stats.pt`: Save these using a separate
            `save_calibration_artifacts(...)` method.
            - `processor_config.json`: Save processor separately using
            `processor.save_pretrained(...)`.
            - `tokenizer`: Save with `tokenizer.save_pretrained(...)`.

        Notes:
            - The model is saved in a hybrid format with quantized and unquantized layers
            as present in `self.model`.
            - Temporary placeholder weight files (`model.safetensors`/`.bin`) from
            the config save step are removed before saving real weights.
            - This function does not save the raw `state_dict` as `.pt`.

        Example:
            >>> quantizer.save_quant_model("/path/to/save", safetensors=True)
        """
        save_dir = save_dir.rstrip("/")
        os.makedirs(save_dir, exist_ok=True)

        class EmptyModule(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x

        # Embed quantization config for compatibility
        self.model.config.quantization_config = self.quant_config
        self.model.generation_config.do_sample = True

        logger.info(f"Saving model configuration to {save_dir}")
        try:
            self.model.save_pretrained(save_dir, state_dict=EmptyModule().state_dict())
        except Exception as e:
            logger.error(f"Failed to save model configuration: {e}")

        # Remove any placeholder weight files
        for path in [f"{save_dir}/model.safetensors", f"{save_dir}/pytorch_model.bin"]:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logger.debug(f"Removed empty state dict file: {path}")
            except Exception as e:
                logger.warning(f"Failed to remove {path}: {e}")

        # ✅ 🔁 SAVE FULL STATE DICT (no filtering)
        full_state_dict = self.model.state_dict()
        logger.info(
            f"Saving full model weights ({len(full_state_dict)} tensors) to {save_dir}"
        )

        try:
            save_torch_state_dict(
                state_dict=full_state_dict,
                save_directory=save_dir,
                max_shard_size=shard_size,
                safe_serialization=safetensors,
                force_contiguous=True,
                shared_tensors_to_discard=getattr(self.model, "_tied_weights_keys", []),
            )
            logger.info(f"Quantized model saved successfully to {save_dir}")
        except Exception as e:
            logger.error(f"Error while saving quantized model: {e}")

    def save_tokenizer(self, save_dir: str) -> None:
        """
        Save the tokenizer to the specified directory.

        This is required by Hugging Face-compatible workflows to ensure that the model can
        consistently tokenize input text at inference or during downstream fine-tuning.

        Why this matters:
            - Tokenizer vocab and configuration (e.g., special tokens, padding, truncation)
            must match exactly between training and inference.
            - Hugging Face's `from_pretrained(...)` methods expect the tokenizer files
            (`tokenizer_config.json`, `vocab.json`, `tokenizer.model`, etc.) to be saved
            in the same directory as the model or explicitly provided.

        Args:
            save_dir (str): Path to directory where tokenizer files will be saved.

        Output:
            The tokenizer files will be saved in Hugging Face-compatible format,
            allowing seamless reload with `AutoTokenizer.from_pretrained(save_dir)`.
        """
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            os.makedirs(save_dir, exist_ok=True)
            self.tokenizer.save_pretrained(save_dir)
            logger.info(f"✅ Tokenizer saved to {save_dir}")
        else:
            logger.warning("⚠️ No tokenizer found to save.")

    # Optional
    def save_processor(self, save_dir: str) -> None:
        """
        Save the processor configuration (typically for vision or multi-modal models).

        The processor handles input pre-processing for models that accept multiple modalities,
        such as image-text pairs. For example, it may encapsulate tokenization, resizing,
        normalization, or feature extraction steps.

        Args:
            save_dir (str): Path to directory where processor config will be saved.
        """
        if hasattr(self, "processor") and self.processor is not None:
            os.makedirs(save_dir, exist_ok=True)
            self.processor.save_pretrained(save_dir)
            logger.info(f"✅ Processor saved to {save_dir}")
        else:
            logger.info("ℹ️ No processor present — skipping save.")

    # Optional
    def save_metadata(self, save_dir: str) -> None:
        """
        Save model-related metadata such as quantization parameters to a JSON file.

        This metadata provides transparency and reproducibility for downstream consumers
        or tools that inspect the quantization setup.

        Args:
            save_dir (str): Path to directory where metadata.json will be saved.

        Output:
            - metadata.json: Contains quantization config, group size, bit width, etc.
        """
        os.makedirs(save_dir, exist_ok=True)
        metadata = {
            "quantization_config": self.quant_config,
            "group_size": getattr(self, "group_size", None),
            "w_bit": getattr(self, "w_bit", None),
            "calibration_samples": getattr(self, "max_calib_samples", None),
        }
        filepath = os.path.join(save_dir, "metadata.json")
        with open(filepath, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✅ Metadata saved to {filepath}")

    def save_calibration_artifacts(self, save_dir: str) -> None:
        """
        Save calibration artifacts used during quantization.

        These include:
        - `scales.pt`: Layer-wise scaling factors.
        - `clips.pt`: Optional clipping thresholds for activation normalization.

        Args:
            save_dir (str): Path to directory where calibration files will be saved.
        """
        os.makedirs(save_dir, exist_ok=True)
        if hasattr(self, "all_scales"):
            torch.save(self.all_scales, os.path.join(save_dir, "scales.pt"))
            logger.info("✅ Calibration scales saved (scales.pt)")

        if hasattr(self, "all_clips"):
            torch.save(self.all_clips, os.path.join(save_dir, "clips.pt"))
            logger.info("✅ Clipping thresholds saved (clips.pt)")

        if not hasattr(self, "all_scales") and not hasattr(self, "all_clips"):
            logger.info("ℹ️ No calibration artifacts found to save.")

    def build_and_persist_model(
        self,
        load_dir: str,
        save_dir: str,
        safetensors: bool = True,
        shard_size: str = "5GB",
    ) -> None:
        """
        End-to-end workflow: loads quantized sublayer weights, replaces unquantized layers,
        and persists the full quantized model package.

        Args:
            load_dir (str): Directory where quantized sublayer files (.pt) are stored.
            save_dir (str): Directory to save the final model artifacts.
            safetensors (bool): Whether to save the model in `.safetensors` format.
            shard_size (str): Maximum shard size for large model files (e.g., '5GB').

        Steps Performed:
            - Loads quantized sublayers from disk
            - Replaces Linear layers with WQLinear_GEMM where needed
            - Saves the full model, tokenizer, processor, metadata, and calibration artifacts
        """
        logger.info(f"🚀 Starting model patch-and-save pipeline")
        logger.info(f"🔍 Loading quantized layers from: {load_dir}")

        # Load layer file data from disk -> replace/update quant layers in self.model
        self.load_saved_layer_weights(load_dir_path=load_dir)

        # Save model & other files to disk
        logger.info(f"💾 Persisting fully quantized model to: {save_dir}")

        self.save_quant_model(
            save_dir=save_dir, safetensors=safetensors, shard_size=shard_size
        )
        self.save_quant_config(save_dir=save_dir)
        self.save_tokenizer(save_dir=save_dir)
        self.save_processor(save_dir=save_dir)
        self.save_metadata(save_dir=save_dir)
        self.save_calibration_artifacts(save_dir=save_dir)

        logger.info(f"✅ Model fully built and saved to: {save_dir}")
