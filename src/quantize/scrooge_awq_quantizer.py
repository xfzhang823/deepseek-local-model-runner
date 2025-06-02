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
import gc
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
from utils.find_layer_type import is_attention_layer, is_mlp_layer
from utils.offload_to_cpu import offload_tensor_to_cpu
from utils.check_existing import are_block_sublayers_quantized
from quantize.quantize_utils import (
    unwrap_to_transformer,
    flatten_scales_or_clip_list,
    get_safe_parallel_sample_count,
    persist_quantized_layer,
    load_quantized_layers_into_model,
    move_rope_to_device,
    move_module_to_device,
    clear_up_module_memory,
    forward_with_memory_chunking,
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
        self.modules_to_not_convert = ["o_proj", "embed_tokens", "lm_head"]
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

        # --- STEP 1: Compute grouped weight mean ---
        try:
            weight = layer.weight.detach()  # create a copy of the weights
            out_features, in_features = weight.shape  # get dim (for reshaping)

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

            clear_memory(weight_grouped)

        except Exception as e:
            logger.error(f"[{layer_name}] Error during weight scale computation: {e}")
            raise

        # --- STEP 2: Compute input mean (per input channel), chunked to avoid OOM
        try:
            # ✅ run on calibration_device
            inp_flat = inp.to(calibration_device).abs().view(-1, inp.shape[-1])
            num_elements = inp_flat.size(0)
            num_channels = inp_flat.size(1)
            chunk_size = min(
                self.max_chunk_memory // (inp_flat.element_size() * 2 * num_channels),
                num_elements,
            )

            x_sum = torch.zeros(
                num_channels, dtype=torch.float32, device=calibration_device
            )  # setup holder; calibrate on cpu
            for i in range(0, num_elements, chunk_size):
                end = min(i + chunk_size, num_elements)
                x_sum += inp_flat[i:end].to(torch.float32).sum(dim=0)

            # Project input activation mean into output channels
            x_mean_flat = (
                (x_sum / num_elements).to(inp.dtype).to(forward_device)
            )  # [in_features]; forward on gpu
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

            log_gpu_usage(
                f"[{layer_name}] Group {g}: before scale search"
            )  # ☑️ # Log VRAM

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
            f"[{get_op_name(module, layer)}] best_scales preview:{scales[:3, :]}"
        )

        assert scales.shape == (out_features, num_groups), (
            f"[{layer_name}] Expected best_scales shape {[out_features, num_groups]}, "
            f"got {scales.shape}"
        )

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

        history = []
        best_ratio = -1
        best_scales = None
        best_error = float("inf")

        # Set n grid and tolerance depending on tensor size
        if fp16_output.shape[-1] >= 8192:  # For big MLPs like gate_proj/down_proj
            n_grid = 10
            early_stop_tolerance = 1.001
        else:
            n_grid = 20
            early_stop_tolerance = 1.05
        # todo: delete later
        # org_sd = {k: v.cpu() for k, v in module2inspect.state_dict().items()}

        # Cache original weights directly instead of full state_dict
        original_weights = {
            fc: fc.weight[:, group_idx * group_size : (group_idx + 1) * group_size]
            .detach()
            .clone()
            for fc in linears2scale
        }

        # todo: no need to call to cpu inside the loop
        # x = x.to("cpu")
        # fp16_output = fp16_output.to("cpu")
        # for fc in linears2scale:
        #     fc.cpu()

        # Move to the right device
        device = x.device
        x_mean = x_mean.view(-1).to(device)
        w_mean = w_mean.view(-1).to(device)
        fp16_output = fp16_output.to(device)

        assert not torch.isnan(w_mean).any(), "w_mean contains NaNs"  # extra guard
        assert not torch.isnan(x_mean).any(), "x_mean contains NaNs"  # extra guard

        w_slice = torch.cat([original_weights[fc] for fc in linears2scale], dim=0)
        std_anchor = w_slice.std().item()
        std_anchor = max(std_anchor, 1e-5)  # avoid zero or tiny

        ratios = torch.linspace(0.25, 2.0, n_grid)
        for ratio_idx, ratio in enumerate(ratios):
            scale = ratio * std_anchor
            scale = max(scale.item(), 1e-5)

            # Broadcast shape: (out_features, 1)
            scales_view = torch.full((w_slice.shape[0], 1), scale, device=device)

            # Quantize each layer
            for fc in linears2scale:
                with torch.no_grad():
                    fc.weight[:, start:end].copy_(original_weights[fc])  # restore
                    fc.weight[:, start:end].copy_(
                        self.pseudo_quantize_tensor(fc.weight[:, start:end] / scale)[0]
                        * scale
                    )

            # Forward + error
            int_w_output = self._module_forward(x, module2inspect, kwargs).clamp(
                torch.finfo(fp16_output.dtype).min, torch.finfo(fp16_output.dtype).max
            )

            loss = self._compute_loss(fp16_output, int_w_output, device)

            history.append(loss)
            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                best_scales = scales_view[:, 0].clone()
                logger.debug(
                    f"[Group {group_idx}] New best scale = {scale:.6f}, loss = {loss:.6f}"
                )
            elif loss > best_error * early_stop_tolerance:
                logger.debug(
                    f"[Group {group_idx}] Early exit at ratio {ratio:.3f}, loss = {loss:.6f}"
                )
                break

        if best_ratio == -1:
            logger.error(f"No valid scale found for group. Loss history: {history}")
            raise Exception

        assert best_scales is not None  # Add extra guard
        assert torch.isnan(best_scales).sum() == 0, best_scales

        return best_scales.detach().to(device)

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

        Args:
            weight (Tensor): [out_features, in_features]
            scales (Tensor): [out_features, num_groups]
            group_size (int): Number of input dims per group

        Returns:
            zeros (Tensor): [out_features, num_groups]
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

        Compute per-output, per-group zero-points for symmetric quantization.
        """
        O, I = weight.shape

        if I % group_size != 0:
            raise ValueError(
                f"Invalid group size: in_features={I} not divisible by group_size={group_size}"
            )

        G = I // group_size  # number of groups
        device = weight.device
        zeros = torch.empty(
            (O, G), device=device, dtype=scales.dtype
        )  # allocate output

        # Comute zero points
        for g in range(G):
            start = g * group_size
            end = (g + 1) * group_size

            # Get group weight slice: [O, group_size]
            group_weights = weight[:, start:end]

            # Compute mean across input dims: [O]
            mean_g = group_weights.mean(dim=1)

            # Get corresponding scale vector for this group: [O]
            scale_g = scales[:, g]

            # Ensure both tensors are on same device
            mean_g = mean_g.to(device)
            scale_g = scale_g.to(device)

            # Compute zero point: [O]
            zeros[:, g] = torch.floor(-mean_g / scale_g + 0.5)

            # Debug log for first few values
            if g == 0:
                logger.debug(
                    f"[compute_zeros] Group {g}: "
                    f"mean_g[0]={mean_g[0].item():.4f}, "
                    f"scale_g[0]={scale_g[0].item():.4f}, "
                    f"zero[0]={zeros[0, g].item():.2f}"
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

            # module = self.modules[idx] = self.modules[idx].to(
            #     device
            # )  # Move transformer block to GPU and update list
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
            logger.debug(f"[Block {idx}] Input features captured.")

            # 🚀 Move everything needed to GPU
            move_module_to_device(module=module, input_feat=input_feat, device="cuda")

            # Defensive check: reset or filter tensors
            if self.module_kwargs is not None:
                for k, v in self.module_kwargs.items():
                    if isinstance(v, torch.Tensor):
                        self.module_kwargs[k] = v.to(device)

                module_config = self.awq_model.get_layers_for_scaling(
                    module, input_feat, self.module_kwargs
                )
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

                # # Move RoPE back to GPU if needed
                # if is_attention_layer([str(group_name)]) or is_attention_layer(
                #     layer_names
                # ):
                #     move_rope_to_device(self.model, device)

                # 1. Compute scales per layer
                scales_dict = {}
                for layer in layers:
                    layer_name = name_to_layer.get(layer)
                    if layer_name is None:
                        raise ValueError(f"Layer not found in name mapping: {layer}")
                    logger.info(f"🔎 [scale] {layer_name}")

                    log_gpu_usage(
                        f"[calibrate] Block {idx} - before scale search ({layer_name})"
                    )  # ⬅️ log VRAM

                    best_scales = self._search_best_scale(
                        module=module,
                        prev_op=prev_op,
                        layer=layer,
                        inp=input_feat[layer_name],
                    )

                    best_scales = best_scales.to(device)
                    scales_dict[layer_name] = best_scales

                    log_gpu_usage(
                        f"[calibrate] Block {idx} - after scale search ({layer_name})"
                    )  # ⬅️ log VRAM

                    logger.debug(
                        f"[raw scales] {layer_name} → shape: {best_scales.shape}, preview: {best_scales[:5].tolist()}"
                    )

                # 2. Normalize only if this is a QKV group (by checking all layer names)
                layer_names = [name_to_layer[layer] for layer in layers]

                if is_attention_layer([str(group_name)]) or is_attention_layer(
                    layer_names
                ):
                    # if all(any(tag in name for tag in qkv_tags) for name in layer_names):
                    logger.info("🔁 Normalizing QKV group scales across shared input")
                    normalized_scales = normalize_scales_across_groups(
                        layers=layers,
                        name_to_layer=name_to_layer,
                        scales_dict=scales_dict,
                    )
                else:
                    normalized_scales = scales_dict  # no change

                # ✅ Update scale tensors with normalized layers
                for layer in layers:
                    name = name_to_layer[layer]
                    scales_dict[name] = normalized_scales[name]

                    logger.debug(
                        f"[normalized scales] {name} → shape: {scales_dict[name].shape}, "
                        f"preview: {scales_dict[name][:5].tolist()}"
                    )

                log_gpu_usage(
                    f"[calibrate] Block {idx} - after normalizing scales"
                )  # ⬅️ log VRAM

                # 3. Apply scale, compute zeros, clip, quantize, save, clear VRAM
                for layer in layers:

                    # * Save a copy of original weights
                    original_weight = layer.weight.detach().clone()
                    logger.info("Created copy of original weight.")

                    layer_name = name_to_layer[layer]

                    # Apply scales
                    scales = scales_dict[layer_name]
                    logger.info(f"🧪 [apply] scale to {layer_name}")
                    apply_scale_all_groups(layer=layer, scales=scales)
                    logger.info(f"Scales applied to {layer_name}")

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

                    # Compute zeros
                    zeros = self.compute_zeros(layer.weight, scales, self.group_size)
                    logger.debug(
                        f"[zeros] {layer_name} → {zeros.shape}, first 5: {zeros[:5].tolist()}"
                    )

                    zeros = offload_tensor_to_cpu(
                        tensor=zeros
                    )  # ✅ Move to CPU after zero computation

                    # * Restore original weights & clear VRAM cache
                    layer.weight.data.copy_(original_weight)
                    del original_weight
                    gc.collect()
                    torch.cuda.empty_cache()
                    logger.info("Original weights restored and VRAM memory cleared.")

                    # * ✅ Quantize & replace sub-layer with quantized GEMM layer

                    # ✅ Load all to GPU before GEMM creation
                    device = (
                        layer.weight.device
                        if layer.weight.is_cuda
                        else get_best_device()
                    )
                    scales = scales.to(device)
                    zeros = zeros.to(device)
                    layer = layer.to(device)

                    # ! Transpose to -> [G, O] (input feature, output feature shape)
                    scales_t = scales.t().contiguous()  # shape: [G, O]
                    zeros_t = zeros.t().contiguous()  # shape: [G, O]

                    # todo: debug; delete later
                    # 🔍 Quantization debug peek
                    logger.debug("Checking before WQLinear_GEMM:")
                    logger.debug(
                        f"[{layer_name}] weight: max={layer.weight.abs().max():.6f}, min={layer.weight.min():.6f}, shape={layer.weight.shape}"
                    )
                    logger.debug(
                        f"[{layer_name}] scales: shape={scales_t.shape}, max={scales_t.max():.6f}, min={scales_t.min():.6f}, mean={scales_t.mean():.6f}"
                    )
                    logger.debug(
                        f"[{layer_name}] zeros: shape={zeros_t.shape}, max={zeros_t.max():.2f}, min={zeros_t.min():.2f}, mean={zeros_t.float().mean():.2f}"
                    )

                    # Optional: print actual small slice
                    logger.debug(
                        f"[{layer_name}] scale[0, :5] = {scales_t[0, :5].tolist()}"
                    )
                    logger.debug(
                        f"[{layer_name}] zero[0, :5]  = {zeros_t[0, :5].tolist()}"
                    )
                    logger.debug(
                        f"[{layer_name}] weight[0, :5] = {layer.weight[0, :5].tolist()}"
                    )

                    quantized_layer = WQLinear_GEMM.from_linear(
                        linear=layer,
                        w_bit=self.w_bit,
                        group_size=self.group_size,
                        scales=scales_t,
                        zeros=zeros_t,
                    )

                    # todo: debug; delete later
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

                    # Optional: log first few values
                    logger.debug(
                        f"[{layer_name}] qweight[0, :10] = {quantized_layer.qweight[0, :10].tolist()}"
                    )
                    logger.debug(
                        f"[{layer_name}] qzeros[0, :10]  = {quantized_layer.qzeros[0, :10].tolist()}"
                    )
                    logger.debug(
                        f"[{layer_name}] scales[0, :5]   = {quantized_layer.scales[0, :5].tolist()}"
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
                    del scales, zeros, scales_t, zeros_t
                    gc.collect()
                    torch.cuda.empty_cache()

                    # ✅ NOW safe to move entire module/layer to CPU
                    layer.cpu()

                    log_gpu_usage(
                        f"[free_input] Freed input_feat for {layer_name}"
                    )  # ⬅️ log VRAM

                # Count loops
                total_layers_quantized += len(layers)

                # # Move RoPE embeddings off GPU if attention was used
                # if is_attention_layer([str(group_name)]) or is_attention_layer(
                #     layer_names
                # ):
                #     move_rope_to_device(self.model, "cpu")

            log_gpu_usage(
                f"[calibrate] Block {idx} - end of block before memory cleanup"
            )  # ⬅️ log VRAM

            clear_up_module_memory(
                module=module,
                input_feat=input_feat,
                device="cpu",
            )

            # # todo: commented out for now; delete the code later
            # # Move current module back to CPU and update list
            # module = module.cpu()
            # self.modules[idx] = module

            # # Move shared embeddings off GPU
            # self.move_embed(self.model, "cpu")

            # # Trigger cleanup
            # torch.cuda.empty_cache()
            # gc.collect()

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
        Loads sublayer weights in the exact order expected by the model,
        using its internal layer structure to reconstruct filenames.
        """
        from awq.utils.module import get_named_linears

        if not os.path.isdir(load_dir_path):
            raise FileNotFoundError(f"❌ Directory does not exist: {load_dir_path}")

        logger.info(
            f"📥 Loading quantized sublayers from {load_dir_path} using model structure."
        )

        loaded_count = 0
        for block_idx, block in enumerate(self.model.model.layers):
            named_linears = get_named_linears(block)
            for name, layer in named_linears.items():
                filename = f"model.layers.{block_idx}.{name}.pt"
                filepath = os.path.join(load_dir_path, filename)
                if not os.path.exists(filepath):
                    logger.warning(f"⚠️ Missing: {filename}")
                    continue
                try:
                    state_dict = torch.load(filepath, map_location="cpu")
                    layer.load_state_dict(state_dict)
                    logger.debug(
                        f"✅ Loaded {filename} into model.layers.{block_idx}.{name}"
                    )
                    loaded_count += 1
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load {filename}: {e}")

        logger.info(f"✅ Loaded {loaded_count} quantized sublayers into model.")

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

    # todo: commented out; this code computed a 1D vector for each group
    # todo: delete later
    # def _compute_best_scale_groupwise_old_version(
    #     self,
    #     x: torch.Tensor,
    #     w_mean: torch.Tensor,
    #     x_mean: torch.Tensor,
    #     module2inspect: torch.nn.Module,
    #     linears2scale: List[nn.Linear],
    #     fp16_output: torch.Tensor,
    #     group_idx: int,  # ✔️ Added to autoawq code
    #     group_size: int,  # ✔️ Added to autoawq code
    #     kwargs: Dict = {},
    # ):
    #     """
    #     * Replace official autoawq's _compute_best_scale method to accomodate
    #     * per group calculation efficiently.

    #     Compute loss and select best scales

    #     L(s) = || Q(W * s) (s^-1 * X) - W * X ||
    #     Q: weight quantization function | pseudo_quantize_tensor(W * s)
    #     X: inputs from calib dataset    | X
    #     W: original weights in FP16     | layer
    #     s: per channel scaling factor   | s^-1 * X

    #     Grid search to find the best quantization scale for a specific input group.

    #     This method evaluates candidate scales for a single input group (specified
    #     by `group_idx`) by temporarily modifying the corresponding slice of weight
    #     matrices in `linears2scale`. The modified layer(s) are then used in a forward
    #     pass to compute the quantized output, which is compared against the original
    #     FP16 output to compute reconstruction loss.

    #     Args:
    #         x (torch.Tensor): Full input tensor to the layer (shape: [B, in_features]).
    #         w_mean (torch.Tensor): Per-output-channel mean of normalized weights
    #             for this group (shape: [out_features]).
    #         x_mean (torch.Tensor): Scalar input mean for this group, broadcasted
    #             per output channel (shape: [out_features]).
    #         module2inspect (nn.Module): The module to forward for computing
    #             the quantized output.
    #             Typically a single Linear layer, but can also be a higher-level container.
    #         linears2scale (List[nn.Linear]): List of Linear layers in which the scale
    #             should be applied.
    #             Usually contains a single layer.
    #         fp16_output (torch.Tensor): Original output of the unquantized full layer
    #             (shape: [B, out_features]), used as the target for loss comparison.
    #         group_idx (int): Index of the input group currently being quantized.
    #         group_size (int): Number of input dimensions in each group.
    #         kwargs (Dict, optional): Additional keyword arguments passed to the module's
    #             forward method (e.g., attention masks). Defaults to empty.

    #     Returns:
    #         torch.Tensor: Optimal scale vector for this group, per output channel
    #         (shape: [out_features]).
    #     """
    #     start = group_idx * group_size
    #     end = (group_idx + 1) * group_size

    #     history = []
    #     best_ratio = -1
    #     best_scales = None
    #     best_error = float("inf")

    #     # Set n grid and tolerance depending on tensor size
    #     if fp16_output.shape[-1] >= 8192:  # For big MLPs like gate_proj/down_proj
    #         n_grid = 10
    #         early_stop_tolerance = 1.001
    #     else:
    #         n_grid = 20
    #         early_stop_tolerance = 1.05
    #     # todo: delete later
    #     # org_sd = {k: v.cpu() for k, v in module2inspect.state_dict().items()}

    #     # Cache original weights directly instead of full state_dict
    #     original_weights = {
    #         fc: fc.weight[:, group_idx * group_size : (group_idx + 1) * group_size]
    #         .detach()
    #         .clone()
    #         for fc in linears2scale
    #     }

    #     # todo: no need to call to cpu inside the loop
    #     # x = x.to("cpu")
    #     # fp16_output = fp16_output.to("cpu")
    #     # for fc in linears2scale:
    #     #     fc.cpu()

    #     # Move to the right device
    #     device = x.device
    #     x_mean = x_mean.view(-1).to(device)
    #     w_mean = w_mean.view(-1).to(device)
    #     fp16_output = fp16_output.to(device)

    #     assert not torch.isnan(w_mean).any(), "w_mean contains NaNs"  # extra guard
    #     assert not torch.isnan(x_mean).any(), "x_mean contains NaNs"  # extra guard

    #     for ratio_idx in range(n_grid):
    #         # create new scales
    #         ratio = ratio_idx / n_grid

    #         # NOTE: s^-1 * x is fused here, according to paper
    #         if self.duo_scaling:
    #             # 🧠 Core AWQ scaling: balances input and weight magnitudes
    #             scales = (
    #                 x_mean.pow(ratio_idx) / (w_mean.pow(1 - ratio_idx) + 1e-4)
    #             ).clamp(min=1e-4)
    #         else:
    #             # 🛡 Fallback if dual scaling is off: normalize x_mean-based scales
    #             scales = x_mean.pow(ratio_idx).clamp(min=1e-4)
    #             scales = scales / (scales.mean(dim=-1, keepdim=True) + 1e-6)

    #         scales = scales.clamp(min=1e-4, max=10.0)  # extra clamping to prevent drift
    #         scales_view = scales.view(-1, 1).to(
    #             device
    #         )  # ☑️ updated: flip dim to broadcast across num of groups input dims

    #         # avoid scaling values that overflow
    #         scales[torch.isinf(scales)] = 1
    #         scales[torch.isnan(scales)] = 1

    #         # Q(W * s)
    #         # inplace quantization + manual restore (no state_dict reload)
    #         for fc in linears2scale:
    #             with torch.no_grad():
    #                 # Restore original slice
    #                 fc.weight[:, start:end].copy_(original_weights[fc])
    #                 # Apply scaling and quantization
    #                 fc.weight[:, start:end].mul_(scales_view)
    #                 fc.weight[:, start:end].copy_(
    #                     self.pseudo_quantize_tensor(fc.weight[:, start:end])[0]
    #                     / scales_view
    #                 )

    #             # todo: commented out; too computational intensive
    #             # fc.weight[:, start:end].mul_(scales_view)  # ☑️ updated to slice weights
    #             # fc.weight.data[:, start:end] = (
    #             #     self.pseudo_quantize_tensor(fc.weight.data[:, start:end])[0]
    #             #     / scales_view
    #             # )  # ☑️ updated to slice weights
    #             # logger.debug(f"Group {group_idx}: scale shape = {scales_view.shape}")

    #         # W * X
    #         int_w_output = self._module_forward(x, module2inspect, kwargs)
    #         int_w_output = int_w_output.clip(
    #             torch.finfo(int_w_output.dtype).min, torch.finfo(int_w_output.dtype).max
    #         )  # clamp
    #         int_w_output = int_w_output.to(device)  # ☑️ use dynamic device

    #         # compute mean squared error (L2 norm)
    #         loss = self._compute_loss(fp16_output, int_w_output, device)

    #         history.append(loss)
    #         if loss < best_error:
    #             best_error = loss
    #             best_ratio = ratio_idx
    #             best_scales = scales.clone()
    #             logger.debug(
    #                 f"[Group {group_idx}] New best ratio = {ratio:.3f}, loss = {loss:.6f}"
    #             )
    #         # module2inspect.load_state_dict(org_sd) # No need to load this

    #         # ✅ Early stopping if loss isn't improving much
    #         elif loss > best_error * early_stop_tolerance:
    #             logger.debug(
    #                 f"[Group {group_idx}] Early exit at ratio {ratio:.3f}, loss = {loss:.6f}"
    #             )
    #             break

    #     if best_ratio == -1:
    #         logger.error(f"No valid scale found for group. Loss history: {history}")
    #         raise Exception

    #     assert best_scales is not None  # Add extra guard
    #     assert torch.isnan(best_scales).sum() == 0, best_scales

    #     return best_scales.detach().to(device)
