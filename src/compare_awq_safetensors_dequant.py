import sys
import torch
from typing import Dict, List, Tuple, Any, Optional, TextIO
from safetensors.torch import load_file
from pathlib import Path
from utils.awq_tensor_utils import (
    unpack_qweight_4bit_int32,
    unpack_qzeros_4bit_int32,
    dequantize_qweights,
)


def stat(t: torch.Tensor) -> Dict[str, Any]:
    """
    Compute basic statistics for a tensor (min, max, mean, std, shape, etc.).

    Args:
        t: Input tensor.

    Returns:
        Dictionary of statistics for the input tensor.
    """
    t = t.float()
    return {
        "shape": tuple(t.shape),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "mean": float(t.mean().item()),
        "std": float(t.std().item()),
        "dtype": str(t.dtype),
        "zeros": int((t == 0).sum().item()),
        "nans": int(t.isnan().sum().item() if hasattr(t, "isnan") else 0),
    }


def compare_stats(
    s1: Dict[str, Any], s2: Dict[str, Any], threshold: float = 2.0
) -> Optional[str]:
    """
    Compare statistics between two tensors and flag if differences exceed threshold.

    Args:
        s1: Stats dict from `stat()` for tensor 1.
        s2: Stats dict from `stat()` for tensor 2.
        threshold: Relative difference threshold for flagging.

    Returns:
        Description of flagged differences, or None if all are within threshold.
    """
    issues = []
    for statname in ["mean", "std", "min", "max"]:
        v1, v2 = abs(s1[statname]), abs(s2[statname])
        if (v1 == 0 and v2 != 0) or (v2 == 0 and v1 != 0):
            issues.append(f"{statname.upper()} all zero/NaN in one model")
        elif min(v1, v2) > 0 and max(v1, v2) / min(v1, v2) > threshold:
            issues.append(
                f"{statname.upper()} differs by >{threshold}x ({s1[statname]:.2g} vs {s2[statname]:.2g})"
            )
    return "; ".join(issues) if issues else None


def find_matching_layers(
    a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]
) -> List[str]:
    """
    Identify base names of quantized layers present (with all required tensors)
    in both models.

    Args:
        a: Dict of tensors from model A.
        b: Dict of tensors from model B.

    Returns:
        List of layer base names present in both models.
    """
    layers_a = set(k.rsplit(".qweight", 1)[0] for k in a if k.endswith(".qweight"))
    layers_b = set(k.rsplit(".qweight", 1)[0] for k in b if k.endswith(".qweight"))
    candidates = sorted(layers_a & layers_b)
    matching = []
    for base in candidates:
        if all(
            base + suf in a and base + suf in b
            for suf in [".qweight", ".qzeros", ".scales"]
        ):
            matching.append(base)
    return matching


def unpack_and_dequantize_model(
    model: Dict[str, torch.Tensor], layers: List[str]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, Any]]]:
    """
    For each layer:
      - Unpacks qweight and qzeros from AWQ-packed int32 format.

      - Expects all tensors in **mathematical shape convention** before unpack/transpose:
          * qweight: [in_features_packed, out_features], int32 (packed 4-bit)
          * qzeros:  [in_features_packed, num_groups], int32 (packed 4-bit)
          * scales:  [in_features, num_groups], float16 or float32

      - Unpacks using provided utils:
          * unpack_qweight_4bit_int32(qweight): returns [in_features, out_features], int8
          * unpack_qzeros_4bit_int32(qzeros): returns [in_features, num_groups], int8

      - Transposes all unpacked tensors to standard PyTorch shape convention:
          * qweight: [out_features, in_features], int8
          * qzeros:  [out_features, num_groups], int8 or uint8
          * scales:  [out_features, num_groups], float32

      - Infers group size per layer (group_size = in_features // num_groups).
      - Dequantizes using groupwise scales and (optionally) zero points.
      - Skips layers with incompatible or unexpected shapes.

    Args:
        model: Dict mapping tensor names to tensors, where:
            * '{base}.qweight': [in_features_packed, out_features], int32
            * '{base}.qzeros':  [in_features_packed, num_groups], int32
            * '{base}.scales':  [in_features, num_groups], float16 or float32
        layers: List of layer base names (excluding .qweight, .qzeros, .scales).

    Returns:
        Tuple:
            - Dict[str, torch.Tensor]: Dequantized float32 weights for each layer,
                [out_features, in_features]
            - Dict[str, Dict[str, torch.Tensor]]: Unpacked (and transposed) tensors
                per layer:
                * 'qweight': [out_features, in_features], int8
                * 'qzeros':  [out_features, num_groups], int8 or uint8
                * 'scales':  [out_features, num_groups], float32

    Raises:
        ValueError: On shape mismatch or incompatible dimensions.
        TypeError: On dtype mismatch.

    Example:
        For a layer 'model.layers.0.self_attn.q_proj':
          model['model.layers.0.self_attn.q_proj.qweight']:
            [in_features_packed, out_features], int32
          model['model.layers.0.self_attn.q_proj.qzeros']:
            [in_features_packed, num_groups], int32
          model['model.layers.0.self_attn.q_proj.scales']:
            [in_features, num_groups], float16/float32

        After unpacking and transposing, qweight: [out_features, in_features], etc.
    """
    deq_weights = {}
    unpacked_artifacts = {}

    for base in layers:
        try:
            qw_raw = model[base + ".qweight"]  # [in_features_packed, out_features]
            qz_raw = model[base + ".qzeros"]  # [in_features_packed, num_groups]
            sc_raw = model[base + ".scales"]  # [in_features, num_groups] or similar

            # Unpack (still in math convention: in_features first)
            qw = unpack_qweight_4bit_int32(qw_raw)  # [in_features, out_features], int8
            qz = unpack_qzeros_4bit_int32(qz_raw)  # [in_features, num_groups], int8
            sc = (
                sc_raw.float() if sc_raw.dtype != torch.float32 else sc_raw
            )  # [in_features, num_groups], float32

            # Transpose to PyTorch convention ([out_features, in_features], etc.)
            qw = qw.T.contiguous()  # [out_features, in_features], int8
            qz = (
                qz.T.contiguous()
            )  # [num_groups, in_features] → [in_features, num_groups] if needed
            sc = (
                sc.T.contiguous()
            )  # [num_groups, in_features] → [in_features, num_groups] if needed

            # Now align all to [out_features, ...] on axis 0
            O, I = qw.shape
            G = sc.shape[0]
            if O != sc.shape[0]:
                # If after transpose, sc is [num_groups, in_features], swap
                if sc.shape[1] == O:
                    sc = sc.T.contiguous()
                    qz = qz.T.contiguous()
                else:
                    raise ValueError(
                        f"Can't align scales/qzeros for {base}: {sc.shape}, {qz.shape}, {qw.shape}"
                    )
            # Now, should have [out_features, in_features], [out_features, num_groups]
            O, I = qw.shape
            O2, G = sc.shape
            if O != O2:
                print(f"Skipping {base}: out_features mismatch {O} vs {O2}")
                continue
            if I % G != 0:
                print(
                    f"Skipping {base}: in_features ({I}) not divisible by num_groups ({G})"
                )
                continue
            if qz.shape != sc.shape:
                print(
                    f"Skipping {base}: qzeros shape {qz.shape} does not match scales {sc.shape}"
                )
                continue

            group_size = I // G

            print(
                f"[{base}] qweight.shape: {qw.shape}, qzeros.shape: {qz.shape}, scales.shape: {sc.shape}, group_size: {group_size}"
            )

            deq = dequantize_qweights(qw, sc, group_size, qz)
            deq_weights[base] = deq
            unpacked_artifacts[base] = {"qweight": qw, "qzeros": qz, "scales": sc}
        except Exception as e:
            print(f"Skipping {base}: {e}")

    return deq_weights, unpacked_artifacts


def compare_dequantized_weights(
    deq_a: Dict[str, torch.Tensor],
    deq_b: Dict[str, torch.Tensor],
    threshold: float = 2.0,
) -> Dict[str, str]:
    results = {}
    for base in deq_a.keys():
        if base not in deq_b:
            results[base] = "Missing in model B"
            continue
        stat_a = stat(deq_a[base])
        stat_b = stat(deq_b[base])
        issues = compare_stats(stat_a, stat_b, threshold)
        if issues:
            results[base] = issues
    return results


def summarize_results(
    layers: List[str],
    issues: Dict[str, str],
    output: TextIO,
) -> None:
    total = len(layers)
    flagged = len(issues)
    print("\n===== SUMMARY TABLE =====", file=output)
    print(f"Total quantized layers compared: {total}", file=output)
    print(f"Layers flagged: {flagged}", file=output)
    if flagged:
        print("\nFlagged layers:", file=output)
        for base, msg in issues.items():
            print(f"  ⚠️  {base}: {msg}", file=output)
    else:
        print("All layers are within threshold.", file=output)


def do_comparison(
    a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor], out_stream: TextIO
):
    # 1. Find matching layers
    matching_layers = find_matching_layers(a, b)
    print(f"Found {len(matching_layers)} matching quantized layers.", file=out_stream)

    # 2. Unpack and dequantize everything
    deq_a, art_a = unpack_and_dequantize_model(a, matching_layers)
    deq_b, art_b = unpack_and_dequantize_model(b, matching_layers)

    # 3. Compare
    flagged_issues = compare_dequantized_weights(deq_a, deq_b, threshold=2.0)

    # 4. Print stats for each layer
    for base in matching_layers:
        if base not in deq_a or base not in deq_b:
            print(
                f"Skipping stats for {base}: not present in one or both models after unpack/dequant.",
                file=out_stream,
            )
            continue
        print(f"\nLayer: {base}", file=out_stream)
        print("Model A stats:", stat(deq_a[base]), file=out_stream)
        print("Model B stats:", stat(deq_b[base]), file=out_stream)
        if base in flagged_issues:
            print(f"  ⚠️  Issues: {flagged_issues[base]}", file=out_stream)
        else:
            print("  OK: No significant issues.", file=out_stream)

    # 5. Summary
    summarize_results(matching_layers, flagged_issues, out_stream)


def main():
    """
    Modular AWQ safetensors model comparison tool.
    Usage: python compare_awq_safetensors_dequant.py <model_a.safetensors>
    <model_b.safetensors> [output_file]
    """
    usage = "Usage: python compare_awq_safetensors_dequant.py <model_a.safetensors> \
<model_b.safetensors> [output_file]"
    if not (3 <= len(sys.argv) <= 4):
        print(usage)
        sys.exit(1)
    path_a, path_b = sys.argv[1], sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) == 4 else None

    print(f"Loading Model A: {Path(path_a).name}")
    a = load_file(path_a)
    print(f"Loading Model B: {Path(path_b).name}")
    b = load_file(path_b)

    if out_path:
        with open(out_path, "w") as out_stream:
            do_comparison(a, b, out_stream)
        print(f"\n✅ Full comparison written to {out_path}")
    else:
        do_comparison(a, b, sys.stdout)


if __name__ == "__main__":
    main()
