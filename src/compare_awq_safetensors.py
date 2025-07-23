"compare_awq_safetensors.py"

from safetensors.torch import load_file
import sys
from pathlib import Path


def stat(t):
    t = t.float()
    return {
        "shape": tuple(t.shape),
        "min": t.min().item(),
        "max": t.max().item(),
        "mean": t.mean().item(),
        "std": t.std().item(),
        "dtype": str(t.dtype),
        "zeros": (t == 0).sum().item(),
        "nans": t.isnan().sum().item() if hasattr(t, "isnan") else 0,
    }


def pretty_stats(stats):
    return ", ".join(f"{k}: {v}" for k, v in stats.items())


def flag_layer(key, s1, s2, threshold=2.0):
    # Returns a message or None if no flag
    issues = []
    if s1["shape"] != s2["shape"]:
        issues.append("SHAPE mismatch")
    for statname in ["zeros", "nans"]:
        if s1[statname] > 0 or s2[statname] > 0:
            issues.append(f"{statname.upper()} detected")
    for statname in ["mean", "std", "min", "max"]:
        v1, v2 = abs(s1[statname]), abs(s2[statname])
        denom = max(v1, v2, 1e-6)
        # Flag if either is zero and the other is nonzero (all zeros or NaNs)
        if (v1 == 0 and v2 != 0) or (v2 == 0 and v1 != 0):
            issues.append(f"{statname.upper()} all zero/NaN in one model")
        # Flag if > 2x diff (but only if both not very close to zero)
        elif min(v1, v2) > 0 and max(v1, v2) / min(v1, v2) > threshold:
            issues.append(
                f"{statname.upper()} differs by >{threshold}x ({s1[statname]:.2g} vs {s2[statname]:.2g})"
            )
    return "; ".join(issues) if issues else None


def compare_dicts(a, b, output, summary_flags, threshold=2.0):
    a_keys, b_keys = list(a.keys()), list(b.keys())
    all_keys = sorted(set(a_keys) | set(b_keys))
    order_ok = a_keys == b_keys

    if not order_ok:
        print("\n❗Order of keys differs!", file=output)
        print("First 10 keys in A:", a_keys[:10], file=output)
        print("First 10 keys in B:", b_keys[:10], file=output)
    else:
        print("✅ Order of layer keys matches.", file=output)

    missing_in_b = [k for k in a_keys if k not in b]
    missing_in_a = [k for k in b_keys if k not in a]
    if missing_in_b:
        print(f"❌ Keys in A but missing in B ({len(missing_in_b)}):", file=output)
        for k in missing_in_b:
            print("  " + k, file=output)
    if missing_in_a:
        print(f"❌ Keys in B but missing in A ({len(missing_in_a)}):", file=output)
        for k in missing_in_a:
            print("  " + k, file=output)

    print(
        "\nComparing matching keys - detailed (qweight, scales, qzeros):", file=output
    )
    for key in all_keys:
        if (
            key in a
            and key in b
            and any(x in key for x in ["qweight", "scales", "qzeros"])
        ):
            s1, s2 = stat(a[key]), stat(b[key])
            layer_flag = flag_layer(key, s1, s2, threshold)
            if layer_flag:
                summary_flags.append((key, layer_flag))
            # Always print details
            print(
                f"{'⚠️' if layer_flag else 'OK'}  {key}: {layer_flag or 'No major issues'}",
                file=output,
            )
            print(f"    A: {pretty_stats(s1)}", file=output)
            print(f"    B: {pretty_stats(s2)}", file=output)
    print("\nComparison complete.", file=output)


if __name__ == "__main__":
    usage = "Usage: python compare_awq_safetensors_with_summary.py <path_scrooge> <path_hansen> [output_file]"
    if not (3 <= len(sys.argv) <= 4):
        print(usage)
        sys.exit(1)
    path_a, path_b = sys.argv[1], sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) == 4 else None
    print(f"Loading Scrooge: {Path(path_a).name}")
    a = load_file(path_a)
    print(f"Loading Hansen: {Path(path_b).name}")
    b = load_file(path_b)
    summary_flags = []
    # Write to file or stdout
    out_stream = open(out_path, "w") if out_path else sys.stdout
    try:
        # Reserve room for summary, fill in later
        print("===== QUANTIZED LLM COMPARISON SUMMARY =====", file=out_stream)
        print("Summary of flagged issues:", file=out_stream)
        print("(Details for each flagged layer are printed below)\n", file=out_stream)
        print("=" * 48 + "\n", file=out_stream)
        compare_dicts(a, b, out_stream, summary_flags)
        # Write summary at the top (if writing to file, re-open and insert)
        if out_path:
            out_stream.close()
            lines = open(out_path).readlines()
            summary = ["\n===== FLAGGED LAYERS (POTENTIAL ISSUES) =====\n"]
            if summary_flags:
                for key, flag in summary_flags:
                    summary.append(f"⚠️  {key}: {flag}\n")
            else:
                summary.append("No layers flagged as potential issues. All good!\n")
            # Insert after the summary header (line 3)
            out_idx = 3
            lines = lines[:out_idx] + summary + lines[out_idx:]
            with open(out_path, "w") as f:
                f.writelines(lines)
            print(f"\n✅ Comparison with summary written to {out_path}")
        else:
            print("\n===== FLAGGED LAYERS (POTENTIAL ISSUES) =====")
            if summary_flags:
                for key, flag in summary_flags:
                    print(f"⚠️  {key}: {flag}")
            else:
                print("No layers flagged as potential issues. All good!")
    finally:
        if out_path and not out_stream.closed:
            out_stream.close()
