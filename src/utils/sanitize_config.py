import json
from pathlib import Path
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def sanitize_config(
    config_path_or_dir: str | Path,
    allowed_versions: Optional[list[str]] = None,
    default_bits: int = 4,
    default_group_size: int = 128,
    default_version: str = "gemm",
    required_fields: Optional[list[str]] = None,
) -> Path:
    """
    Normalize and verify quantization config inside config.json for AWQ loading.

    Args:
        config_path_or_dir (str | Path): Either path to `config.json` or model dir containing it.
        allowed_versions (list[str] | None): Valid values for version (e.g., ['gemm', 'gemv']).
        default_bits (int): Default bit-width (used as `bits`) if missing or incorrect.
        default_group_size (int): Default group size if missing or incorrect.
        default_version (str): Default version if missing or invalid.
        required_fields (list[str] | None): Fields required inside quantization_config.

    Returns:
        Path: Final resolved path to the sanitized config file.

    Raises:
        FileNotFoundError: If config.json doesn't exist.
        ValueError: If required fields are missing.
    """
    path = Path(config_path_or_dir).expanduser().resolve()
    config_path = path if path.is_file() else path / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"❌ config.json not found at: {config_path}")

    with config_path.open("r") as f:
        config = json.load(f)

    qc = config.get("quantization_config", {})
    changes_made = False

    try:
        if not isinstance(qc.get("bits"), int):
            qc["bits"] = int(qc.get("bits", default_bits))
            logger.warning("⚠️ Coerced bits to int.")
            changes_made = True

        if not isinstance(qc.get("group_size"), int):
            qc["group_size"] = int(qc.get("group_size", default_group_size))
            logger.warning("⚠️ Coerced group_size to int.")
            changes_made = True

        version = str(qc.get("version", default_version)).lower()
        if allowed_versions and version not in allowed_versions:
            logger.warning(
                f"⚠️ Unknown version '{version}' — fallback to '{default_version}'"
            )
            version = default_version
        if qc.get("version") != version:
            qc["version"] = version
            changes_made = True

        if "zero_point" not in qc:
            qc["zero_point"] = True
            changes_made = True
        else:
            qc["zero_point"] = bool(qc["zero_point"])

        config["quantization_config"] = qc

        if "_name_or_path" in config:
            del config["_name_or_path"]
            changes_made = True

    except Exception as e:
        logger.error(f"❌ Failed to normalize config: {e}")
        raise

    required_fields = required_fields or ["bits", "group_size", "version", "zero_point"]
    missing = [f for f in required_fields if f not in qc]
    if missing:
        raise ValueError(
            f"❌ config.json missing required quantization_config fields: {missing}"
        )

    if changes_made:
        with config_path.open("w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"✅ Patched and saved config.json → {config_path}")
    else:
        logger.info(f"ℹ️ config.json already clean → {config_path}")

    return config_path
