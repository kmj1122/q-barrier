"""Validate evaluation checkpoint paths before constructing environments."""
from pathlib import Path


def require_evaluation_checkpoint(config) -> str:
    """Resolve an explicit path or the original run/epoch layout and fail fast.

    This checks file existence only. Load checkpoints exclusively from sources
    you trust; model deserialization is performed by the original AMAGO loader.
    """
    explicit_path = config.get("load_checkpoint_path")
    if explicit_path:
        path = Path(str(explicit_path)).expanduser().resolve()
    else:
        epoch = config.get("load_checkpoint_epoch")
        if epoch is None:
            raise ValueError(
                "Evaluation requires a trained checkpoint. Pass "
                "load_checkpoint_path=/path/to/policy_epoch_N.pt. "
                "Bundled weights and compatible presets are in pretrained/."
            )
        path = (
            Path(str(config.get("ckpt_base_dir", "./checkpoints")))
            / str(config.get("run_name", ""))
            / "ckpts" / "policy_weights" / f"policy_epoch_{int(epoch)}.pt"
        ).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Evaluation checkpoint not found: {path}. "
            "Pass load_checkpoint_path to a trusted trained checkpoint."
        )
    return str(path)
