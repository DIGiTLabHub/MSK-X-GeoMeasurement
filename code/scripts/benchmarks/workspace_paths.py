"""Portable storage paths; input records are separate from generated outputs."""
import os
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("MSK_WORKSPACE", str(CODE_ROOT))).expanduser().resolve()


def split_manifest(dataset):
    """Return the immutable split manifest distributed with this release."""
    return CODE_ROOT / "records" / "controlled_cross_model_v2" / dataset / f"{dataset}_results.json"
