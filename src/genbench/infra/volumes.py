"""Modal volume definitions and mount path constants."""

import modal

from genbench.config import DATASETS_PATH, MODELS_PATH, REFERENCE_PATH, RESULTS_PATH

reference_vol = modal.Volume.from_name("genbench-reference", create_if_missing=True)
datasets_vol = modal.Volume.from_name("genbench-datasets", create_if_missing=True)
results_vol = modal.Volume.from_name("genbench-results", create_if_missing=True)
models_vol = modal.Volume.from_name("genbench-models", create_if_missing=True)

VOLUME_MOUNTS = {
    REFERENCE_PATH: reference_vol,
    DATASETS_PATH: datasets_vol,
    RESULTS_PATH: results_vol,
    MODELS_PATH: models_vol,
}
