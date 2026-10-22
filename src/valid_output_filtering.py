from pathlib import Path
import argparse
import numpy as np

from emma import EmmaDataset, HDF5Source
import hdf5plugin

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset-root",
    type=Path,
    required=True,
    help="Directory containing the EMMa dataset.",
)
parser.add_argument(
    "--augmentation-file",
    type=Path,
    required=True,
    help="Path to the SAM3 augmentation HDF5 file.",
)
args = parser.parse_args()

DATASET_ROOT = args.dataset_root
SAM3_PATH = args.augmentation_file

AUGMENTATION_DIR = SAM3_PATH.parent
VALID_ASINS_PATH = AUGMENTATION_DIR / "valid_asins.npy"

ds = EmmaDataset(DATASET_ROOT)

source = HDF5Source("sam3_thresholding_masks", SAM3_PATH, mode="r")
asins = set(source.keys())

def has_valid_mask(source, asin):
    # No filtering
    return True

valid_asins = [
    asin
    for asin in asins
    if has_valid_mask(source, asin)
]
valid_asins = np.asarray(valid_asins, dtype=str)

if not len(valid_asins):
    raise RuntimeError("No valid ASINs.")

print(f"Augmentation ASINs: {len(asins)}")
print(f"Excluded ASINs: {len(asins) - len(valid_asins)}")

AUGMENTATION_DIR.mkdir(parents=True, exist_ok=True)
np.save(VALID_ASINS_PATH, valid_asins)
print(f"Saved {len(valid_asins)} ASINs to: {VALID_ASINS_PATH}")