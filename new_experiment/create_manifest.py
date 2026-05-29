# generate_manifest.py
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import (
    DATASET_INDEX, DATASET_LIST, GLOBAL_SEEDS,
    TRAIN_TEST_RATIOS, TRAINING_DISTRIBUTIONS, TEST_DISTRIBUTIONS
)

lines = []
for dta_name in DATASET_LIST:
    n_classes = int(DATASET_INDEX.loc[dta_name, "classes"])
    train_ds = TRAINING_DISTRIBUTIONS[n_classes]
    test_ds = TEST_DISTRIBUTIONS[n_classes]
    for seed_idx, seed in enumerate(GLOBAL_SEEDS):
        for dt_idx in range(len(TRAIN_TEST_RATIOS)):
            for tr_idx in range(len(train_ds)):
                for te_idx in range(len(test_ds)):
                    lines.append(f"{dta_name},{seed},{dt_idx},{tr_idx},{te_idx}")

with open("manifest.txt", "w") as f:
    f.write("\n".join(lines))
print(f"Total work units: {len(lines)}")