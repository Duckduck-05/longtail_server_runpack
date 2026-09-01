import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from imbdiff_cm.metrics import (
    calculate_frechet_distance,
    polynomial_mmd_kid,
)


def load_features(feature_dir, prefix):
    return np.load(Path(feature_dir) / f"{prefix}_2048.npy")


def fid(gen, real):
    return calculate_frechet_distance(
        np.mean(gen, axis=0),
        np.cov(gen, rowvar=False),
        np.mean(real, axis=0),
        np.cov(real, rowvar=False),
    )


def summarize(values):
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "all": [float(v) for v in values]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", default="features")
    parser.add_argument("--generated_prefix", required=True)
    parser.add_argument("--real_prefix", default="real_cifar100_train")
    parser.add_argument("--output", default=None)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--overall_samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    gen = load_features(args.feature_dir, args.generated_prefix)
    real = load_features(args.feature_dir, args.real_prefix)
    if gen.shape[0] < 2 or real.shape[0] < 2:
        raise ValueError("FID and KID require at least two real and generated feature vectors.")
    fids, kids = [], []

    for _ in range(args.repeats):
        sample_size = min(args.overall_samples, gen.shape[0])
        idx = np.random.choice(gen.shape[0], size=sample_size, replace=False)
        gen_sub = gen[idx]
        fids.append(fid(gen_sub, real))
        kids.append(polynomial_mmd_kid(gen_sub, real))

    results = {
        "FID": summarize(fids),
        "KID": summarize(kids),
    }

    print("------------summary-----------------------------")
    for key, value in results.items():
        print(f"{key}: {value['mean']}, {value['std']}, all: {value['all']}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
