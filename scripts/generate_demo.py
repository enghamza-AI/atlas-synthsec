
from __future__ import annotations

from pathlib import Path
import sys

import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config
from src.pipeline import run_pipeline


OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "demo"
    / "atlas_results.pkl"
)


def main() -> None:
    config = load_config()

    print("Running Atlas pipeline...")
    output = run_pipeline(config, data_mode="demo")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(output, OUTPUT_PATH)

    print()
    print("Atlas demo results saved successfully.")
    print(f"File: {OUTPUT_PATH}")
    print(f"Campaigns: {output['n_campaigns']:,}")
    print(f"Model MAE: {output['model_mae']}")
    print(f"Silhouette score: {output['silhouette_avg']}")
    print(f"% with measured label: {output['pct_measured']:.1%}")


if __name__ == "__main__":
    main()

