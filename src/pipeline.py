

from __future__ import annotations

import pandas as pd

from src.config_loader import load_config
from src.synthetic_data import generate_synthetic_campaigns
from src.cleaning import clean_campaigns
from src.feature_engineering import build_campaign_features
from src.segmentation import segment_campaigns
from src.scoring import train_impact_model, predict_impact
from src.recommendations import generate_recommendations


def run_pipeline(config: dict, data_mode: str = "demo") -> dict:

    if data_mode == "demo":
        campaigns_path = config["app"]["demo_csv_path"]
        weekly_path = campaigns_path.replace("demo_sample.csv", "demo_sample_weekly.csv")
        raw_campaigns = pd.read_csv(campaigns_path, parse_dates=["start_date"])
        raw_weekly = pd.read_csv(weekly_path, parse_dates=["week_date"])
    elif data_mode == "local":
        generated = generate_synthetic_campaigns(config)
        raw_campaigns = generated["campaigns"]
        raw_weekly = generated["weekly_performance"]
    else:
        raise ValueError(
            f"Unknown data_mode '{data_mode}'. Expected 'demo' or 'local'."
        )

    cleaned = clean_campaigns(raw_campaigns, raw_weekly, config)
    features = build_campaign_features(
        cleaned["campaigns"], cleaned["weekly_performance"], config
    )
    segmented = segment_campaigns(features, config)
    model_bundle = train_impact_model(segmented, config)
    scored = predict_impact(segmented, model_bundle)
    result = generate_recommendations(scored, config)

    return {
        "result": result,
        "model_mae": model_bundle["test_mae"],
        "silhouette_avg": float(result["silhouette_avg"].iloc[0]),
        "n_campaigns": len(result),
        "pct_measured": float(result["has_incrementality_label"].mean()),
    }


if __name__ == "__main__":
   
    cfg = load_config()
    output = run_pipeline(cfg, data_mode="local")
    print(f"Scored {output['n_campaigns']:,} campaigns")
    print(f"Model MAE: {output['model_mae']} ROAS points")
    print(f"Silhouette score: {output['silhouette_avg']}")
    print(f"% with a real measured label: {output['pct_measured']:.1%}")
    print(output["result"]["segment_name"].value_counts())
