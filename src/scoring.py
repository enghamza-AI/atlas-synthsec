
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

MODEL_FEATURE_COLS_BASE = [
    "audience_size",
    "weeks_active",
    "total_spend",
    "total_impressions",
    "total_clicks",
    "total_conversions",
    "ctr",
    "cost_per_click",
    "cost_per_conversion",
    "naive_roas",
    "spend_share_pct",
]


def train_impact_model(features: pd.DataFrame, config: dict) -> dict:

    scoring_cfg = config["scoring"]

 
    trainable = features[features["has_incrementality_label"]].copy()

   
    channel_dummies = pd.get_dummies(trainable["channel"], prefix="channel")
    trainable = pd.concat([trainable, channel_dummies], axis=1)
    channel_cols = list(channel_dummies.columns)

    feature_cols = MODEL_FEATURE_COLS_BASE + channel_cols
    feature_cols = [c for c in feature_cols if c in trainable.columns]

    X = trainable[feature_cols].to_numpy()
    y = trainable["estimated_incremental_roas"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=scoring_cfg["test_size"],
        random_state=scoring_cfg["random_state"],
    )

    model = RandomForestRegressor(
        n_estimators=scoring_cfg["n_estimators"],
        max_depth=scoring_cfg["max_depth"],
        random_state=scoring_cfg["random_state"],
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, preds)

    return {
        "model": model,
        "feature_cols": feature_cols,
        "channel_cols": channel_cols,
        "test_mae": round(float(test_mae), 2),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def predict_impact(features: pd.DataFrame, model_bundle: dict) -> pd.DataFrame:

    feature_cols = model_bundle["feature_cols"]
    channel_cols = model_bundle["channel_cols"]
    model = model_bundle["model"]

    df = features.copy()
    channel_dummies = pd.get_dummies(df["channel"], prefix="channel")
    for col in channel_cols:
        if col not in channel_dummies.columns:
            channel_dummies[col] = 0
    df_with_dummies = pd.concat([df, channel_dummies], axis=1)

    X = df_with_dummies[feature_cols].to_numpy()
    preds = model.predict(X)

    out = df.copy()
    out["predicted_incremental_roas"] = np.round(preds, 2)


    out["impact_score"] = np.where(
        out["has_incrementality_label"],
        out["estimated_incremental_roas"],
        out["predicted_incremental_roas"],
    ).round(2)

    return out


if __name__ == "__main__":
  
    from config_loader import load_config
    from synthetic_data import generate_synthetic_campaigns
    from cleaning import clean_campaigns
    from feature_engineering import build_campaign_features
    from segmentation import segment_campaigns

    cfg = load_config()
    raw = generate_synthetic_campaigns(cfg)
    cleaned = clean_campaigns(raw["campaigns"], raw["weekly_performance"], cfg)
    features = build_campaign_features(cleaned["campaigns"], cleaned["weekly_performance"], cfg)
    segmented = segment_campaigns(features, cfg)

    bundle = train_impact_model(segmented, cfg)
    print(f"Trained on {bundle['n_train']} holdout-tested campaigns, "
          f"tested on {bundle['n_test']}, MAE = {bundle['test_mae']} ROAS points")

    scored = predict_impact(segmented, bundle)
    print(scored[["campaign_id", "channel", "segment_name", "naive_roas",
                   "impact_score"]].sort_values("impact_score", ascending=False).head())
