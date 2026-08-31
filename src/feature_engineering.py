

from __future__ import annotations

import numpy as np
import pandas as pd


def build_campaign_features(
    campaigns: pd.DataFrame, weekly: pd.DataFrame, config: dict
) -> pd.DataFrame:

    fe_cfg = config["feature_engineering"]
    min_holdout_size = fe_cfg["min_holdout_size_for_label"]
    avg_order_value = config["synthetic_data"]["avg_order_value"]


    rollup = weekly.groupby("campaign_id").agg(
        weeks_active=("week_number", "count"),
        total_spend=("spend", "sum"),
        total_impressions=("impressions", "sum"),
        total_clicks=("clicks", "sum"),
        total_conversions=("conversions", "sum"),
    ).reset_index()

    features = campaigns.merge(rollup, on="campaign_id", how="inner")

    features["ctr"] = np.where(
        features["total_impressions"] > 0,
        features["total_clicks"] / features["total_impressions"],
        0.0,
    )
    features["cost_per_click"] = np.where(
        features["total_clicks"] > 0,
        features["total_spend"] / features["total_clicks"],
        np.nan,
    )
    features["cost_per_conversion"] = np.where(
        features["total_conversions"] > 0,
        features["total_spend"] / features["total_conversions"],
        np.nan,
    )

  
    features["naive_roas"] = np.where(
        features["total_spend"] > 0,
        (features["total_conversions"] * avg_order_value) / features["total_spend"],
        np.nan,
    )

    total_spend_all = features["total_spend"].sum()
    features["spend_share_pct"] = round(
        100 * features["total_spend"] / total_spend_all, 2
    )

 
    features["has_incrementality_label"] = (
        features["has_holdout_test"] & (features["holdout_size"] >= min_holdout_size)
    )

   
    exposed_size = features["audience_size"] - features["holdout_size"]
    expected_organic_conversions = features["measured_organic_rate"] * exposed_size

    features["estimated_incremental_conversions"] = np.where(
        features["has_incrementality_label"],
        features["total_conversions"] - expected_organic_conversions,
        np.nan,
    )
 
    features["estimated_incremental_conversions"] = features[
        "estimated_incremental_conversions"
    ].clip(lower=0)

    features["estimated_incremental_roas"] = np.where(
        features["has_incrementality_label"] & (features["total_spend"] > 0),
        (features["estimated_incremental_conversions"] * avg_order_value)
        / features["total_spend"],
        np.nan,
    )

    return features


if __name__ == "__main__":
 
    from config_loader import load_config
    from synthetic_data import generate_synthetic_campaigns
    from cleaning import clean_campaigns

    cfg = load_config()
    raw = generate_synthetic_campaigns(cfg)
    cleaned = clean_campaigns(raw["campaigns"], raw["weekly_performance"], cfg)
    features = build_campaign_features(cleaned["campaigns"], cleaned["weekly_performance"], cfg)
    print(f"Built features for {len(features):,} campaigns")
    print(f"Campaigns with a usable incrementality label: "
          f"{features['has_incrementality_label'].sum():,} "
          f"({features['has_incrementality_label'].mean():.1%})")
    print(features[["channel", "naive_roas", "estimated_incremental_roas"]]
          .dropna(subset=["estimated_incremental_roas"])
          .groupby("channel").mean().round(2))
