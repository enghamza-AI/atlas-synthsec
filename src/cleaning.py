

from __future__ import annotations

import pandas as pd


def clean_campaigns(campaigns: pd.DataFrame, weekly: pd.DataFrame, config: dict) -> dict:
   
    rules = config["cleaning"]

    campaigns_df = campaigns.copy()
    required_campaign_cols = ["campaign_id", "channel", "start_date",
                               "duration_weeks", "audience_size", "has_holdout_test"]
    campaigns_df = campaigns_df.dropna(subset=required_campaign_cols)
    campaigns_df["start_date"] = pd.to_datetime(campaigns_df["start_date"], errors="coerce")
    campaigns_df = campaigns_df.dropna(subset=["start_date"])
    campaigns_df = campaigns_df[campaigns_df["audience_size"] >= 1]
    campaigns_df = campaigns_df.drop_duplicates(subset=["campaign_id"], keep="first")

    weekly_df = weekly.copy()
    required_weekly_cols = ["campaign_id", "week_number", "week_date",
                             "spend", "impressions", "clicks", "conversions"]
    weekly_df = weekly_df.dropna(subset=required_weekly_cols)
    weekly_df["week_date"] = pd.to_datetime(weekly_df["week_date"], errors="coerce")
    weekly_df = weekly_df.dropna(subset=["week_date"])

    if rules.get("drop_negative_metrics", True):
       
        for col in ["spend", "impressions", "clicks", "conversions"]:
            weekly_df = weekly_df[weekly_df[col] >= 0]

    weekly_df = weekly_df.drop_duplicates(subset=["campaign_id", "week_number"], keep="first")

   
    valid_campaign_ids = set(campaigns_df["campaign_id"])
    weekly_df = weekly_df[weekly_df["campaign_id"].isin(valid_campaign_ids)]

   
    week_counts = weekly_df.groupby("campaign_id")["week_number"].transform("count")
    weekly_df = weekly_df[week_counts >= rules["min_weeks_per_campaign"]]
    campaigns_df = campaigns_df[
        campaigns_df["campaign_id"].isin(set(weekly_df["campaign_id"]))
    ]

    campaigns_df = campaigns_df.sort_values("campaign_id").reset_index(drop=True)
    weekly_df = weekly_df.sort_values(["campaign_id", "week_number"]).reset_index(drop=True)

    return {"campaigns": campaigns_df, "weekly_performance": weekly_df}


if __name__ == "__main__":
   
    from config_loader import load_config
    from synthetic_data import generate_synthetic_campaigns

    cfg = load_config()
    raw = generate_synthetic_campaigns(cfg)
    cleaned = clean_campaigns(raw["campaigns"], raw["weekly_performance"], cfg)
    print(f"Campaigns: {len(raw['campaigns']):,} -> {len(cleaned['campaigns']):,}")
    print(f"Weekly rows: {len(raw['weekly_performance']):,} -> "
          f"{len(cleaned['weekly_performance']):,}")
