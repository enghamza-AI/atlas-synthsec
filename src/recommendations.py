
from __future__ import annotations

import pandas as pd


def generate_recommendations(scored: pd.DataFrame, config: dict) -> pd.DataFrame:
 
    rec_cfg = config["recommendations"]
    scale_floor = rec_cfg["scale_incremental_roas_floor"]

    out = scored.copy()

    actions = []
    urgencies = []

    for _, row in out.iterrows():
        segment = row["segment_name"]
        impact = row["impact_score"]
        naive = row["naive_roas"]
       
        confidence = "measured" if row["has_incrementality_label"] else "estimated"

        if segment == "audit":
            action = (
                f"{confidence.capitalize()} impact score is only "
                f"{impact:.1f}x ROAS despite a naive {naive:.1f}x — this "
                f"campaign is largely reaching people who'd have "
                f"converted anyway. Cut budget and reallocate rather "
                f"than judging it by the naive number."
            )
            urgency = "high"

        elif segment == "scale":
            action = (
                f"{confidence.capitalize()} incremental ROAS is "
                f"{impact:.1f}x — genuinely driving new business, not "
                f"just capturing existing demand. Increase budget here."
            )
            urgency = "high"

        elif segment == "cut":
            action = (
                f"{confidence.capitalize()} incremental ROAS is only "
                f"{impact:.1f}x. Not a vanity-metrics problem — this "
                f"campaign simply isn't working. Cut or fundamentally "
                f"rework it."
            )
            urgency = "medium"

        else:  # sustain
            action = (
                f"{confidence.capitalize()} incremental ROAS is "
                f"{impact:.1f}x, above the {scale_floor:.1f}x floor for "
                f"scaling consideration but not a standout. Keep "
                f"running at current budget."
            )
            urgency = "low"

        if confidence == "estimated":
            action += " (No holdout test has run on this campaign — this is a model estimate, not a measured result.)"

        actions.append(action)
        urgencies.append(urgency)

    out["recommended_action"] = actions
    out["urgency"] = urgencies
    return out


if __name__ == "__main__":
   
    from config_loader import load_config
    from synthetic_data import generate_synthetic_campaigns
    from cleaning import clean_campaigns
    from feature_engineering import build_campaign_features
    from segmentation import segment_campaigns
    from scoring import train_impact_model, predict_impact

    cfg = load_config()
    raw = generate_synthetic_campaigns(cfg)
    cleaned = clean_campaigns(raw["campaigns"], raw["weekly_performance"], cfg)
    features = build_campaign_features(cleaned["campaigns"], cleaned["weekly_performance"], cfg)
    segmented = segment_campaigns(features, cfg)
    bundle = train_impact_model(segmented, cfg)
    scored = predict_impact(segmented, bundle)
    recs = generate_recommendations(scored, cfg)
    print(recs["urgency"].value_counts())
    print(recs[["campaign_id", "segment_name", "urgency",
                 "recommended_action"]].head(3).to_string())
