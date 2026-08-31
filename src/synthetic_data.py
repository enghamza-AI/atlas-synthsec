

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_campaigns(config: dict) -> dict:

    sd = config["synthetic_data"]
    rng = np.random.default_rng(sd["random_seed"])

    n_campaigns = sd["n_campaigns"]
    channels = sd["channels"]
    channel_names = list(channels.keys())

    sim_end_date = pd.Timestamp.today().normalize()

    campaign_rows = []
    weekly_rows = []

    for campaign_id in range(1, n_campaigns + 1):
        channel = channel_names[rng.integers(0, len(channel_names))]
        ch = channels[channel]

        duration_weeks = int(
            rng.integers(sd["duration_weeks_min"], sd["duration_weeks_max"] + 1)
        )
        start_offset_weeks = int(rng.integers(0, 52))
        start_date = sim_end_date - pd.Timedelta(weeks=start_offset_weeks)

        audience_size = max(
            500,
            int(round(rng.lognormal(
                mean=sd["audience_size_lognormal_mean"],
                sigma=sd["audience_size_lognormal_sigma"],
            ))),
        )

   
        organic_propensity = float(np.clip(
            ch["organic_propensity_base"]
            + rng.normal(0, ch["organic_propensity_base"] * 0.2),
            ch["organic_propensity_base"] * 0.2,
            0.9,
        ))
        true_lift = max(
            0.0,
            ch["true_lift_base"] + rng.normal(0, ch["true_lift_base"] * 0.25),
        )

        has_holdout = bool(rng.random() < sd["holdout_test_fraction"])

        if has_holdout:
            holdout_size = int(audience_size * sd["holdout_size_fraction"])
            exposed_size = audience_size - holdout_size
           
            holdout_conversions = rng.binomial(holdout_size, organic_propensity)
            measured_organic_rate = (
                holdout_conversions / holdout_size if holdout_size > 0 else np.nan
            )
        else:
            holdout_size = 0
            exposed_size = audience_size
            measured_organic_rate = np.nan

     
        exposed_conversion_rate = min(0.95, organic_propensity + true_lift)
        total_exposed_conversions = int(
            rng.binomial(exposed_size, exposed_conversion_rate)
        )

        total_impressions = max(
            1, int(audience_size * sd["impressions_per_person"] * rng.lognormal(0, 0.15))
        )
        total_spend = round(total_impressions / 1000.0 * ch["cpm"], 2)
        total_clicks = int(
            round(total_impressions * max(0.0, ch["ctr_base"] + rng.normal(0, 0.005)))
        )

      
        weekly_weights = rng.dirichlet(np.ones(duration_weeks) * 2.0)

        weekly_impressions = _split_integer(total_impressions, weekly_weights, rng)
        weekly_clicks = _split_integer(total_clicks, weekly_weights, rng)
        weekly_conversions = _split_integer(total_exposed_conversions, weekly_weights, rng)

        for w in range(duration_weeks):
            week_date = start_date + pd.Timedelta(weeks=w)
            week_spend = round(weekly_impressions[w] / 1000.0 * ch["cpm"], 2)
            weekly_rows.append(
                {
                    "campaign_id": campaign_id,
                    "week_number": w,
                    "week_date": week_date,
                    "spend": week_spend,
                    "impressions": weekly_impressions[w],
                    "clicks": weekly_clicks[w],
                    "conversions": weekly_conversions[w],
                }
            )

        campaign_rows.append(
            {
                "campaign_id": campaign_id,
                "channel": channel,
                "start_date": start_date,
                "duration_weeks": duration_weeks,
                "audience_size": audience_size,
                "has_holdout_test": has_holdout,
                "holdout_size": holdout_size,
                "measured_organic_rate": measured_organic_rate,
            }
        )

    campaigns = pd.DataFrame(campaign_rows)
    weekly_performance = pd.DataFrame(weekly_rows).sort_values(
        ["campaign_id", "week_number"]
    ).reset_index(drop=True)

    return {"campaigns": campaigns, "weekly_performance": weekly_performance}


def _split_integer(total: int, weights: np.ndarray, rng: np.random.Generator) -> list:
 
    raw = total * weights
    floor_parts = np.floor(raw).astype(int)
    remainder = int(total - floor_parts.sum())
    if remainder > 0:
     
        fractional = raw - floor_parts
        probs = fractional / fractional.sum() if fractional.sum() > 0 else None
        chosen = rng.choice(len(weights), size=remainder, replace=False, p=probs) \
            if probs is not None else rng.choice(len(weights), size=remainder, replace=False)
        for idx in np.atleast_1d(chosen):
            floor_parts[idx] += 1
    return floor_parts.tolist()


if __name__ == "__main__":
   
    from config_loader import load_config

    cfg = load_config()
    data = generate_synthetic_campaigns(cfg)
    campaigns, weekly = data["campaigns"], data["weekly_performance"]
    print(f"Generated {len(campaigns):,} campaigns, {len(weekly):,} weekly rows")
    print(f"Campaigns with a holdout test: {campaigns['has_holdout_test'].sum():,} "
          f"({campaigns['has_holdout_test'].mean():.1%})")
    print(campaigns.head())
    print(weekly.head())
