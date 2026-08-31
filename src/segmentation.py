
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def segment_campaigns(features: pd.DataFrame, config: dict) -> pd.DataFrame:

    seg_cfg = config["segmentation"]
    feature_cols = seg_cfg["clustering_features"]

    X = _prepare_clustering_features(features, feature_cols)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=seg_cfg["n_clusters"],
        random_state=seg_cfg["random_state"],
        n_init=seg_cfg["n_init"],
    )
    cluster_ids = kmeans.fit_predict(X_scaled)

    sample_size = min(2000, len(X_scaled))
    sample_idx = np.random.default_rng(seg_cfg["random_state"]).choice(
        len(X_scaled), size=sample_size, replace=False
    )
    sil_score = silhouette_score(X_scaled[sample_idx], cluster_ids[sample_idx])

    out = features.copy()
    out["cluster_id"] = cluster_ids
    out["silhouette_avg"] = round(float(sil_score), 3)
    out["segment_name"] = _name_segments(out, config)

    return out


def _prepare_clustering_features(features: pd.DataFrame, feature_cols: list) -> np.ndarray:

    df = features.copy()

    labeled = df[df["has_incrementality_label"]]
    if len(labeled) > 0 and labeled["naive_roas"].sum() > 0:
       
        avg_ratio = (
            labeled["estimated_incremental_roas"].sum() / labeled["naive_roas"].sum()
        )
    else:
        avg_ratio = 1.0

    df["estimated_incremental_roas"] = df["estimated_incremental_roas"].fillna(
        df["naive_roas"] * avg_ratio
    )

    return df[feature_cols].to_numpy()


def _name_segments(df: pd.DataFrame, config: dict) -> pd.Series:
   
    rec_cfg = config["recommendations"]
    gap_mult = rec_cfg["vanity_gap_multiplier"]

   
    incremental_for_ranking = df["estimated_incremental_roas"].fillna(df["naive_roas"])

    centroid_stats = df.assign(_incr=incremental_for_ranking).groupby("cluster_id").agg(
        mean_incremental=("_incr", "mean"),
        mean_naive=("naive_roas", "mean"),
    )

    
    remaining = list(centroid_stats.index)
    label_by_cluster = {}

    scale_id = centroid_stats.loc[remaining, "mean_incremental"].idxmax()
    label_by_cluster[scale_id] = "scale"
    remaining.remove(scale_id)

    cut_id = centroid_stats.loc[remaining, "mean_incremental"].idxmin()
    label_by_cluster[cut_id] = "cut"
    remaining.remove(cut_id)

    for cid in remaining:
        label_by_cluster[cid] = "sustain"

    names = df["cluster_id"].map(label_by_cluster)


    has_large_gap = (
        df["has_incrementality_label"]
        & (df["naive_roas"] >= df["estimated_incremental_roas"] * gap_mult)
    )
    names = names.where(~has_large_gap, "audit")

    return names


if __name__ == "__main__":
   
    from config_loader import load_config
    from synthetic_data import generate_synthetic_campaigns
    from cleaning import clean_campaigns
    from feature_engineering import build_campaign_features

    cfg = load_config()
    raw = generate_synthetic_campaigns(cfg)
    cleaned = clean_campaigns(raw["campaigns"], raw["weekly_performance"], cfg)
    features = build_campaign_features(cleaned["campaigns"], cleaned["weekly_performance"], cfg)
    segmented = segment_campaigns(features, cfg)
    print(segmented["segment_name"].value_counts())
    print(f"\nSilhouette score: {segmented['silhouette_avg'].iloc[0]}")
