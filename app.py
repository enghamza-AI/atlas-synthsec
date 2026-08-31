

from __future__ import annotations

import streamlit as st
import plotly.express as px

from src.config_loader import load_config, get_data_mode
from src.pipeline import run_pipeline

st.set_page_config(
    page_title="Atlas — Marketing Intelligence Dashboard",
    page_icon=":material/campaign:",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def cached_load_config() -> dict:
    return load_config()


@st.cache_data(show_spinner=False)
def cached_run_pipeline(data_mode: str) -> dict:
    config = cached_load_config()
    return run_pipeline(config, data_mode=data_mode)


with st.sidebar:
    st.markdown("### Atlas")
    st.caption("SynthSec marketing intelligence demo")

    config = cached_load_config()
    default_mode = get_data_mode(config)

    data_mode = st.radio(
        "Data mode",
        options=["demo", "local"],
        index=0 if default_mode == "demo" else 1,
        help=(
            "demo: pre-generated 150-campaign sample, fast on free tier. "
            "local: full synthetic generation, for development only."
        ),
    )

    st.divider()
    st.caption(
        "All data on this page is synthetic — generated to demonstrate "
        "the methodology, never real client data."
    )


st.title("Atlas — which campaigns create business impact?")
st.markdown(
    "A live demo of SynthSec's marketing intelligence methodology, built "
    "entirely on **synthetic** campaign performance and holdout-test data."
)

with st.expander("What is this, and what problem does it solve?", expanded=True):
    st.markdown(
        """
Most marketing dashboards report "conversions" from last-click
attribution — but that number counts everyone who converted after
seeing an ad, including people who would have bought anyway. A branded
search ad can show a spectacular ROAS while creating almost no real
incremental business, because it mostly reaches people already
searching for the brand by name.

Atlas answers the real question two ways:

1. **Tier** (unsupervised) — scale, sustain, cut, or audit, where
   "audit" specifically flags campaigns whose naive metrics and true
   measured impact disagree sharply.
2. **Impact score** (supervised, ROAS) — the true incremental ROAS
   where a randomized holdout test measured it directly, or a model
   estimate (trained on the campaigns that WERE tested) where one
   wasn't run. Every number is labeled measured or estimated — never
   blended together silently.
        """
    )

st.divider()

if "pipeline_output" not in st.session_state:
    st.session_state.pipeline_output = None
if "last_data_mode" not in st.session_state:
    st.session_state.last_data_mode = None

run_col, status_col = st.columns([1, 3])
with run_col:
    run_clicked = st.button("Run analysis", type="primary", use_container_width=True)

if run_clicked:
    with st.status("Running Atlas pipeline...", expanded=True) as status:
        st.write("Loading campaign and weekly performance data...")
        st.write("Cleaning and computing naive vs. incremental ROAS...")
        st.write("Segmenting campaigns (K-Means)...")
        st.write("Training incremental-impact model on holdout-tested campaigns...")
        output = cached_run_pipeline(data_mode)
        st.session_state.pipeline_output = output
        st.session_state.last_data_mode = data_mode
        status.update(label="Analysis complete", state="complete", expanded=False)

output = st.session_state.pipeline_output

if output is None:
    st.info("Click **Run analysis** to generate the dashboard.")
    st.stop()

if st.session_state.last_data_mode != data_mode:
    st.warning(
        f"Showing results for '{st.session_state.last_data_mode}' mode — "
        f"click Run analysis again to refresh for '{data_mode}' mode."
    )

result = output["result"]

st.divider()
st.subheader("Dashboard")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Campaigns analyzed", f"{output['n_campaigns']:,}")
m2.metric("Segments found", result["segment_name"].nunique())
m3.metric("Model MAE (ROAS pts)", output["model_mae"])
m4.metric("% with real holdout test", f"{output['pct_measured']:.0%}")

segment_counts = result["segment_name"].value_counts().reset_index()
segment_counts.columns = ["segment", "count"]
fig_segments = px.bar(
    segment_counts, x="segment", y="count", color="segment",
    title="Campaigns per segment",
)
fig_segments.update_layout(showlegend=False, height=360)
st.plotly_chart(fig_segments, use_container_width=True)
del segment_counts, fig_segments

st.subheader("Insights")

fig_scatter = px.scatter(
    result,
    x="naive_roas",
    y="impact_score",
    color="segment_name",
    symbol="has_incrementality_label",
    hover_data=["campaign_id", "channel", "total_spend"],
    title="Naive ROAS vs. true impact score, by segment "
          "(circle = estimated, diamond = measured)",
)
fig_scatter.add_shape(
    type="line", x0=0, y0=0,
    x1=result["naive_roas"].max(), y1=result["naive_roas"].max(),
    line=dict(dash="dot", color="gray"),
)
fig_scatter.update_layout(height=440)
st.plotly_chart(fig_scatter, use_container_width=True)
st.caption(
    "The dotted line is where naive ROAS = true impact. Campaigns far "
    "below it are largely capturing demand that would have existed "
    "anyway — the vanity-metrics gap this project exists to surface."
)
del fig_scatter

by_channel = result.groupby("channel").agg(
    naive_roas=("naive_roas", "mean"),
    impact_score=("impact_score", "mean"),
).reset_index()
fig_channel = px.bar(
    by_channel.melt(id_vars="channel", var_name="metric", value_name="roas"),
    x="channel", y="roas", color="metric", barmode="group",
    title="Naive vs. impact ROAS by channel",
)
fig_channel.update_layout(height=380)
st.plotly_chart(fig_channel, use_container_width=True)
del by_channel, fig_channel

st.subheader("Recommended actions")
st.caption("Sorted by urgency — start at the top.")

urgency_order = {"high": 0, "medium": 1, "low": 2}
display_df = result.copy()
display_df["_sort"] = display_df["urgency"].map(urgency_order)
display_df = display_df.sort_values(
    ["_sort", "impact_score"], ascending=[True, False]
)

st.dataframe(
    display_df[
        [
            "campaign_id",
            "channel",
            "segment_name",
            "urgency",
            "naive_roas",
            "impact_score",
            "recommended_action",
        ]
    ].head(50),
    use_container_width=True,
    hide_index=True,
)
del display_df

st.divider()
st.caption(
    "Atlas is a demo built on synthetic data. A client engagement runs "
    "the identical pipeline against real ad-platform/CRM exports — see "
    "about_the_project.md for how that swap works."
)
