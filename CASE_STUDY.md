# Case study: Atlas — Marketing Intelligence Dashboard

## The problem

Marketing dashboards almost universally report last-click attribution: a campaign gets credit for every conversion that follows exposure, whether or not the ad actually changed anyone's behavior. This systematically favors channels that reach people who already intended to convert — branded search, retargeting — while undervaluing channels that reach genuinely new audiences. A CFO cutting the budget of a high-naive-ROAS retargeting campaign and reallocating it to a lower-naive-ROAS cold-audience channel is often making exactly the right call, and a naive dashboard would never tell them that.

## The approach

Atlas is built around a real experimental technique, simulated honestly:

1. **A simulated randomized holdout test** withholds a portion of a campaign's audience from ever seeing the ad. That group's conversion rate reveals what would have happened without the campaign — the true causal baseline. Comparing the exposed group's rate to it isolates the ad's actual incremental effect, separate from organic intent.
2. **Naive ROAS and true incremental ROAS are computed side by side** for every campaign that ran a holdout test (~30% of campaigns, realistically — most companies don't test everything). The gap between them is the project's central finding.
3. **A supervised regression model**, trained only on holdout-tested campaigns, learns the relationship between easily-observed features (spend, CTR, channel, naive ROAS) and true incremental ROAS — then estimates incremental impact for the ~70% of campaigns that never had a holdout test, which is the only economically realistic way to get impact estimates across an entire campaign portfolio.
4. **Unsupervised segmentation** groups campaigns into scale/sustain/cut tiers, plus an "audit" override that flags any campaign where naive and true metrics disagree sharply — regardless of how well it otherwise performs.

Every number in the final output is explicitly labeled measured (from a real test) or estimated (from the model) — never blended together silently, because a marketing lead needs to know which kind of number they're looking at before reallocating budget on it.

## Why synthetic data, and an honest tuning story

No client data exists for this demo. The first version of the generator produced technically-correct but implausible numbers — ROAS values in the hundreds, because channel conversion-rate assumptions were calibrated for realism of the underlying story (branded search converts a large share of its audience) without checking the resulting business metric. The rates were rescaled by roughly 40x, preserving the relative story (which channels look better naively than truly) while landing ROAS in a believable 2x-30x range. This is documented as a real development step in `concepts.md`, not smoothed over.

## Results (on the bundled 150-campaign demo sample)

- 4 campaign segments recovered with a silhouette score of 0.60
- 32% of campaigns have a real, holdout-measured incrementality label; the impact model estimates the rest
- The clearest naive-vs-true gap: `paid_search_brand` shows 16.7x naive ROAS but only 2.96x true incremental ROAS — a ~5.6x overstatement
- `paid_social` and `out_of_home`, which look mediocre naively, hold up much closer to their naive numbers under true measurement — exactly the channels a naive dashboard would undervalue

## What this demonstrates for client work

The same pipeline (`src/pipeline.py`) is built to run unmodified against real ad-platform exports (Google/Meta Ads reporting) combined with a client's own holdout-test results where available — swapping `DATA_MODE=local` and pointing the loader at real campaign and weekly-performance tables requires no changes to cleaning, feature engineering, segmentation, scoring, or recommendation logic. This is the fourth proof point that the same pipeline architecture generalizes — and the first to demonstrate genuine causal reasoning, not just pattern recognition, as part of that architecture.
