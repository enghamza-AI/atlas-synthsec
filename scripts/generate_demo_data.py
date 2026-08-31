

from __future__ import annotations

from src.config_loader import load_config
from src.synthetic_data import generate_synthetic_campaigns


def main() -> None:
    config = load_config()
    campaigns_path = config["app"]["demo_csv_path"]
    weekly_path = campaigns_path.replace("demo_sample.csv", "demo_sample_weekly.csv")

    generated = generate_synthetic_campaigns(config)
    generated["campaigns"].to_csv(campaigns_path, index=False)
    generated["weekly_performance"].to_csv(weekly_path, index=False)

    print(f"Wrote {len(generated['campaigns']):,} campaigns to {campaigns_path}")
    print(f"Wrote {len(generated['weekly_performance']):,} weekly rows to {weekly_path}")


if __name__ == "__main__":
    main()
