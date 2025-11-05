import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Compute liquidity early-warning signals from market microstructure.
    """

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        required_cols = [
            "bid_price1", "ask_price1", "bid_size1", "ask_size1",
            "depth_level1", "depth_level5", "dealer_inventory"
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        feats = pd.DataFrame(index=df.index)

        feats["spread"] = df["ask_price1"] - df["bid_price1"]
        feats["mid_price"] = (df["ask_price1"] + df["bid_price1"]) / 2
        feats["order_book_imbalance"] = (
            (df["bid_size1"] - df["ask_size1"]) /
            (df["bid_size1"] + df["ask_size1"] + 1e-6)
        )
        feats["market_depth_decay_rate"] = df["depth_level1"] / (df["depth_level5"] + 1e-6)
        feats["volatility"] = feats["mid_price"].pct_change().rolling(10).std().fillna(0)
        feats["dealer_inventory_changes"] = df["dealer_inventory"].diff().fillna(0)
        feats["flash_crash_proximity_indicators"] = feats["volatility"] * feats["spread"]

        return feats.fillna(0)
