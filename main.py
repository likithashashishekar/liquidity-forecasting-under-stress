import os
import sys
import pandas as pd

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.predictor import LiquidityPredictor
from src.config import CONFIG

if __name__ == "__main__":
    # Load your dataset
    df = pd.read_csv("data/sample_lob_data.csv", parse_dates=["timestamp"])

    # Initialize and train model
    predictor = LiquidityPredictor(CONFIG)
    predictor.fit(df)

    # Make a liquidity forecast
    forecast = predictor.predict_liquidity_crunch(df)

    # Display results
    print("\n✅ Liquidity Forecast Results\n")
    print(f"Liquidity Risk Score: {forecast.risk_score:.2f}")
    print(f"Time to Crunch (min): {forecast.time_to_crunch_min:.2f}")
    print("Top Signal Drivers:")
    for k, v in forecast.top_signals.items():
        print(f"  {k}: {v:.4f}")
