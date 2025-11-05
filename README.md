# 💧 Liquidity Forecasting Under Stress Scenarios

> _"Liquidity disappears when you need it most — predict when it will vanish."_

This project builds a **machine learning pipeline** to forecast **liquidity dry-ups under market stress**.  
It’s designed with quantitative trading and risk management applications in mind.

---

## 🧠 Motivation

During crises or flash crashes, liquidity evaporates suddenly — making it impossible to unwind positions.  
This model detects **early warning signals** in market microstructure data.

---

## 🧩 Features

| Feature | Description |
|----------|-------------|
| `spread` | Ask - Bid difference |
| `order_book_imbalance` | Pressure between buyers/sellers |
| `market_depth_decay_rate` | Thinness of the order book |
| `volatility` | Rolling mid-price standard deviation |
| `dealer_inventory_changes` | Market-maker exposure shifts |
| `flash_crash_proximity_indicators` | Volatility × Spread composite risk metric |

---

## ⚙️ Model Pipeline

Data → Feature Engineering → Scaling → XGBoost Classifier → Risk Score Forecast

yaml
Copy code

---

## 🧮 Example Output

✅ Model trained successfully.

✅ Liquidity Forecast Results

Liquidity Risk Score: 0.74
Time to Crunch (min): 22.58
Top Signal Drivers:
market_depth_decay_rate: 0.30
volatility: 0.22
dealer_inventory_changes: 0.18
order_book_imbalance: 0.16
flash_crash_proximity_indicators: 0.14

css
Copy code

---

## 📈 Visualization

Example: Liquidity risk evolution over time
```python
import matplotlib.pyplot as plt

plt.plot(df["timestamp"], predictor.model.predict_proba(
    predictor.scaler.transform(predictor.fe.build_features(df)))[:, 1])
plt.title("Liquidity Risk Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Risk Score")
plt.show()
🛠️ Tech Stack
Python 3.13

Pandas / NumPy

Scikit-Learn

XGBoost

Matplotlib / Seaborn

Jupyter Notebook

🚀 Run Locally
bash
Copy code
git clone https://github.com/<your-username>/liquidity-forecasting-under-stress.git
cd liquidity-forecasting-under-stress
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python main.py
🏦 Why Jane Street Would Care
Predictive signals for liquidity disappearance

Robust risk-aware model during flash crashes

Quantitative edge in execution and market making

📧 Contact
Author: Likki Shashikala Somashekar
Roles: Quantitative Trader | Data Scientist | AI-ML Researcher
LinkedIn: [Add your profile link]
GitHub: https://github.com/<your-username>

yaml
Copy code

---

## ✅ 4️⃣ Push updated README

After saving `README.md`:

```powershell
git add README.md
git commit -m "Add professional README"
git push