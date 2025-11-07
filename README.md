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

