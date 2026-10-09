# Directional-Forecaster

A desktop app that predicts whether a stock's price will go **up or down** in the next time step, using an LSTM neural network trained on that stock's own recent data.

## Features

- **Live stock data** — enter any ticker and pull its price history through yfinance.
- **Ticker autocomplete** — suggestions appear as you type.
- **Flexible timeframes** — choose a period (1 day to 3 months) and interval (1 to 90 minutes).
- **Custom technical indicators** — pick which features feed the model: returns, moving averages (5/10/20), volatility, RSI, MACD, and Bollinger Bands.
- **LSTM price-direction model** — a PyTorch neural network classifies the next step as up or down.
- **Automatic tuning** — Bayesian optimization finds the best hyperparameters (learning rate, hidden size, layers, indicator windows) for each run.
- **Train / validate / test split** — the model is validated and tested on unseen data before predicting.
- **Prediction with confidence** — outputs the predicted direction for the next time step and how confident the model is.
- **GPU support** — uses CUDA when available, falls back to CPU.

## How to Run

1. Install the requirements:

   ```bash
   pip install -r Directional-Forecaster/requirements.txt
   ```

2. From the repo root, run:

   ```bash
   python Directional-Forecaster/main.py
   ```

3. Enter a ticker, pick a time period and interval, select your features, and hit **Submit**.

## How It Works

1. Downloads recent price history for the chosen ticker.
2. Computes the technical indicators you selected.
3. Automatically tunes and trains the LSTM on that data.
4. Tests it on unseen data, then predicts the next step's direction.

*Built as a personal machine-learning project — not financial advice.*
