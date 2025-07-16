# Stock Prediction with AI

A project to predict stock prices using insider trading data from the French financial markets. Basically trying to see if we can beat the market by following what company insiders are doing.

**Live demo**: https://s1ge.com (abandoned project)

## What it does

This project scrapes insider trading data from the **AMF BDIF** (French financial regulator's database) and uses AI to predict stock movements. The idea is that insiders know more than the public, so their trades can be predictive signals.

## The Data Pipeline

### 1. Data Collection (`worker/`)
- Fetches insider trading declarations from AMF's API
- Downloads PDF documents for each declaration
- Parses French-formatted documents (dates, amounts, etc.)
- Stores everything in PostgreSQL

### 2. Model (`model.ipynb` & `model.py`)

**Features:**
- **Insider Elo Rating**: Like chess ELO but for insiders - tracks how good each insider is at timing the market
- **Consensus Analysis**: Counts buyers vs sellers over time windows to see if insiders are bullish/bearish
- **Holdings Ratio**: How much of their position each insider is trading
- **Momentum**: Price trends using linear regression
- **Market Timing**: Performance relative to CAC40

**Model:**
- Uses LightGBM for the heavy lifting
- Hyperparameter optimization with Optuna
- Cross-validation and backtesting
- Portfolio construction based on model predictions

The notebook (`model.ipynb`) shows the full ML workflow - from data preprocessing to feature engineering to model training and evaluation. It's basically a complete ML pipeline for financial prediction.

### 3. Web App (`app/`)
- FastAPI backend to serve the strategies
- Shows performance charts, transaction details
- Calculates returns, drawdowns, Sharpe ratios

## The approach

The core idea is that insider trading data contains predictive signals that most people ignore. The ML model learns patterns like:

- Which insiders are consistently good at timing
- What transaction sizes indicate strong conviction
- How consensus among insiders affects stock performance
- Market timing patterns

The `model.ipynb` notebook shows the full ML pipeline - data cleaning, feature engineering, model training, hyperparameter tuning, and backtesting.

## Note

This was initially designed as a SaaS but ended up being more of a research project. The code shows a complete ML pipeline for financial predictions.
