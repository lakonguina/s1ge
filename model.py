# %%
%load_ext autoreload
%autoreload 2

# %%
from sqlmodel import Session, create_engine, select
from datetime import date, timedelta
import pandas as pd
import numpy as np
from scipy import stats
from sqlmodel import SQLModel
from database.model import Company, TransactionNature, TransactionInstrument, Person, Transaction, Currency
import yfinance as yf

# Configuration de la base de données PostgreSQL
engine = create_engine("postgresql://dinitie:dinitie@localhost:5432/dinitie")

# Extraction des transactions 2020-2024
with Session(engine) as session:
    query_transactions = (
        select(Transaction, Company, Person, TransactionNature, TransactionInstrument, Currency)
        .join(Company, Transaction.id_company == Company.id_company)
        .join(Person, Transaction.id_person == Person.id_person)
        .join(TransactionNature, Transaction.id_transaction_nature == TransactionNature.id_transaction_nature)
        .join(TransactionInstrument, Transaction.id_transaction_instrument == TransactionInstrument.id_transaction_instrument)
        .join(Currency, Transaction.id_currency == Currency.id_currency)
        .where(Transaction.date_ != None)
        .where(Company.isin != None)
        .where(Company.ticker != None)
        .where(Currency.name == "Euro")
    )
    results = session.exec(query_transactions).all()

# Conversion en DataFrame avec inspection des dates
transactions = []
invalid_dates = []

for t, co, p, n, i, cu in results:
    raw_date = t.date_  # Inspectez la date brute
    try:
        # Convertir la date avec errors='coerce' pour gérer les cas problématiques
        date_converted = pd.to_datetime(raw_date, errors='coerce')
        
        # Vérifiez si la conversion a échoué (NaT)
        if pd.isna(date_converted):
            invalid_dates.append((t.id_transaction, raw_date, "Conversion échouée (NaT)"))
            continue
        
        # Vérifiez si l'année est dans une plage raisonnable
        if date_converted.year < 2000 or date_converted > pd.Timestamp(date.today()):
            invalid_dates.append((t.id_transaction, raw_date, f"Année hors plage: {date_converted.year}"))
            continue
        # Ajoutez la transaction valide
        transactions.append({
            "id_company": t.id_company,
            "ticker": co.ticker,
            "date": date_converted,
            "volume": t.volume,
            "price": t.price,
            "nature": n.name,
            "id_person": p.id_person,
            "person": p.name,
        })
    except Exception as e:
        # Capturez toutes les erreurs inattendues
        invalid_dates.append((t.id_transaction, raw_date, f"Erreur: {str(e)}"))
        continue

# Créez le DataFrame
transactions = pd.DataFrame(transactions)

# Affichez les informations de débogage
print(f"Nombre de transactions valides: {len(transactions)}")
if invalid_dates:
    print("\nDates problématiques trouvées:")
    for tx_id, raw_date, error in invalid_dates:
        print(f"Transaction {tx_id}: Date brute = {raw_date}, Erreur = {error}")

# %%
transactions

# %%
from database.model import Quote


quotes = session.exec(select(Quote, Company.ticker).join(Company, Quote.id_company == Company.id_company)).all()

# %%
# Replace the original price_data construction block with this
price_data = {}
for q, ticker in quotes:
    if ticker not in price_data:
        price_data[ticker] = []
    price_data[ticker].append({"Date": q.date_, "Close": q.close, "Volume": q.volume})

for ticker in price_data:
    # Convert to DataFrame
    df = pd.DataFrame(price_data[ticker])
    # Ensure Date is datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    # Remove duplicates, keeping the last entry for each date
    df = df.drop_duplicates(subset="Date", keep="last")
    # Localize timezone if not already done
    if df["Date"].dt.tz is None:
        df["Date"] = df["Date"].dt.tz_localize("Europe/Paris", ambiguous="NaT", nonexistent="shift_forward")
    # Set Date as index
    df = df.set_index("Date")
    # Verify the index is unique
    if not df.index.is_unique:
        raise ValueError(f"Duplicate dates remain in price_data for ticker {ticker} after deduplication")
    price_data[ticker] = df
    print(f"Timezone localized and deduplicated for ticker {ticker}, rows: {len(df)}")

# %%
def process_features(txs, price_data, fundamentals):
    txs["trade_value"] = np.log(txs["volume"] * txs["price"] + 1)
    
    # Fix NaNs in nature
    conditions = [
        txs["nature"].str.lower().str.contains("acquisition", na=False),
        txs["nature"].str.lower().str.contains("cession", na=False) & ~txs["nature"].str.lower().str.contains("succession", na=False)
    ]
    choices = ["acquisition", "cession"]
    txs["nature"] = np.select(conditions, choices, default="unknown")
    txs_clean = txs[txs["nature"].isin(["acquisition", "cession"])].copy().reset_index(drop=True)
    
    # Consensus
    txs_clean["consensus"] = 0
    for ticker in txs_clean["ticker"].unique():
        ticker_data = txs_clean[txs_clean["ticker"] == ticker].copy()
        for idx, row in ticker_data.iterrows():
            current_date = row["date"]
            lookback_start = current_date - pd.Timedelta(days=90)
            lookback_data = ticker_data[(ticker_data["date"] >= lookback_start) & 
                                      (ticker_data["date"] <= current_date)]
            buyers = lookback_data[lookback_data["nature"] == "acquisition"]["id_person"].nunique()
            sellers = lookback_data[lookback_data["nature"] == "cession"]["id_person"].nunique()
            txs_clean.at[idx, "consensus"] = buyers - sellers
    
    # Insider Elo (custom, timezone fix)
    txs_clean["insider_elo"] = 1500
    for id_person in txs_clean["id_person"].unique():
        person_trades = txs_clean[txs_clean["id_person"] == id_person].sort_values("date").copy()
        elo = 1500
        for idx, trade in person_trades.iterrows():
            ticker = trade["ticker"]
            trans_date = pd.Timestamp(trade["date"])
            direction = 1 if trade["nature"] == "acquisition" else -1
            if ticker in price_data:
                prices = price_data[ticker]["Close"]
                # Ensure prices.index is timezone-aware
                if prices.index.tz is None:
                    prices = prices.copy()
                    prices.index = prices.index.tz_localize("Europe/Paris")
                trans_date_tz = trans_date.tz_localize("Europe/Paris")
                current_idx = prices.index.get_indexer([trans_date_tz], method="nearest")[0]
                if current_idx != -1:
                    future_date_tz = trans_date_tz + pd.Timedelta(days=30)  # Keeps timezone
                    future_idx = prices.index.get_indexer([future_date_tz], method="nearest")[0]
                    if future_idx != -1 and future_idx < len(prices):
                        ret = (prices.iloc[future_idx] - prices.iloc[current_idx]) / prices.iloc[current_idx]
                        expected = 1 / (1 + 10 ** (-elo / 400))
                        actual = 1 if direction * ret > 0 else 0
                        elo += 32 * (actual - expected)
                        txs_clean.at[idx, "insider_elo"] = elo
    
    # Holdings Ratio
    txs_clean["holdings_ratio"] = 0.0
    for id_person in txs_clean["id_person"].unique():
        person_trades = txs_clean[txs_clean["id_person"] == id_person].copy()
        for ticker in person_trades["ticker"].unique():
            ticker_trades = person_trades[person_trades["ticker"] == ticker].sort_values("date")
            total_shares = 0
            for idx, row in ticker_trades.iterrows():
                change = row["volume"] if row["nature"] == "acquisition" else -row["volume"]
                total_shares_before = total_shares
                total_shares += change
                if total_shares_before > 0:
                    txs_clean.at[idx, "holdings_ratio"] = change / total_shares_before
    
    txs_clean["momentum"] = 0.0
    for idx, row in txs_clean.iterrows():
        ticker = row["ticker"]
        trans_date = pd.Timestamp(row["date"])
        if ticker in price_data:
            prices = price_data[ticker]["Close"]
            cac40_prices = price_data["^FCHI"]["Close"]
            if prices.index.tz is None:
                prices = prices.copy()
                prices.index = prices.index.tz_localize("Europe/Paris")
            if cac40_prices.index.tz is None:
                cac40_prices = cac40_prices.copy()
                cac40_prices.index = cac40_prices.index.tz_localize("Europe/Paris")
            trans_date_tz = trans_date.tz_localize("Europe/Paris")
            lookback_start = trans_date_tz - pd.Timedelta(days=180)  # 90 days
            stock_window = prices[(prices.index >= lookback_start) & (prices.index <= trans_date_tz)]
            if len(stock_window) > 10:
                x = np.arange(len(stock_window))
                slope, _, _, _, std_err = stats.linregress(x, stock_window)
                t_stat = slope / std_err if std_err != 0 else 0
                volatility = stock_window.pct_change().std() if stock_window.pct_change().std() != 0 else 1
                stock_momentum = (t_stat / volatility) * 100  # Scaled to 100
                cac40_window = cac40_prices[(cac40_prices.index >= lookback_start) & (cac40_prices.index <= trans_date_tz)]
                if len(cac40_window) > 10:
                    x_cac = np.arange(len(cac40_window))
                    slope_cac, _, _, _, std_err_cac = stats.linregress(x_cac, cac40_window)
                    t_stat_cac = slope_cac / std_err_cac if std_err_cac != 0 else 0
                    volatility_cac = cac40_window.pct_change().std() if cac40_window.pct_change().std() != 0 else 1
                    cac40_momentum = (t_stat_cac / volatility_cac) * 100  # Scaled to 100
                    txs_clean.at[idx, "momentum"] = stock_momentum - cac40_momentum

    """
    # Price-To-Book Ratio
    txs_clean["pb_ratio"] = np.nan
    for ticker in txs_clean["ticker"].unique():
        if ticker in fundamentals:
            txs_clean.loc[txs_clean["ticker"] == ticker, "pb_ratio"] = fundamentals[ticker]["priceToBook"]
    txs_clean["pb_bin"] = pd.qcut(txs_clean["pb_ratio"], q=5, labels=False, duplicates="drop", retbins=False)
    
    # Company Size
    txs_clean["company_size"] = np.nan
    for ticker in txs_clean["ticker"].unique():
        if ticker in fundamentals and "^FCHI" in price_data:
            market_cap = fundamentals[ticker]["marketCap"]
            if market_cap:
                cac40_level = price_data["^FCHI"]["Close"].iloc[-1]
                adjusted_size = market_cap / (cac40_level * 1e9)
                txs_clean.loc[txs_clean["ticker"] == ticker, "company_size"] = adjusted_size
    """
    
    # Excess return
    txs_clean["excess_return"] = 0.0
    for idx, row in txs_clean.iterrows():
        ticker = row["ticker"]
        trans_date = pd.Timestamp(row["date"])
        if ticker in price_data:
            prices = price_data[ticker]["Close"]
            cac40_prices = price_data["^FCHI"]["Close"]
            if prices.index.tz is None:
                prices = prices.copy()
                prices.index = prices.index.tz_localize("Europe/Paris")
            if cac40_prices.index.tz is None:
                cac40_prices = cac40_prices.copy()
                cac40_prices.index = cac40_prices.index.tz_localize("Europe/Paris")
            trans_date_tz = trans_date.tz_localize("Europe/Paris")
            current_idx = prices.index.get_indexer([trans_date_tz], method="nearest")[0]
            cac40_current_idx = cac40_prices.index.get_indexer([trans_date_tz], method="nearest")[0]
            if current_idx != -1 and cac40_current_idx != -1:
                future_date_tz = trans_date_tz + pd.Timedelta(days=30)
                future_idx = prices.index.get_indexer([future_date_tz], method="nearest")[0]
                cac40_future_idx = cac40_prices.index.get_indexer([future_date_tz], method="nearest")[0]
                if future_idx != -1 and future_idx < len(prices) and cac40_future_idx != -1:
                    stock_ret = (prices.iloc[future_idx] - prices.iloc[current_idx]) / prices.iloc[current_idx]
                    cac40_ret = (cac40_prices.iloc[cac40_future_idx] - cac40_prices.iloc[cac40_current_idx]) / cac40_prices.iloc[cac40_current_idx]
                    txs_clean.at[idx, "excess_return"] = stock_ret - cac40_ret
    
    
    txs_clean["significant"] = (txs_clean["excess_return"] > 0).astype(int)
    
    return txs_clean

# %%
# Re-run preprocessing with the updated function
processed_transactions = process_features(transactions, price_data, None)

# %%
features = ["trade_value", "consensus", "insider_elo", "holdings_ratio", "momentum"]

# Check for NaNs and clean
print("NaN counts in features:")
print(processed_transactions[features].isna().sum())
processed_transactions_clean = processed_transactions.dropna(subset=features)
print(f"Rows after removing NaN: {len(processed_transactions_clean)} (original: {len(processed_transactions)})")  

# %%
def evaluate_portfolio(portfolio, price_data):
    returns = []
    print("\nEvaluating portfolio weeks:")
    for idx, row in portfolio.iterrows():
        start_date = pd.Timestamp(row["date"]).tz_localize("Europe/Paris")
        end_date = start_date + pd.Timedelta(days=7)
        week_ret = 0
        if row["tickers"]:
            print(f"Week {idx}: {start_date} - Tickers: {row['tickers']}")
            for ticker, weight in zip(row["tickers"], row["weights"]):
                if ticker in price_data:
                    prices = price_data[ticker]["Close"]
                    start_idx = prices.index.get_indexer([start_date], method="nearest")[0]
                    end_idx = prices.index.get_indexer([end_date], method="nearest")[0]
                    if start_idx != -1 and end_idx != -1 and end_idx < len(prices):
                        ret = (prices.iloc[end_idx] - prices.iloc[start_idx]) / prices.iloc[start_idx]
                        week_ret += ret * weight
                    else:
                        print(f"  No valid price data for {ticker} between {start_date} and {end_date}")
            returns.append({"date": start_date, "return": week_ret})
        else:
            print(f"Week {idx}: {start_date} - No tickers")
            returns.append({"date": start_date, "return": 0.0})  # Default to 0 if no tickers
    portfolio_returns = pd.DataFrame(returns)
    print(f"Portfolio returns shape: {portfolio_returns.shape}")
    return portfolio_returns

# %%

def implement_strategy(data, start_date, end_date, rank_min=9, score_threshold=0.3):
    portfolio = []
    current_date = start_date
    
    while current_date <= end_date:
        lookback_start = current_date - pd.Timedelta(days=90)
        recent_trades = data[(data["date"] >= lookback_start) & 
                            (data["date"] <= current_date) & 
                            (data["conviction_rank"] >= rank_min) & 
                            (data["nature"] == "acquisition")].copy()
        
        if len(recent_trades) > 0:
            recent_trades["score"] = recent_trades["conviction"]
            weekly_portfolio = recent_trades.groupby("ticker").agg({"score": "mean"}).reset_index()
            weekly_portfolio["week"] = current_date
            portfolio.append(weekly_portfolio)
        
        current_date += pd.Timedelta(days=1)
    
    if len(portfolio) == 0:
        return [], 0.0
    
    portfolio_df = pd.concat(portfolio)
    portfolio_df = portfolio_df.groupby("ticker").agg({"score": "mean"}).reset_index()
    portfolio_df = portfolio_df[portfolio_df["score"] >= score_threshold]
    if len(portfolio_df) == 0:
        return [], 0.0
    
    # Localize timestamps for price lookup
    start_date_tz = pd.Timestamp(start_date).tz_localize("Europe/Paris")
    end_date_tz = pd.Timestamp(end_date).tz_localize("Europe/Paris")
    
    week_return = 0.0
    for ticker in portfolio_df["ticker"]:
        if ticker in price_data:
            prices = price_data[ticker]["Close"]
            start_idx = prices.index.get_indexer([start_date_tz], method="nearest")[0]
            end_idx = prices.index.get_indexer([end_date_tz], method="nearest")[0]
            if start_idx != -1 and end_idx != -1 and end_idx < len(prices):
                ret = (prices.iloc[end_idx] - prices.iloc[start_idx]) / prices.iloc[start_idx]
                week_return += ret / len(portfolio_df)
    
    return portfolio_df, week_return

# %%
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
import optuna
from sklearn.model_selection import train_test_split

def run_backtest_with_daily_results(processed_data, price_data, start_date="2020-03-01", end_date="2025-12-31", 
                                    features=None, rank_min=9, score_threshold=0.3, window_years=3):
    if features is None:
        features = ["trade_value", "consensus", "insider_elo", "holdings_ratio", 
                    "momentum"]
    
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    
    # Align to Sunday
    current_date = start_date + pd.Timedelta(days=(6 - start_date.weekday()) % 7)  # Next Sunday
    if current_date < start_date:
        current_date += pd.Timedelta(days=7)
    
    daily_dates = []
    portfolio_daily_returns = []
    cac_daily_returns = []
    weekly_dates = []
    portfolio_weekly_returns = []
    cac_weekly_returns = []
    weekly_holdings = []
    
    earliest_date = processed_data["date"].min()
    print(f"Earliest date in data: {earliest_date}")
    
    def optimize_lgbm(trial, X_train, y_train, X_val, y_val):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50)
        }
        model = LGBMClassifier(**params, random_state=42, verbosity=-1)
        model.fit(X_train, y_train)
        return model.score(X_val, y_val)
    
    while current_date <= end_date:
        window_start = current_date - pd.Timedelta(days=365.25 * window_years)
        train_data = processed_data[(processed_data["date"] >= window_start) & 
                                   (processed_data["date"] < current_date)].copy()
        test_data = processed_data[(processed_data["date"] >= current_date) & 
                                 (processed_data["date"] < current_date + pd.Timedelta(days=7))].copy()
        
        if len(train_data) > 0 and len(test_data) > 0:
            print(f"Week starting {current_date}: Train {window_start} to {current_date}, {len(train_data)} rows, Test {len(test_data)} rows")
            X_train = train_data[features]
            y_train = (train_data["excess_return"] > 0).astype(int)
            X_test = test_data[features]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Split training data for validation in Optuna
            X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
                X_train_scaled, y_train, test_size=0.2, random_state=42
            )
            
            # Optimize LightGBM hyperparameters
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: optimize_lgbm(trial, X_train_opt, y_train_opt, X_val_opt, y_val_opt), 
                         n_trials=20)
            best_params = study.best_params
            
            # Train LightGBM with optimized parameters
            lgbm_model = LGBMClassifier(**best_params, random_state=42, verbosity=-1)
            lgbm_model.fit(X_train_scaled, y_train)
            lgbm_conviction = lgbm_model.predict_proba(X_test_scaled)[:, 1]
            
            # Train RandomForest (unchanged)
            rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            rf_model.fit(X_train_scaled, y_train)
            rf_conviction = rf_model.predict_proba(X_test_scaled)[:, 1]
            
            # Combine predictions
            test_data["conviction"] = (lgbm_conviction + rf_conviction) / 2
            test_data["conviction_rank"] = 1
            high_conviction = test_data["conviction"] > 0.55
            if high_conviction.sum() > 0:
                test_data.loc[high_conviction, "conviction_rank"] = pd.qcut(
                    test_data.loc[high_conviction, "conviction"], 10, labels=False, duplicates="drop"
                ) + 1
            
            week_start = current_date
            week_end = min(current_date + pd.Timedelta(days=6), end_date)
            portfolio, week_return = implement_strategy(test_data, week_start, week_end, 
                                                       rank_min=rank_min, score_threshold=score_threshold)
            weekly_holdings.append({
                "date": week_start,
                "holdings": portfolio
            })
            
            week_start_tz = pd.Timestamp(week_start).tz_localize("Europe/Paris")
            week_end_tz = pd.Timestamp(week_end).tz_localize("Europe/Paris")
            
            # Calculate CAC 40 weekly return
            cac_prices = price_data["^FCHI"]["Close"]
            cac_start_idx = cac_prices.index.get_indexer([week_start_tz], method="nearest")[0]
            cac_end_idx = cac_prices.index.get_indexer([week_end_tz], method="nearest")[0]
            if cac_start_idx != -1 and cac_end_idx != -1 and cac_end_idx < len(cac_prices):
                cac_week_return = (cac_prices.iloc[cac_end_idx] - cac_prices.iloc[cac_start_idx]) / cac_prices.iloc[cac_start_idx]
            else:
                cac_week_return = 0.0
            
            # Store weekly returns
            weekly_dates.append(week_start)
            portfolio_weekly_returns.append(week_return)
            cac_weekly_returns.append(cac_week_return)
            
            # Distribute weekly returns evenly across days
            daily_range = pd.date_range(start=week_start_tz, end=week_end_tz, freq="D")
            num_days = len(daily_range)
            if num_days > 0:
                daily_portfolio_return = (1 + week_return) ** (1 / num_days) - 1
                daily_cac_return = (1 + cac_week_return) ** (1 / num_days) - 1
                
                for day in daily_range:
                    daily_dates.append(day)
                    portfolio_daily_returns.append(daily_portfolio_return)
                    cac_daily_returns.append(daily_cac_return)
                    print(f"Day {day.date()}: Strategy = {daily_portfolio_return:.4f}, CAC = {daily_cac_return:.4f}")
            else:
                daily_dates.append(week_start_tz)
                portfolio_daily_returns.append(0.0)
                cac_daily_returns.append(0.0)
            
            print(f"Week {week_start}: Strategy = {week_return:.4f}, CAC = {cac_week_return:.4f}")
        
        current_date += pd.Timedelta(days=7)
    
    # Create DataFrame with daily returns
    results_df = pd.DataFrame({
        "Date": daily_dates,
        "Strategy_Return": portfolio_daily_returns,
        "CAC40_Return": cac_daily_returns
    }).set_index("Date")
    
    # Calculate cumulative returns (for plotting)
    results_df["Strategy_Cumulative"] = (1 + results_df["Strategy_Return"]).cumprod() - 1
    results_df["CAC40_Cumulative"] = (1 + results_df["CAC40_Return"]).cumprod() - 1
    
    # Use weekly returns for yearly performance
    portfolio_weekly_series = pd.Series(portfolio_weekly_returns, index=weekly_dates)
    cac_weekly_series = pd.Series(cac_weekly_returns, index=weekly_dates)
    
    for year in range(2020, 2026):
        year_portfolio = portfolio_weekly_series[portfolio_weekly_series.index.year == year]
        year_cac = cac_weekly_series[cac_weekly_series.index.year == year]
        if not year_portfolio.empty:
            strategy_return = (1 + year_portfolio).prod() - 1
            cac_return = (1 + year_cac).prod() - 1
            print(f"{year}: Strategy = {strategy_return:.4f}, CAC 40 = {cac_return:.4f}, "
                  f"Outperformance = {strategy_return - cac_return:.4f}")
    
    return results_df, weekly_holdings
# Run the backtest
start_date = "2020-01-01"
end_date = "2025-03-21"
results_df, weekly_holdings = run_backtest_with_daily_results(processed_transactions_clean, price_data, 
                                            start_date=start_date, end_date=end_date, 
                                            rank_min=9, score_threshold=0.3, window_years=3)

# %%
for year, group in results_df.groupby(results_df.index.year):
    if not group.empty:
        strategy_return = (1 + group["Strategy_Return"]).prod() - 1
        cac_return = (1 + group["CAC40_Return"]).prod() - 1
        print(f"{year}: Strategy = {strategy_return:.4f}, CAC 40 = {cac_return:.4f}, "
              f"Outperformance = {strategy_return - cac_return:.4f}")
    else:
        print(f"{year}: No daily data available in results_df")

# %%
results_df

# %%
weekly_holdings

# %%
# Optional: Plot the results
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(results_df.index, results_df["Strategy_Cumulative"], label="Strategy Cumulative Return", color="blue")
plt.plot(results_df.index, results_df["CAC40_Cumulative"], label="CAC 40 Cumulative Return", color="orange")
plt.title(f"Strategy vs CAC 40 Cumulative Returns ({start_date} to {end_date})")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
from datetime import date
from pandas import Timestamp, DataFrame

def transform_holdings_to_transactions(holdings_list, price_data):
    """
    Transform holdings list into transaction dictionaries with BUY and SELL, including ROI percentage.
    
    Args:
        holdings_list: List of dicts with 'date' (Timestamp) and 'holdings' (DataFrame or empty list).
        price_data: Dict of DataFrames with ticker as key, containing 'Close' prices and DatetimeIndex.
    
    Returns:
        List of dicts with keys: id_strategy_transaction, nature, date_, conviction_score, company, roi.
    """
    transactions = []
    previous_tickers = set()
    buy_prices = {}  # Dict to store {ticker: (date, price)} for the last BUY

    for i, entry in enumerate(holdings_list):
        entry_date = entry["date"]
        holdings = entry["holdings"]
        
        # Convert Timestamp to date
        transaction_date = entry_date.date() if isinstance(entry_date, Timestamp) else entry_date
        transaction_date_tz = pd.Timestamp(entry_date).tz_localize("Europe/Paris") if isinstance(entry_date, Timestamp) else pd.Timestamp(entry_date).tz_localize("Europe/Paris")
        
        # Current tickers and scores
        current_tickers = set()
        current_scores = {}
        if isinstance(holdings, DataFrame) and not holdings.empty:
            current_tickers = set(holdings["ticker"])
            current_scores = dict(zip(holdings["ticker"], holdings["score"]))
        
        # SELL: Tickers in previous week but not current
        if i > 0:
            for ticker in previous_tickers - current_tickers:
                roi = None
                if ticker in buy_prices and ticker in price_data:
                    buy_date, buy_price = buy_prices[ticker]
                    prices = price_data[ticker]["Close"]
                    if prices.index.tz is None:
                        prices = prices.copy()
                        prices.index = prices.index.tz_localize("Europe/Paris")
                    sell_idx = prices.index.get_indexer([transaction_date_tz], method="nearest")[0]
                    if sell_idx != -1 and sell_idx < len(prices):
                        sell_price = prices.iloc[sell_idx]
                        if buy_price is not None and buy_price != 0:
                            roi = ((sell_price - buy_price) / buy_price) * 100
                transactions.append({
                    "id_strategy_transaction": None,
                    "nature": "SELL",
                    "date_": transaction_date,
                    "conviction_score": None,
                    "company": ticker,
                    "return_": roi
                })
                # Remove ticker from buy_prices after selling
                if ticker in buy_prices:
                    del buy_prices[ticker]
        
        # BUY: Tickers in current week but not previous
        for ticker in current_tickers - previous_tickers:
            buy_price = None
            if ticker in price_data:
                prices = price_data[ticker]["Close"]
                if prices.index.tz is None:
                    prices = prices.copy()
                    prices.index = prices.index.tz_localize("Europe/Paris")
                buy_idx = prices.index.get_indexer([transaction_date_tz], method="nearest")[0]
                if buy_idx != -1 and buy_idx < len(prices):
                    buy_price = prices.iloc[buy_idx]
                buy_prices[ticker] = (transaction_date_tz, buy_price)
            transactions.append({
                "id_strategy_transaction": None,
                "nature": "BUY",
                "date_": transaction_date,
                "conviction_score": float(current_scores[ticker]) if ticker in current_scores else None,
                "company": ticker,
                "return_": None
            })
        
        # Update previous tickers
        previous_tickers = current_tickers
    
    return transactions

# %%
def get_or_create(session: Session, model: SQLModel, query_keys: list, **kwargs) -> SQLModel:
    """
    Retrieve an existing instance of the model or create a new one.

    Parameters:
    - session: The SQLModel session to use.
    - model: The SQLModel class to query.
    - query_keys: A list of keys from kwargs to use as the query condition.
                   Example: ['id'] to use kwargs['id'] for the query.
    - **kwargs: Additional keyword arguments to create a new instance if not found.

    Returns:
    - SQLModel: The found or newly created instance.
    """
    # Build query conditions properly using SQLModel's syntax
    query = select(model)
    for key in query_keys:
        if key in kwargs:
            # Get the column attribute from the model class
            column = getattr(model, key)
            # Add the filter condition
            query = query.where(column == kwargs[key])
    # Execute the query
    instance = session.exec(query).first()
    if instance:
        return instance
    else:
        instance = model(**kwargs)
        session.add(instance)
        session.flush()
        return instance

# %%

transactions = transform_holdings_to_transactions(
    weekly_holdings,
    price_data
)

# %%
from database.model import StrategyReturn, StrategyTransaction, Strategy

with Session(engine) as session:
    insider_purchases_strategy = get_or_create(session, Strategy, query_keys=["name"], name="Achat d'initiés")
    cac_40_strategy = get_or_create(session, Strategy, query_keys=["name"], name="CAC 40")

    existing_returns = session.exec(
        select(StrategyReturn).where(StrategyReturn.id_strategy == insider_purchases_strategy.id_strategy and
        StrategyReturn.id_strategy == cac_40_strategy.id_strategy)
    )
    existing_transactions = session.exec(
        select(StrategyTransaction).where(StrategyTransaction.id_strategy == insider_purchases_strategy.id_strategy)
    )
    for r in existing_returns:
        session.delete(r)
    for tx in existing_transactions:
        session.delete(tx)
    
    for idx, row in results_df.iterrows():
        strategy_return = StrategyReturn(
            strategy=insider_purchases_strategy,
            date_=idx.date(),
            return_=row["Strategy_Return"],
            cumulative_return=row["Strategy_Cumulative"]
        ) 
        cac_strategy_return = StrategyReturn(
            strategy=cac_40_strategy,
            date_=idx.date(),
            return_=row["CAC40_Return"],
            cumulative_return=row["CAC40_Cumulative"]
        )
        session.add(strategy_return)
        session.add(cac_strategy_return)
    
    for tx in transactions:
        company = get_or_create(session, Company, query_keys=["name"], name=tx["company"])
        session.add(StrategyTransaction(
            id_strategy=insider_purchases_strategy.id_strategy,
            nature=tx["nature"],
            date_=tx["date_"],
            conviction_score=tx["conviction_score"],
            id_company=company.id_company,
            return_=tx["return_"] if tx["return_"] is not None else None
        ))
    session.commit()


