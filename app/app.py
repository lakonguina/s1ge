from fastapi import Depends, FastAPI, Request
from sqlmodel import create_engine, Session, select
from database.model import StrategyReturn, Strategy, StrategyTransaction, StrategyTransactionNature
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from babel.dates import format_date
import os
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

def format_date_fr(date_obj):
    """
    Format a date object to a French date string.
    Handles various date formats including strings, dictionaries, and datetime objects.
    """
    if date_obj is None:
        return "Date inconnue"
        
    # Handle dictionary format (from strategy_returns)
    if isinstance(date_obj, dict) and 'date_' in date_obj:
        date_obj = date_obj['date_']
    
    # If it's a string in ISO format (YYYY-MM-DD)
    if isinstance(date_obj, str):
        try:
            date_obj = datetime.strptime(date_obj, '%Y-%m-%d')
        except ValueError:
            try:
                # Try another common format
                date_obj = datetime.strptime(date_obj, '%d/%m/%Y')
            except ValueError:
                # Return the string as is if we can't parse it
                return date_obj
    
    # Use babel format for consistent localization
    try:
        return format_date(date_obj, locale='fr')
    except Exception as e:
        # Fallback if babel formatting fails
        try:
            return date_obj.strftime("%d %B %Y").lower()
        except:
            return str(date_obj)


BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

static_dir = os.path.join(BASE_DIR, "static")
engine = create_engine(os.getenv("POSTGRES_URL"))

def get_session():
    with Session(engine) as session:
        yield session


app = FastAPI()
templates = Jinja2Templates(directory=TEMPLATES_DIR)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
        }
    )

def calculate_annual_return(returns):
    if not returns:
        return 0.0
    
    sorted_returns = sorted(returns, key=lambda x: getattr(x, 'date_'))
    return_values = [getattr(r, 'return_', 0) for r in sorted_returns]
    
    if not return_values or all(v == 0 for v in return_values):
        return 0.0
    
    # Calculate geometric mean for annualized return
    decimal_returns = [1 + r/100 for r in return_values]
    product = 1.0
    for ret in decimal_returns:
        product *= ret
    
    # Get time period
    start_date = sorted_returns[0].date_
    end_date = sorted_returns[-1].date_
    days = (end_date - start_date).days
    years = days / 365.25 if days > 0 else 1
    
    # Annualized return formula
    annualized = (product ** (1/years)) - 1
    return round(annualized * 100, 2)

def calculate_max_drawdown(returns):
    if not returns:
        return 0.0
    
    sorted_returns = sorted(returns, key=lambda x: getattr(x, 'date_'))
    return_values = [getattr(r, 'return_', 0) for r in sorted_returns]
    
    if not return_values or all(v == 0 for v in return_values):
        return 0.0
    
    # Calculate cumulative returns
    cumulative = [1.0]
    for r in return_values:
        cumulative.append(cumulative[-1] * (1 + r/100))
    
    max_dd = 0.0
    peak = cumulative[0]
    
    for value in cumulative:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        max_dd = max(max_dd, drawdown)
    
    return round(max_dd * 100, 2)

def calculate_total_return(returns):
    if not returns:
        return 0.0
    
    sorted_returns = sorted(returns, key=lambda x: getattr(x, 'date_'))
    return_values = [getattr(r, 'return_', 0) for r in sorted_returns]
    
    if not return_values or all(v == 0 for v in return_values):
        return 0.0
    
    # Calculate compounded total return
    total = 1.0
    for r in return_values:
        total *= (1 + r/100)
    
    return round((total - 1) * 100, 2)

@app.get("/strategies")
def strategies(request: Request, session: Session = Depends(get_session)):
    strategies = session.exec(
        select(Strategy)
        .where(Strategy.slug != "cac-40")
    ).all()
    
    strategy_data = []
    for strategy in strategies:
        strategy_returns = session.exec(
            select(StrategyReturn)
            .where(StrategyReturn.id_strategy == strategy.id_strategy)
        ).all()
        
        annual_return = calculate_annual_return(strategy_returns)
        max_drawdown = calculate_max_drawdown(strategy_returns)
        total_return = calculate_total_return(strategy_returns)
        
        # Calculate cumulative returns for charting
        sorted_returns = sorted(strategy_returns, key=lambda x: x.date_)
        
        # If there are no returns, add a sample point to avoid an empty chart
        if not sorted_returns:
            # Add a default data point to show at least something on the chart
            chart_data = [{
                "date_": str(datetime.now().date()),
                "cumulative_return": 0.0
            }]
        else:
            # Calculate and create chart data as before
            chart_data = []
            cumulative = 1.0
            for r in sorted_returns:
                cumulative *= (1 + r.return_/100)
                chart_data.append({
                    "date_": str(r.date_),
                    "cumulative_return": round((cumulative - 1) * 100, 2)
                })
        
        # Add to strategy data dictionary - no CAC40 data included
        strategy_data.append({
            "strategy": strategy,
            "annual_return": annual_return,
            "max_drawdown": max_drawdown,
            "total_return": total_return,
            "returns": chart_data  # Use our manually created serializable data
        })
    
    return templates.TemplateResponse(
        "strategies.html",
        {"request": request, "strategy_data": strategy_data}
    )

def match_transactions(transactions):
    """
    Match BUY transactions with corresponding SELL transactions to create complete trades.
    Also identify open positions (buys without sells).
    """
    if not transactions:
        return [], []
    
    # Sort transactions by date
    sorted_transactions = sorted(transactions, key=lambda x: x.date_ if x.date_ else datetime.min.date())
    
    # Separate buys and sells
    buys = [tx for tx in sorted_transactions if tx.nature == StrategyTransactionNature.BUY]
    sells = [tx for tx in sorted_transactions if tx.nature == StrategyTransactionNature.SELL]
    
    # Use a simpler matching logic - just pair buys and sells sequentially
    completed_trades = []
    open_positions = []
    
    # If we have both buys and sells, create pairs
    i = 0
    while i < len(buys) and i < len(sells):
        buy = buys[i]
        sell = sells[i]
        
        # Only match if sell date is after buy date
        if buy.date_ and sell.date_ and sell.date_ > buy.date_:
            days_held = (sell.date_ - buy.date_).days
            
            # Get ticker from company if available (handle different possible object structures)
            ticker = None
            if hasattr(buy, 'company') and buy.company:
                ticker = getattr(buy.company, 'ticker', None) or getattr(buy.company, 'name', None)
            elif hasattr(buy, 'security') and buy.security:
                ticker = getattr(buy.security, 'ticker', None) or getattr(buy.security, 'name', None)
            
            if not ticker:
                ticker = f"TX-{buy.id_strategy_transaction}"
            
            completed_trades.append({
                "buy_transaction": buy,
                "sell_transaction": sell,
                "entry_date": buy.date_,
                "exit_date": sell.date_,
                "days_held": days_held,
                "conviction_score": buy.conviction_score,
                "return_value": sell.return_,
                "ticker": ticker,
                "is_open": False
            })
        i += 1
    
    # Remaining buys are open positions
    for j in range(i, len(buys)):
        buy = buys[j]
        days_held = (datetime.now().date() - buy.date_).days if buy.date_ else 0
        
        # Get ticker from company if available (handle different possible object structures)
        ticker = None
        if hasattr(buy, 'company') and buy.company:
            ticker = getattr(buy.company, 'ticker', None) or getattr(buy.company, 'name', None)
        elif hasattr(buy, 'security') and buy.security:
            ticker = getattr(buy.security, 'ticker', None) or getattr(buy.security, 'name', None)
        
        if not ticker:
            ticker = f"TX-{buy.id_strategy_transaction}"
        
        open_positions.append({
            "transaction": buy,
            "entry_date": buy.date_,
            "days_held": days_held,
            "conviction_score": buy.conviction_score,
            "ticker": ticker,
            "is_open": True
        })
    
    return completed_trades, open_positions

def create_weekly_view(strategy, transactions, returns):
    """
    Create a complete weekly view of the strategy performance,
    including weeks without transactions.
    
    Parameters:
    - strategy: The strategy object
    - transactions: List of strategy transactions
    - returns: List of strategy returns
    
    Returns a list of week objects, each containing:
    - week_start: Start date (Sunday)
    - week_end: End date (Saturday)
    - transactions: List of transactions in that week (may be empty)
    - cumulative_return: The cumulative return as of the end of that week
    """
    if not returns:
        return []
    
    # Find start and end dates of the entire period
    try:
        # Try to get the first return date as start
        first_return = min(r.date_ for r in returns if r.date_)
        # Get either the last return date or today as the end
        last_return = max(r.date_ for r in returns if r.date_)
        end_date = max(last_return, datetime.now().date())
    except (ValueError, AttributeError):
        # If returns are empty or dates are missing, use reasonable defaults
        end_date = datetime.now().date()
        first_return = end_date - timedelta(days=90)  # Default to 90 days
    
    # Adjust start_date to the previous Sunday
    days_since_sunday = (first_return.weekday() + 1) % 7
    start_date = first_return - timedelta(days=days_since_sunday)
    
    # Adjust end_date to the next Saturday
    days_to_saturday = (6 - end_date.weekday()) % 7
    end_date = end_date + timedelta(days=days_to_saturday)
    
    # Group transactions by week
    tx_by_date = {}
    for tx in transactions:
        if not tx.date_:
            continue
        tx_by_date.setdefault(tx.date_, []).append(tx)
    
    # Calculate cumulative returns by date
    cumulative_by_date = {}
    cumulative = 1.0
    sorted_returns = sorted(returns, key=lambda x: x.date_)
    for r in sorted_returns:
        if not r.date_:
            continue
        cumulative *= (1 + r.return_/100)
        cumulative_by_date[r.date_] = round((cumulative - 1) * 100, 2)
    
    # Generate all weeks in the period
    weeks = []
    current_week_start = start_date
    
    while current_week_start <= end_date:
        current_week_end = current_week_start + timedelta(days=6)
        
        # Find all transactions in this week
        week_transactions = []
        current_date = current_week_start
        while current_date <= current_week_end:
            if current_date in tx_by_date:
                week_transactions.extend(tx_by_date[current_date])
            current_date += timedelta(days=1)
        
        # Find the latest cumulative return for this week
        week_return = None
        current_date = current_week_end
        while current_date >= current_week_start:
            if current_date in cumulative_by_date:
                week_return = cumulative_by_date[current_date]
                break
            current_date -= timedelta(days=1)
        
        # Create the week entry
        weeks.append({
            "week_start": current_week_start,
            "week_end": current_week_end,
            "transactions": week_transactions,
            "has_transactions": len(week_transactions) > 0,
            "return_value": week_return
        })
        
        # Move to next week
        current_week_start += timedelta(days=7)
    
    return weeks

def get_current_week_range():
    """
    Obtient la semaine actuelle (dimanche-samedi).
    Retourne un tuple (date_début, date_fin).
    """
    today = datetime.now().date()
    
    # Calculer le début de la semaine (dimanche)
    days_since_sunday = (today.weekday() + 1) % 7
    week_start = today - timedelta(days=days_since_sunday)
    
    # Calculer la fin de la semaine (samedi)
    week_end = week_start + timedelta(days=6)
    
    return (week_start, week_end)

@app.get("/strategies/{slug}")
def strategy(slug: str, request: Request, session: Session = Depends(get_session)):
    strategy = session.exec(
        select(Strategy)
        .where(Strategy.slug == slug)
    ).first()
    
    strategy_returns = session.exec(
        select(StrategyReturn)
        .where(StrategyReturn.id_strategy == strategy.id_strategy)
    ).all()
    
    cac40_returns = session.exec(
        select(StrategyReturn)
        .join(Strategy)
        .where(Strategy.slug == "cac-40")
    ).all()
    
    # Create serializable data for charts
    strategy_chart_data = []
    
    # Sort returns by date
    sorted_returns = sorted(strategy_returns, key=lambda x: x.date_) if strategy_returns else []
    
    """
    if not sorted_returns:
        # Add a default point if no data is available
        today = datetime.now().date()
        strategy_chart_data.append({
            "date_": str(today),
            "cumulative_return": 0.0
        })
    else:
        cumulative = 1.0
        for r in sorted_returns:
            cumulative *= (1 + r.return_/100)
            strategy_chart_data.append({
                "date_": str(r.date_),
                "cumulative_return": round((cumulative - 1) * 100, 2)
            })
    """
    for r in sorted_returns:
        strategy_chart_data.append({
            "date_": str(r.date_),
            "cumulative_return": r.cumulative_return
        })
    
    # Calculate CAC40 cumulative returns
    cac40_chart_data = []
    cumulative = 1.0
    for r in sorted(cac40_returns, key=lambda x: x.date_):
        cumulative *= (1 + r.return_/100)
        cac40_chart_data.append({
            "date_": str(r.date_),
            "cumulative_return": round((cumulative - 1) * 100, 2)
        })
    
    # Get strategy transactions with company information
    strategy_transactions = session.exec(
        select(StrategyTransaction)
        .where(StrategyTransaction.id_strategy == strategy.id_strategy)
    ).all()
    
    # Match buy/sell transactions to create complete trades
    completed_trades, open_positions = match_transactions(strategy_transactions)
    
    # Obtenir la semaine actuelle
    current_week_range = get_current_week_range()
    
    # Make the format_date_fr function available in templates
    templates.env.globals["format_date_fr"] = format_date_fr

    return templates.TemplateResponse(
        "strategy.html",
        {
            "request": request,
            "strategy": strategy,
            "strategy_returns": strategy_chart_data,
            "cac40": cac40_chart_data,
            "strategy_transactions": strategy_transactions,
            "completed_trades": completed_trades,
            "open_positions": open_positions,
            "current_week_range": current_week_range
        }
    )
