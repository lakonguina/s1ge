from fastapi import Depends, FastAPI, Request
from sqlmodel import create_engine, Session, select
from database.model import StrategyReturn, Strategy, StrategyTransaction
from database.model import Strategy
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from pathlib import Path
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

static_dir = os.path.join(BASE_DIR, "static")
engine = create_engine("postgresql://dinitie:dinitie@postgres:5432/dinitie")

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
        .where(Strategy.id_strategy != 2)
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
        .where(StrategyReturn.id_strategy == 2)
    ).all()
    
    # Create serializable data for charts
    strategy_chart_data = []
    cumulative = 1.0
    for r in sorted(strategy_returns, key=lambda x: x.date_):
        cumulative *= (1 + r.return_/100)
        strategy_chart_data.append({
            "date_": str(r.date_),
            "cumulative_return": round((cumulative - 1) * 100, 2)
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
    
    return templates.TemplateResponse(
        "strategy.html",
        {
            "request": request,
            "strategy": strategy,
            "strategy_returns": strategy_chart_data,
            "cac40": cac40_chart_data
        }
    )
