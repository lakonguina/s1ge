from fastapi import Depends, FastAPI, Request
from sqlmodel import create_engine, Session, select
from database.model import StrategyReturn, Strategy, StrategyTransaction
from database.model import Strategy
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from pathlib import Path


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

@app.get("/strategies")
def strategies(request: Request, session: Session = Depends(get_session)):
    strategies = session.exec(
        select(Strategy)
        .where(Strategy.id_strategy != 2)
    ).all()
    return templates.TemplateResponse(
        "strategies.html",
        {"request": request, "strategies": strategies}
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
    return templates.TemplateResponse(
        "strategy.html",
        {
            "request": request,
            "strategy": strategy,
            "strategy_returns": strategy_returns,
            "cac40": cac40_returns
        }
    )
