from fastapi import Depends, FastAPI, Request
from sqlmodel import create_engine, Session, select
from database.model import StrategyReturn, Strategy, StrategyTransaction
from database.model import Strategy
from fastapi.templating import Jinja2Templates
import os
from pathlib import Path

# Set up the correct templates directory path
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

engine = create_engine("postgresql://dinitie:dinitie@postgres:5432/dinitie")

def get_session():
    with Session(engine) as session:
        yield session


app = FastAPI()
templates = Jinja2Templates(directory=TEMPLATES_DIR)


@app.get("/")
def index(request: Request, session: Session = Depends(get_session)):
    strategy_returns = session.exec(
        select(StrategyReturn)
        .join(Strategy)
        .where(Strategy.name == "Insider Purchases")
    ).all()
    cac40_returns = session.exec(
        select(StrategyReturn)
        .join(Strategy)
        .where(Strategy.name == "CAC 40")
    ).all()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "insider_purchases": strategy_returns,
            "cac40": cac40_returns
        }
    )