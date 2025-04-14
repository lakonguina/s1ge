from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import UniqueConstraint
from datetime import date
from enum import Enum


class Company(SQLModel, table=True):
    __tablename__ = "companies"
    id_company: int | None = Field(default=None, primary_key=True)
    name: str | None = Field(default=None)
    isin: str | None = Field(default=None)
    ticker: str | None = Field(default=None)

    quotes: list["Quote"] = Relationship(back_populates="company")
    persons: list["Person"] = Relationship(back_populates="companies")
    transactions: list["Transaction"] = Relationship(back_populates="company")


class Declaration(SQLModel, table=True):
    __tablename__ = "declarations"
    id_declaration: int | None = Field(default=None, primary_key=True)
    id_person: int | None = Field(default=None, foreign_key="persons.id_person")
    id_document: str = Field(default=None)
    date_: date | None = Field(default=None)

    person: "Person" = Relationship(back_populates="declarations")
    transactions: list["Transaction"] = Relationship(back_populates="declaration")


class Person(SQLModel, table=True):
    __tablename__ = "persons"
    id_person: int | None = Field(default=None, primary_key=True)
    id_company: int = Field(foreign_key="companies.id_company")
    name: str = Field(default=None)

    companies: "Company" = Relationship(back_populates="persons")
    declarations: list["Declaration"] = Relationship(back_populates="person")
    transactions: list["Transaction"] = Relationship(back_populates="person")


class TransactionNature(SQLModel, table=True):
    __tablename__ = "transaction_natures"
    id_transaction_nature: int | None = Field(default=None, primary_key=True)
    name: str = Field(default=None)

    transactions: list["Transaction"] = Relationship(back_populates="transaction_nature")


class TransactionPlace(SQLModel, table=True):
    __tablename__ = "transaction_places"
    id_transaction_place: int | None = Field(default=None, primary_key=True)
    name: str  = Field(default=None)

    transactions: list["Transaction"] = Relationship(back_populates="transaction_place")


class TransactionInstrument(SQLModel, table=True):
    __tablename__ = "transaction_instruments"
    id_transaction_instrument: int | None = Field(default=None, primary_key=True)
    name: str = Field(default=None)

    transactions: list["Transaction"] = Relationship(back_populates="transaction_instrument")


class Currency(SQLModel, table=True):
    __tablename__ = "currencies"
    id_currency: int | None = Field(default=None, primary_key=True)
    name: str = Field(default=None)
    symbol: str | None = Field(default=None)

    transactions: list["Transaction"] = Relationship(back_populates="currency")


class Transaction(SQLModel, table=True):
    __tablename__ = "transactions"
    id_transaction: int | None = Field(default=None, primary_key=True)
    id_company: int = Field(foreign_key="companies.id_company")
    id_declaration: int = Field(foreign_key="declarations.id_declaration")
    id_person: int = Field(foreign_key="persons.id_person")
    id_currency: int = Field(foreign_key="currencies.id_currency")
    id_transaction_nature: int = Field(foreign_key="transaction_natures.id_transaction_nature")
    id_transaction_place: int = Field(foreign_key="transaction_places.id_transaction_place")
    id_transaction_instrument: int = Field(foreign_key="transaction_instruments.id_transaction_instrument") 
    date_: date | None = Field(default=None)
    volume: float = Field(default=None)
    price: float = Field(default=None)
    is_exercice_on_option_or_free_attribution_of_stocks: bool = Field(default=False)

    company: Company = Relationship(back_populates="transactions")
    person: Person = Relationship(back_populates="transactions")
    currency: Currency = Relationship(back_populates="transactions")
    transaction_nature: TransactionNature = Relationship(back_populates="transactions")
    transaction_place: TransactionPlace = Relationship(back_populates="transactions")
    transaction_instrument: TransactionInstrument = Relationship(back_populates="transactions")
    declaration: Declaration = Relationship(back_populates="transactions")
    


class Quote(SQLModel, table=True):
    __tablename__ = "quotes"
    id_quote: int | None = Field(default=None, primary_key=True)
    id_company: int = Field(foreign_key="companies.id_company")
    date_: date | None = Field(default=None)
    open: float | None = Field(default=None)
    high: float | None = Field(default=None)
    low: float | None = Field(default=None)
    close: float | None = Field(default=None)
    volume: float | None = Field(default=None)
    dividends: float | None = Field(default=None)
    stock_splits: float | None = Field(default=None)
    
    __table_args__ = (
        UniqueConstraint('date_', 'id_company', name='unique_company_date'),
    )

    company: Company = Relationship(back_populates="quotes")


class Strategy(SQLModel, table=True):
    __tablename__ = "strategies"
    id_strategy: int | None = Field(default=None, primary_key=True)
    name: str = Field(default=None)

    strategy_transactions: list["StrategyTransaction"] = Relationship(back_populates="strategy")
    strategy_returns: list["StrategyReturn"] = Relationship(back_populates="strategy")


class StrategyTransactionNature(Enum):
    BUY = "BUY"
    SELL = "SELL"


class StrategyTransaction(SQLModel, table=True):
    __tablename__ = "strategy_transactions"
    id_strategy_transaction: int | None = Field(default=None, primary_key=True)
    id_strategy: int = Field(foreign_key="strategies.id_strategy")
    nature: StrategyTransactionNature | None = Field(default=None)
    date_: date | None = Field(default=None)
    conviction_score: float | None = Field(default=None)

    strategy: Strategy = Relationship(back_populates="strategy_transactions")


class StrategyReturn(SQLModel, table=True):
    __tablename__ = "strategy_returns"
    id_strategy_return: int | None = Field(default=None, primary_key=True)
    id_strategy: int = Field(foreign_key="strategies.id_strategy")
    date_: date | None = Field(default=None)
    return_: float | None = Field(default=None)
    cumulative_return: float | None = Field(default=None)

    strategy: Strategy = Relationship(back_populates="strategy_returns")
