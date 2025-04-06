from time import sleep
from sqlmodel import Session, select
from database.model import Company, Quote, Transaction

from database.engine import engine
import yfinance as yf
import pandas as pd

def get_quotes():
    # Fetch all data upfront
    with Session(engine) as session:
        companies = session.exec(select(Company).where(Company.ticker != None)).all()
        for company in companies:
            print(f"Fetching data for {company.ticker}")
            try:
                # Get the latest quote date for this company
                latest_quote: Quote | None = session.exec(select(Quote).where(Quote.id_company == company.id_company).order_by(Quote.date_.desc())).first()
                first_transaction: Transaction | None = session.exec(select(Transaction).where(Transaction.id_company == company.id_company).order_by(Transaction.date_.asc())).first()
                
                start_date = latest_quote.date_ if latest_quote else first_transaction.date_ - pd.Timedelta(days=180) if first_transaction else pd.Timestamp.today() - pd.Timedelta(days=365)
                print(f"Fetching data for {company.ticker} from: {start_date}")
                
                stock = yf.Ticker(f"{company.ticker}.PA")
                hist = stock.history(start=start_date,
                                    end=pd.Timestamp.today())
                if not hist.empty:
                    for index, row in hist.iterrows():
                        # Check if a quote already exists for this company and date
                        existing_quote = session.exec(
                            select(Quote).where(
                                (Quote.id_company == company.id_company) & 
                                (Quote.date_ == index.date())
                            )
                        ).first()
                        
                        # Only add the quote if it doesn't already exist
                        if not existing_quote:
                            print(f"Adding quote for {company.ticker} on {index.date()}")
                            session.add(Quote(
                                id_company=company.id_company,
                                date_=index,
                                open=float(row["Open"]),
                                high=float(row["High"]),
                                low=float(row["Low"]),
                                close=float(row["Close"]),
                                volume=int(row["Volume"]),
                                dividends=float(row["Dividends"]),
                                stock_splits=float(row["Stock Splits"]),
                            ))
                            # Commit each quote to avoid conflicts
                            session.commit()
                        else:
                            print(f"Skipping duplicate quote for {company.ticker} on {index.date()}")
                else:
                    print(f"No price data found for {company.ticker}")
            except Exception as e:
                print(f"Error fetching {company.ticker}: {e}")
            sleep(1)

if __name__ == "__main__":
    get_quotes()