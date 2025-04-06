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

def get_cac40_quotes():
    with Session(engine) as session:
        # Check if CAC40 company exists
        cac40 = session.exec(select(Company).where(Company.ticker == "^FCHI")).first()
        
        # Create CAC40 company if it doesn't exist
        if not cac40:
            print("Creating CAC40 index in database")
            # Find the maximum ID currently in the database and increment by 1
            max_id_result = session.exec(select(Company.id_company).order_by(Company.id_company.desc())).first()
            next_id = (max_id_result or 0) + 1
            print(f"Using ID {next_id} for CAC40 company")
            
            cac40 = Company(
                id_company=next_id,
                name="CAC 40",
                ticker="^FCHI"
            )
            session.add(cac40)
            session.commit()
            # Refresh to get the id_company
            session.refresh(cac40)
        
        print(f"Fetching data for {cac40.ticker}")
        try:
            # Get the latest quote date for CAC40
            latest_quote: Quote | None = session.exec(select(Quote).where(Quote.id_company == cac40.id_company).order_by(Quote.date_.desc())).first()
            
            # For indices, we'll use a default start date if no quotes exist
            start_date = latest_quote.date_ if latest_quote else pd.Timestamp("2017-01-01")
            print(f"Fetching data for {cac40.ticker} from: {start_date}")
            
            # For CAC40, we don't append .PA 
            stock = yf.Ticker(cac40.ticker)
            hist = stock.history(start=start_date,
                                end=pd.Timestamp.today())
            if not hist.empty:
                for index, row in hist.iterrows():
                    # Check if a quote already exists for this date
                    existing_quote = session.exec(
                        select(Quote).where(
                            (Quote.id_company == cac40.id_company) & 
                            (Quote.date_ == index.date())
                        )
                    ).first()
                    
                    # Only add the quote if it doesn't already exist
                    if not existing_quote:
                        print(f"Adding quote for {cac40.ticker} on {index.date()}")
                        session.add(Quote(
                            id_company=cac40.id_company,
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
                        print(f"Skipping duplicate quote for {cac40.ticker} on {index.date()}")
            else:
                print(f"No price data found for {cac40.ticker}")
        except Exception as e:
            print(f"Error fetching {cac40.ticker}: {e}")

if __name__ == "__main__":
    #get_quotes()
    #get_cac40_quotes()