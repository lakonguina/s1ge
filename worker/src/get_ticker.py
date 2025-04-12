import requests
from bs4 import BeautifulSoup
import re
from sqlmodel import Session, select
from database.model import Company
from database.engine import engine
import time

def get_ticker_from_isin(isin):
    """
    Get the ticker symbol for a given ISIN code by scraping Boursorama.
    
    Args:
        isin (str): The ISIN code (e.g., 'FR0010220475')
        
    Returns:
        str: The ticker symbol (e.g., 'ALO') or None if not found
    """
    # First request to the ISIN URL
    url = f"https://www.boursorama.com/cours/{isin}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, allow_redirects=True)
        
        # Check if we were redirected to a ticker URL
        if "1rP" in response.url:
            # Extract ticker from the URL
            ticker_match = re.search(r'1rP([A-Z0-9]+)', response.url)
            if ticker_match:
                return ticker_match.group(1)
        
        # If not redirected, try to find ticker in the page content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for the ticker in various places on the page
        # 1. Try to find it in the title
        title = soup.find('title')
        if title:
            ticker_match = re.search(r'\(([A-Z0-9]+)\)', title.text)
            if ticker_match:
                return ticker_match.group(1)
        
        # 2. Look for it in specific elements that might contain the ticker
        ticker_elements = soup.select('.c-faceplate__isin')
        for element in ticker_elements:
            text = element.text.strip()
            if text:
                # Split the text by spaces and take the last part which should be the ticker
                parts = text.split()
                if len(parts) > 1:  # Make sure we have both ISIN and ticker
                    return parts[-1]  # Return the last part (ticker)
                
        return None
    
    except Exception as e:
        print(f"Error fetching ticker for ISIN {isin}: {e}")
        return None

def update_company_tickers():
    """
    Update the ticker field for all companies in the database that have an ISIN code.
    """
    with Session(engine) as session:
        # Get all companies with ISIN but without ticker
        companies = session.exec(
            select(Company).where(
                Company.isin.is_not(None),
                (Company.ticker.is_(None) | (Company.ticker == ""))
            )
        ).all()
        
        print(f"Found {len(companies)} companies without tickers")
        
        for company in companies:
            print(f"Processing {company.name} with ISIN {company.isin}")
            ticker = get_ticker_from_isin(company.isin)
            
            if ticker:
                print(f"Found ticker {ticker} for {company.name}")
                company.ticker = ticker
                session.add(company)
                session.commit()
            else:
                print(f"No ticker found for {company.name}")
            
            # Be nice to the server
            time.sleep(0.5)

def get_ticker_for_isin(isin):
    """
    Simple function to get a ticker for a single ISIN code.
    """
    ticker = get_ticker_from_isin(isin)
    print(f"ISIN: {isin} -> Ticker: {ticker}")
    return ticker

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # If an ISIN is provided as a command-line argument, get its ticker
        isin = sys.argv[1]
        get_ticker_for_isin(isin)
    else:
        # Otherwise, update all companies in the database
        update_company_tickers()