"""Module for retrieving and processing declaration data from AMF API."""
import datetime
from datetime import datetime, timedelta
import time

import requests
from sqlmodel import Session, func, select

from src.parse import extractor
from database.model import Declaration
from database.engine import engine
from src.quotes import get_quotes


def get_latest_transaction_date() -> datetime.date:
    """Gets the date of the most recent transaction in the database.

    Returns:
        date: The date of the latest declaration or Jan 1, 2019 if no data exists.
    """
    with Session(engine) as session:
        latest_declaration_date = session.exec(
            select(func.max(Declaration.date_))).first()
        if latest_declaration_date:
            return latest_declaration_date
        # If no data at all, start from beginning of 2019
        return datetime(2017, 1, 1).date()


def get_document(path: str) -> bytes:
    """Gets a document from the AMF API.
    
    Args:
        path: Path to the document, which could be in different formats:
              - Full path: /back/api/v1/documents/2023/...
              - Partial path: /documents/2023/...
              - Just the document part: 2023/...
        
    Returns:
        bytes: Document content as bytes.
    """
    base_url = "https://bdif.amf-france.org"
    api_path = "/back/api/v1/documents"
    
    # Handle different path formats
    if path.startswith('/back/api/v1/documents/'):
        # Full path is already provided
        url = f"{base_url}{path}"
    elif path.startswith('/documents/'):
        # Path starts with /documents
        url = f"{base_url}{api_path}{path[10:]}"  # Remove the /documents part
    elif path.startswith('documents/'):
        # Path starts with documents without leading slash
        url = f"{base_url}{api_path}/{path[10:]}"  # Remove the documents/ part
    else:
        # Just the document part (year/id/file.pdf)
        url = f"{base_url}{api_path}/{path}"
    
    print(f"Fetching document from: {url}")
    response = requests.get(url)
    return response.content


def get_declarations() -> None:
    """Fetches and processes declaration details from the AMF API."""
    # Get the current date to use as our end point
    current_date = datetime.now().date()
    # Start with the latest transaction date
    start_date = get_latest_transaction_date()
    while start_date < current_date:
        # Convert to datetime for API formatting
        start_datetime = datetime.combine(start_date, datetime.min.time())
        # End date is either a year later or today, whichever is earlier
        end_datetime = min(
            start_datetime + timedelta(days=365),
            datetime.combine(current_date, datetime.min.time())
        )
        # Format dates for the API
        date_debut = start_datetime.strftime("%Y-%m-%dT22:00:00.000Z")
        date_fin = end_datetime.strftime("%Y-%m-%dT22:59:59.000Z")
        print(f"Fetching declarations from {start_date} to {end_datetime.date()}")
        url = (f"https://bdif.amf-france.org/back/api/v1/informations?"
               f"DateDebut={date_debut}&DateFin={date_fin}&"
               f"TypesInformation=DD&From=0&Size=10000")
        response = requests.get(url)
        list_declarations = response.json()['result']
        # Simply reverse the list to process newest declarations first
        for declaration in reversed(list_declarations):
            print(declaration['dateInformation'], declaration['id'])
            for document in declaration['documents']:
                document_content = get_document(document['path'])
                with Session(engine) as session:
                    extractor(document_content, session)
                time.sleep(1)
        # Move to the next year
        start_date = end_datetime.date()
    print("Fetched all declarations.")
    #get_quotes()