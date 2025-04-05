import pypdfium2
import re
from database.model import (
    Declaration,
    Company,
    Currency,
    Person,
    Transaction,
    TransactionNature,
    TransactionPlace,
    TransactionInstrument,
)

from database.engine import get_or_create

from sqlmodel import Session, select
from datetime import datetime, date

def format_date(date_str: str) -> date | None:
    """Convert a French date string to a Python date object.

    Args:
        date_str: A date string in the format "27 décembre 2024"

    Returns:
        A Python date object or None if the format is unexpected
    """
    french_months = {
        'janvier': '01', 'février': '02', 'mars': '03', 'avril': '04',
        'mai': '05', 'juin': '06', 'juillet': '07', 'août': '08',
        'septembre': '09', 'octobre': '10', 'novembre': '11', 'décembre': '12'
    }
    # Parse the date in format "27 décembre 2024"
    date_parts = date_str.split()
    try:
        if len(date_parts) == 3:
            day = date_parts[0].zfill(2)  # Ensure day is two digits
            month = french_months.get(date_parts[1].lower(), date_parts[1])
        year = date_parts[2]
        # Convert to a Python date object
        return datetime.strptime(f"{day}/{month}/{year}", "%d/%m/%Y").date()
    except Exception as e:
        print(f"Error formatting date {date_str}: {e}")
        return None  # Return None if the date format is unexpected

def extractor(content: str, session: Session):
    pdf = pypdfium2.PdfDocument(content)
    text = "".join([page.get_textpage().get_text_range() for page in pdf])
    text = " ".join(text.splitlines())

    declaration, transaction = text.split('DETAIL DE LA TRANSACTION', 1)
    transaction = re.split(
        r'DETAIL DE LA TRANSACTION(?:\s*\d*)',
        transaction)
    id_document = re.search(r'(\d{4}DD\d+)', declaration).group(1).strip()
    isin = re.search(r'([A-Z]{2}\d*[A-Z0-9]*)\s*-?\s*DD\d+', declaration).group(1).strip()
    date_ = re.search(r'(\d{1,2}\s+\w+\s+\d{4})', declaration).group(1).strip()
    company_name = re.search(r'NOM\s*:\s*(.*?)(?:\s+LEI\s*:|$)', declaration).group(1).strip()
    declarant_name = re.search(r'NOM /FONCTION.*?:(.*?)(?:NOTIFICATION|$)', declaration, re.DOTALL).group(1).strip()

    company = get_or_create(
        session=session,
        model=Company,
        query_keys=["name", "isin"],
        name=company_name,
        isin=isin
    )
    person = get_or_create(
        session=session,
        model=Person,
        query_keys=["name", "id_company"],
        name=declarant_name,
        id_company=company.id_company
    )
    declaration = get_or_create(
        session=session,
        model=Declaration,
        query_keys=["id_document"],
        id_document=id_document,
        id_person=person.id_person,
        date_=format_date(date_),
    )
    # Check if this declaration already has transactions
    existing_transactions = session.exec(
        select(Transaction).where(Transaction.id_declaration == declaration.id_declaration)
    ).all()
    
    # If transactions exist, delete them to allow reprocessing
    if existing_transactions:
        print(f"Declaration {declaration.id_document} already exists with {len(existing_transactions)} transactions. Reprocessing...")
        for t in existing_transactions:
            session.delete(t)
        session.flush()


    for t in transaction:
        date_str, rest = t.split('DATE DE LA TRANSACTION :', 1)[1].split('LIEU DE LA TRANSACTION :', 1)
        place_str, rest = rest.split('NATURE DE LA TRANSACTION :', 1)
        nature_str, rest = rest.split('DESCRIPTION DE L’INSTRUMENT FINANCIER :', 1)
        instrument_str, transactions_str = rest.split('INFORMATION DETAILLEE PAR OPERATION', 1)
        instrument_str = instrument_str.split("CODE D’IDENTIFICATION")[0].strip()
        transaction, aggrat = transactions_str.split("INFORMATIONS AGREGEES")
        transaction = transaction.strip().split("PRIX UNITAIRE :")[1:]
        _, aggrat = aggrat.split("GRATUITES OU DE PERFORMANCES :")
        aggrat = aggrat.split("DATE DE RECEPTION DE LA NOTIFICATION")
        aggrat = aggrat[0].strip()

        place = get_or_create(
            session=session,
            model=TransactionPlace,
            query_keys=["name"],
            name=place_str
        )
        nature = get_or_create(
            session=session,
            model=TransactionNature,
            query_keys=["name"],
            name=nature_str
        )
        instrument = get_or_create(
            session=session,
            model=TransactionInstrument,
            query_keys=["name"],
            name=instrument_str
        )

        t_count = 0
        total_price = 0
        total_volume = 0
        total_price_volume = 0
        for e in transaction:
            price_currency, volume = e.split("VOLUME :")
            price_currency_match = re.search(r'([\d\s]+[.,]?\d*)\s+(.+)', price_currency.strip())
            price = float(price_currency_match.group(1).strip().replace(' ', '').replace(',', '.'))
            currency = price_currency_match.group(2).strip()
            currency = get_or_create(
                session=session,
                model=Currency,
                query_keys=["name"],
                name=currency
            )
            volume = float(volume.strip().replace(' ', '').replace(',', '.'))
            transaction = Transaction(
                id_company=company.id_company,
                id_declaration=declaration.id_declaration,
                id_person=person.id_person,
                id_currency=currency.id_currency,
                id_transaction_place=place.id_transaction_place,
                id_transaction_nature=nature.id_transaction_nature,
                id_transaction_instrument=instrument.id_transaction_instrument,
                price=price,
                volume=volume,
                date_=format_date(date_str),
                is_exercice_on_option_or_free_attribution_of_stocks=False if "NON" in aggrat else True
            )
            session.add(transaction)
            session.commit()
            t_count += 1
            total_price += price
            total_volume += volume
            total_price_volume += price * volume