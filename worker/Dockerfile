FROM python:3.12-slim

COPY worker /worker
COPY database /database

ENV PYTHONPATH="/"

WORKDIR /worker
RUN python -m pip install celery redis sqlmodel psycopg2-binary requests pypdfium2 dotenv yfinance pandas

CMD bash -c "python -m celery -A main beat --loglevel=info & python -m celery -A main worker --loglevel=info"