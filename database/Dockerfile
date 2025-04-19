FROM python:3.12-slim

COPY database /database

RUN apt-get update && apt-get install -y libpq-dev

WORKDIR /database

RUN pip install alembic sqlmodel psycopg2-binary babel

CMD alembic upgrade head && tail -f /dev/null