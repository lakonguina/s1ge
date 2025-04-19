FROM python:3.11

WORKDIR /app
COPY database /database

RUN python -m pip install "fastapi[standard]" sqlmodel psycopg2-binary slugify babel alembic

COPY . .

WORKDIR /app
CMD ["/bin/sh", "-c", "cd /database && python -m alembic upgrade head && cd /app && fastapi dev app/app.py --host 0.0.0.0 --port 8000"]