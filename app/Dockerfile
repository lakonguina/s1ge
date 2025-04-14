FROM python:3.11

WORKDIR /app
COPY database /database

RUN python -m pip install "fastapi[standard]"

COPY . .

CMD fastapi dev app/app.py --host 0.0.0.0