FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.py .
COPY titanic_server.py .

ENV PYTHONUNBUFFERED=1

CMD ["python", "titanic_server.py"]
