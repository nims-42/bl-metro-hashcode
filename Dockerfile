FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RESULTS_DIR=/results

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

# Ring port and ping-req side channel (section 7)
EXPOSE 7000 7001

CMD ["python", "-m", "src.node"]
