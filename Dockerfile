FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8080

ENV PORT=8080

CMD ["bash", "-lc", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT}"]