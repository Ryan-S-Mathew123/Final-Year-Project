FROM python:3.11-slim
WORKDIR /app
COPY apps/server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY apps/server /app/apps/server
COPY models /app/models
WORKDIR /app/apps/server
EXPOSE 8000
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]
