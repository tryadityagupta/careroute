FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only what the running service needs (tests, infra and scripts stay out).
COPY careroute ./careroute
COPY mock_llm ./mock_llm
COPY web ./web
COPY data ./data

# Don't run as root.
RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

EXPOSE 8000
CMD ["uvicorn", "--factory", "careroute.api.app:create_app", "--host", "0.0.0.0", "--port", "8000"]
