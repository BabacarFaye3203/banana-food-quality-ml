FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN pip install --no-cache-dir uv

COPY src ./src

RUN uv sync --frozen

COPY models ./models
COPY scripts ./scripts

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "banana_ml.api:app", "--host", "0.0.0.0", "--port", "8000"]