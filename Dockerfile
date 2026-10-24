# Reproducible CPU environment for experiments (training is faster natively on Apple MPS / CUDA).
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 cmake g++ git curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv
WORKDIR /clarity
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project
COPY . .
RUN uv sync --frozen
CMD ["make", "test"]
