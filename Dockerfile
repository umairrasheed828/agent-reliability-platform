FROM python:3.12-slim

# git is required for the judgekit git dependency
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src ./src
COPY scripts ./scripts
COPY eval ./eval
COPY streamlit_app.py ./

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port", "8501", "--server.address", "0.0.0.0"]