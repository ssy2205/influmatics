FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV MPLCONFIGDIR=/app/.cache/matplotlib

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY influmatics ./influmatics
COPY data ./data
COPY legacy ./legacy
COPY web ./web

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir ".[web,bio]" numpy matplotlib treetime

RUN mkdir -p "$MPLCONFIGDIR" \
    && python -c "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; from matplotlib import font_manager; plt.figure(); plt.plot([0, 1]); plt.close(); print(len(font_manager.fontManager.ttflist))"

CMD exec python -m uvicorn web.server.app:app --host 0.0.0.0 --port "$PORT"
