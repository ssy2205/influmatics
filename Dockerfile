FROM node:20-bookworm-slim AS frontend-build

WORKDIR /app/web/frontend

COPY web/frontend/package*.json ./
RUN npm ci

COPY web/frontend ./

ARG VITE_API_BASE=""
ENV VITE_API_BASE=$VITE_API_BASE

RUN npm run build


FROM python:3.10-slim

ARG NEXTCLADE_VERSION=3.21.2

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV MPLCONFIGDIR=/app/.cache/matplotlib
ENV INFLUMATICS_NEXTCLADE_DATASET=/opt/nextclade-datasets/flu_h3n2_ha

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        fonts-dejavu-core \
        iqtree \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL \
        "https://github.com/nextstrain/nextclade/releases/download/${NEXTCLADE_VERSION}/nextclade-x86_64-unknown-linux-gnu" \
        -o /usr/local/bin/nextclade \
    && chmod +x /usr/local/bin/nextclade \
    && nextclade --version \
    && mkdir -p /opt/nextclade-datasets \
    && nextclade dataset get --name "flu_h3n2_ha" --output-dir "$INFLUMATICS_NEXTCLADE_DATASET" \
    && test -f "$INFLUMATICS_NEXTCLADE_DATASET/pathogen.json"

COPY pyproject.toml README.md ./
COPY influmatics ./influmatics
COPY data ./data
COPY legacy ./legacy
COPY web ./web
COPY --from=frontend-build /app/web/frontend/dist ./web/frontend/dist

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir ".[web,bio]" numpy matplotlib phylo-treetime

RUN (iqtree2 -version || iqtree -version) \
    && treetime --help >/dev/null \
    && nextclade --version

RUN mkdir -p "$MPLCONFIGDIR" \
    && python -c "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; from matplotlib import font_manager; plt.figure(); plt.plot([0, 1]); plt.close(); print(len(font_manager.fontManager.ttflist))"

CMD exec python -m uvicorn web.server.app:app --host 0.0.0.0 --port "$PORT"
