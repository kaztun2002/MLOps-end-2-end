FROM ghcr.io/astral-sh/uv:latest AS uv

FROM python:3.13-slim

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
	UV_COMPILE_BYTECODE=1 \
	UV_LINK_MODE=copy \
	PATH="/opt/venv/bin:/usr/local/bin:/usr/bin:/bin" \
	PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

COPY --from=uv /uv /uvx /bin/

RUN apt-get update \
	&& apt-get install --no-install-recommends -y git libgomp1 \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-install-project

COPY dvc.yaml dvc.lock params.yaml dvc_pipeline.py docker_entrypoint.py ./
COPY .dvc/config .dvc/config
COPY notebooks/sales_timeseries_2025.csv notebooks/sales_timeseries_2025.csv

RUN dvc config core.no_scm true \
	&& groupadd --system app \
	&& useradd --system --gid app --home-dir /app app \
	&& chown -R app:app /app

USER app

ENTRYPOINT ["python", "docker_entrypoint.py"]
CMD ["dvc", "repro"]
