FROM python:3.12.14-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE MANIFEST.in ./
COPY constraints/studies.txt ./constraints/studies.txt
COPY src/ ./src/

# Install the core package with a fixed build backend, without model extras.
RUN python -m pip install --no-cache-dir setuptools==84.0.0 \
    && python -m pip install --no-cache-dir --no-build-isolation \
        --constraint constraints/studies.txt .

COPY examples/api/quickstart.py ./quickstart.py

RUN mkdir -p /app/outputs && chown 1000:1000 /app/outputs
USER 1000:1000

CMD ["python", "quickstart.py"]
