FROM python:3.11-slim

WORKDIR /rhob

# Install core + dev dependencies first for better layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[dev]"

COPY . .

CMD ["pytest", "tests/", "-q"]
