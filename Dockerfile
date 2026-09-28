FROM python:3.13-slim

WORKDIR /code

RUN pip install uv

COPY pyproject.toml uv.lock /code/

RUN uv sync --frozen --no-dev

COPY ./app /code/app

CMD ["uv", "run", "fastapi", "run", "app/main.py", "--port", "80"]
