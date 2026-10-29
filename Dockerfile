
FROM python:3.11-slim

WORKDIR /code

RUN pip install --no-cache-dir uv

COPY requirements-docker.txt /code/

RUN uv pip install --system --no-cache \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2+cpu torchvision==0.17.2+cpu

RUN uv pip install --system --no-cache \
    -r requirements-docker.txt

COPY ./app /code/app

CMD ["fastapi", "run", "app/main.py", "--port", "80"]
