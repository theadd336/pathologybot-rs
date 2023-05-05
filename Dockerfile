FROM python:latest

WORKDIR /workspace

COPY . .

RUN python -m pip install -r requirements-dev.txt
RUN python -m pip install -e .

CMD ["sleep", "infinity"]
