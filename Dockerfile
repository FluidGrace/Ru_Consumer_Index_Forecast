FROM python:3.9

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY *.pkl .

COPY static static

COPY templates/index.html /templates/

COPY *.py .

CMD ["flask", "run", "--host=0.0.0.0"]