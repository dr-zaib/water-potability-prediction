FROM python:3.9

RUN pip install pipenv 

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "model_h2O_potability.bin", "./"]

EXPOSE 9090

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:8080", "predict:app"]