FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

LABEL maintainer='sushmita.iar0@gmail.com'

# install as a package
COPY requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

RUN python -m spacy download en_core_web_lg

RUN python -m nltk.downloader omw-1.4

# copy code
COPY . /app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]