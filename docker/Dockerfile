FROM python:3.7

RUN apt-get update 
RUN pip install --upgrade pip

COPY ./requirements.txt /

RUN pip install wheel

RUN pip install -r requirements.txt

RUN pip install jupyter