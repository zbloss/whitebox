FROM python:3.6-slim-stretch

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /

RUN ls

COPY . /

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        make \
        gcc \
        locales \
        libgdal20 libgdal-dev  

RUN dpkg-reconfigure locales && \
    locale-gen C.UTF-8 && \
    /usr/sbin/update-locale LANG=C.UTF-8

ENV LC_ALL C.UTF-8

RUN pip install -r requirements.txt

EXPOSE 8000

CMD python3 manage.py runserver 0.0.0.0:8000