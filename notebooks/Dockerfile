FROM python:3.6.9

WORKDIR /

COPY . /

# RUN apt-get update && \
#     apt-get install -y \
#         build-essential \
#         make \
#         gcc \
#         locales \
#         libgdal20 libgdal-dev && \
#     python -m pip install numpy cython --no-binary numpy,cython && \
#     python -m pip install \
#         "rasterio>=1.0a12" fiona shapely \
#         --pre --no-binary rasterio,fiona,shapely

# RUN dpkg-reconfigure locales && \
#     locale-gen C.UTF-8 && \
#     /usr/sbin/update-locale LANG=C.UTF-8

# ENV LC_ALL C.UTF-8

RUN pip3 install -r requirements.txt

#CMD jupyter notebook --no-browser --ip 0.0.0.0 --port 8888 --allow-root
