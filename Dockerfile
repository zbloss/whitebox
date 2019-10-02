FROM debian:stretch-slim

WORKDIR /

COPY . /

RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y build-essential python3-dev
RUN apt-get install -y python3 python-distribute python3-pip
RUN pip3 install pip --upgrade


RUN pip3 install -r requirements.txt
RUN pip3 install torch===1.2.0 torchvision===0.4.0 -f https://download.pytorch.org/whl/torch_stable.html

CMD jupyter notebook --no-browser --ip 0.0.0.0 --port 8888 --allow-root /
