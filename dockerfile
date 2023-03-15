FROM python:3.7-slim

MAINTAINER liuning800203@aliyun.com
#
ENV PYTHONUNBUFFERED 1
#
RUN mkdir -p /app/easybert /app/easybert/result /app/easybert/libs
COPY * /app/easybert/
# WORKDIR /app/easybert/libs
# RUN su pip install gensim-3.8.1-cp37-cp37m-manylinux1_x86_64.whl
# RUN su pip install numpy-1.16.5-cp37-cp37m-manylinux1_x86_64.whl
# RUN su pip install torch-1.4.0-cp37-cp37m-manylinux1_x86_64.whl
WORKDIR /app/easybert
RUN su apt-get update
RUN su apt-get install gcc -y
RUN su apt-get install libffi-devel -y
RUN su apt-get install zlib-devel bzip2-devel -y
RUN su apt-get install libcurl4-openssl-dev libssl-dev -y
RUN su apt-get install zip unzip -y
RUN su apt-get install vim -y
# ENV PYCURL_SSL_LIBRARY=openssl
ENV INSTALL_ON_LINUX=1
RUN su python -m pip install --upgrade pip
RUN su pip install -i http://mirrors.aliyun.com/pypi/simple/ --no-cache-dir --trusted-host mirrors.aliyun.com --default-timeout=120 -r requirements.txt

# EXPOSE 5555 8910 8088