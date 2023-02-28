FROM python:3.7-slim

MAINTAINER liuning800203@aliyun.com
#
ENV PYTHONUNBUFFERED 1
#
RUN mkdir -p /app/easybert /app/easybert/result
# COPY * /app/easybert/
COPY ["*","/app/easybert"]
WORKDIR /app/easybert
RUN python -m pip install --upgrade pip
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir -r requirements.txt
# RUN chmod u+x /app/easybert/*.sh
# ENTRYPOINT ["/bin/bash","-c","/app/codf_dispatch_engine/start.sh"]

# EXPOSE 5555 8910 8088