FROM python:3.8-slim

MAINTAINER liuning@shanshu.ai
#
ENV PYTHONUNBUFFERED 1
#
ENV ENGINE_MYSQL_HOST 10.2.1.7
ENV ENGINE_MYSQL_PORT 3306
ENV ENGINE_MYSQL_DBNAME codf_dispatch_engine
ENV ENGINE_MYSQL_USERNAME codf_dispatch_engine_dev
ENV ENGINE_MYSQL_PASSWORD yw^(iTCko4jz$aua
# when deploy dev env, will be overwrited by devops
ENV ENGINE_REDIS_HOST 172.17.0.3
ENV ENGINE_REDIS_PORT 6379
ENV ENGINE_REDIS_USERNAME root
ENV ENGINE_REDIS_PASSWORD 123456
ENV ENGINE_REDIS_CELERY_DBINDEX 0
ENV ENGINE_REDIS_DISPATCH_DBINDEX 1
# when deploy dev env, will be overwrited by devops
ENV ENGINE_RABBITMQ_HOST 172.17.0.5
ENV ENGINE_RABBITMQ_PORT 5672
ENV ENGINE_RABBITMQ_MAN_PORT 15672
ENV ENGINE_RABBITMQ_USERNAME codf_dispatch_engine
ENV ENGINE_RABBITMQ_PASSWORD codf_dispatch_engine
ENV ENGINE_RABBITMQ_VHOST codf_dispatch_engine
#
ENV ENGINE_FLOWER_HOST 127.0.0.1
ENV ENGINE_FLOWER_PORT 5555
#
ENV ENGINE_SNOWFLAKE_HOST 127.0.0.1
ENV ENGINE_SNOWFLAKE_PORT 8910
#
RUN apt-get update && apt-get install -y procps && apt-get install -y vim && apt-get install -y libgomp1 && apt-get install -y gcc python3-dev
RUN mkdir -p /app/codf_dispatch_engine /app/codf_dispatch_engine/tasks /app/codf_dispatch_engine/logs /app/lib3
COPY codf_dispatch_engine/* /app/codf_dispatch_engine/
COPY codf_dispatch_engine/tasks/* /app/codf_dispatch_engine/tasks/
COPY lib3/* /app/lib3/
WORKDIR /app/lib3/
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir sktime-0.7.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
WORKDIR /app/codf_dispatch_engine
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir -r requirements.txt
RUN chmod u+x /app/codf_dispatch_engine/*.sh
ENTRYPOINT ["/bin/bash","-c","/app/codf_dispatch_engine/start.sh"]

EXPOSE 5555 8910 8088