# -*- coding: utf-8 -*-
# @Time         : 2023/3/20 上午11:30
# @Author       : *******
# @File         : persist.py
# @Description  :


from utils.myredis import RedisPipeline
from algo_config import config

mycache = RedisPipeline(**{
    'host': config['REDIS_HOST'],
    'port': config['REDIS_PORT'],
    'password': config['REDIS_PASSWORD'],
    'db': config['REDIS_DBINDEX_0'],
})

field = ['lev1_name', 'lev2_name', 'lev3_name', 'location', 'positive', 'neutral', 'negative', ]
cache_key_template = 'dazhongdianping_comments_sentiment_{lev1_name}_{lev2_name}_{lev3_name}_{location}'


def p(data):
    # redis
    cache_key = cache_key_template.format(lev1_name=data[0], lev2_name=data[1], lev3_name=data[2], location=data[3])
    r = mycache.hmset(cache_key, {
        'lev1_name': data[0],
        'lev2_name': data[1],
        'lev3_name': data[2],
        'location': data[3],
        'positive': data[4],
        'neutral': data[5],
        'negative': data[6],
    })
    # r=mycache.p
