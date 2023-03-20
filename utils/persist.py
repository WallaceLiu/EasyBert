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

cache_key_template = 'dazhongdianping_comments_sentiment_{lev1_name}_{lev2_name}_{lev3_name}_{location}'


def cache(dataset_list):
    """cache positive, neutral, negative percentage

    list, each of item
     {
        'lev1_name': data[0],
        'lev2_name': data[1],
        'lev3_name': data[2],
        'location': data[3],
        'positive': data[4],
        'neutral': data[5],
        'negative': data[6],
    }

    """

    dataset = [(cache_key_template.format(lev1_name=data['lev1_name'],
                                          lev2_name=data['lev2_name'],
                                          lev3_name=data['lev3_name'],
                                          location=data['location']), data) for
               data in dataset_list]
    r = mycache.pipeline_hmset(dataset)
    return r
