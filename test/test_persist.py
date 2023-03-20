# -*- coding: utf-8 -*-
# @Time         : 2023/3/20 上午10:22
# @Author       : *******
# @File         : test_persist.py
# @Description  :
import unittest
from utils.myredis import RedisPipeline
from algo_config import config
from utils.persist import p


class TestPersit(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_cache(self):
        mycache = RedisPipeline(**{
            'host': config['REDIS_HOST'],
            'port': config['REDIS_PORT'],
            'password': config['REDIS_PASSWORD'],
            'db': config['REDIS_DBINDEX_0'],
        })
        r = mycache.hmset('dazhoongdianping_comments_sentiment_北京_北京_海淀区_玉渊潭公园', {
            'lev1_name': '北京',
            'lev2_name': '北京',
            'lev3_name': '玉渊潭公园',
            'positive': 90,
            'neutral': 5,
            'negative': 5,
        })
        print(r)

        print(mycache.hgetall('dazhoongdianping_comments_sentiment_北京_北京_海淀区_玉渊潭公园'))

    def test_p(self):
        p(('北京', '北京', '海淀区', '玉渊潭公园', 90, 5, 5))
