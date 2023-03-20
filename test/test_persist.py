# -*- coding: utf-8 -*-
# @Time         : 2023/3/20 上午10:22
# @Author       : *******
# @File         : test_persist.py
# @Description  :
import unittest
from utils.persist import cache


class TestPersit(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_cache(self):
        r = cache([
            {'lev1_name': '北京',
             'lev2_name': '北京',
             'lev3_name': '海淀区',
             'location': '玉渊潭公园',
             'positive': 90,
             'neutral': 5,
             'negative': 5},
            {'lev1_name': '北京',
             'lev2_name': '北京',
             'lev3_name': '东城区',
             'location': '天坛公园',
             'positive': 90,
             'neutral': 5,
             'negative': 5}
        ])
        print(r)
