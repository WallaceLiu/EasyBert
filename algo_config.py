# -*- coding: utf-8 -*-
"""
@time           : 2021/9/6 下午12:15
@author         : liuning@shanshu.ai
@file           : app_config.py
@description    : Application Configuration

"""
import logging
import logging.config
import os
import sys
import yaml
import time

config = {
    # ------------------------------------------------------------------------------------
    # MySQL Configuration
    # ------------------------------------------------------------------------------------
    'MYSQL_HOST': os.environ.get('MYSQL_HOST') or 'localhost',
    'MYSQL_PORT': os.environ.get('MYSQL_PORT') or 3306,
    'MYSQL_DBNAME': os.environ.get('MYSQL_DBNAME') or 'cctc',
    'MYSQL_USERNAME': os.environ.get('MYSQL_USERNAME') or 'root',
    'MYSQL_PASSWORD': os.environ.get('MYSQL_PASSWORD') or '123456',
    # ------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------
    'MYSQL_CHUNK_SIZE': 1000,
    'MYSQL_THREAD_POOL': 10,
    # ------------------------------------------------------------------------------------
    # Redis Configuration
    # ------------------------------------------------------------------------------------
    'REDIS_HOST': os.environ.get('REDIS_HOST') or 'localhost',
    'REDIS_PORT': os.environ.get('REDIS_PORT') or 6379,
    'REDIS_USERNAME': os.environ.get('REDIS_USERNAME') or '',
    'REDIS_PASSWORD': os.environ.get('REDIS_PASSWORD') or '123456',
    'REDIS_DBINDEX_0': os.environ.get('REDIS_DBINDEX_0') or 0,
    'REDIS_DBINDEX_1': os.environ.get('REDIS_DBINDEX_1') or 1,
}
