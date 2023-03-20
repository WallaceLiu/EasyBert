# -*- coding: utf-8 -*-
"""
@time           : 2019/5/24 下午10:22
@author         : liuning11@jd.com
@file           : myredis.py
@description    : redis pipeline
"""
import redis
from utils.udecorator import synchronized


class RedisPipeline:
    instance = None

    @synchronized
    def __new__(cls, *args, **kwargs):
        """

        Args:
            *args:
            **kwargs:
        """
        if cls.instance is None:
            cls.instance = super().__new__(cls)
            _conf = kwargs
            _pool = redis.ConnectionPool(**_conf)
            cls.red = redis.Redis(connection_pool=_pool)
        return cls.instance

    def pipeline_set(self, dataset):
        with self.red.pipeline(transaction=False) as p:
            for k, v in dataset:
                p.set(k, v)
            result = p.execute()
        return result

    def pipeline_get(self, keys):
        """

        Args:
            keys:   list,

        Returns:

        """
        with self.red.pipeline(transaction=False) as p:
            for key in keys:
                p.get(key)
            result = p.execute()
        return result

    def pipeline_hmset(self, dataset):
        with self.red.pipeline(transaction=False) as p:
            for k, v in dataset:
                p.hmset(k, v)
            result = p.execute()
        return result

    def set(self, name, value):
        return self.red.set(name, value)

    def pipeline_hmset(self, dataset):
        with self.red.pipeline(transaction=False) as p:
            for k, v in dataset:
                p.hmset(k, v)
            result = p.execute()
        return result

    def pipeline_hmget(self, keys, field):
        """

        Args:
            keys:   list,

        Returns:

        """
        with self.red.pipeline(transaction=False) as p:
            for key in keys:
                p.hmget(key, field)
            result = p.execute()
        return result

    def hgetall(self, name):
        r = self.red.hgetall(name)
        return r

    def hmget(self, name, keys):
        r = self.red.hmget(name, keys)
        return r

    def hmset(self, name, keys):
        r = self.red.hset(name, mapping=keys)
        return r

    def pipeline_hvals(self, keys):
        """

        Args:
            keys:   list,

        Returns:

        """
        with self.red.pipeline(transaction=False) as p:
            for key in keys:
                p.hvals(key)
            result = p.execute()
        return result

    #
    # def phgetall(self, keys):
    #     """
    #
    #     :param keys: list
    #     :return:
    #     """
    #     with self.red.pipeline(transaction=False) as p:
    #         for key in keys:
    #             p.hgetall(key)
    #         result = p.execute()
    #     return result
    #
    # def phmget(self, names, key):
    #     """
    #
    #     :param names: list name list
    #     :param keys: val
    #     :return:
    #     """
    #     with self.red.pipeline(transaction=False) as p:
    #         for name in names:
    #             p.hmget(name, key)
    #         result = p.execute()
    #     return result

    def pipeline_exists(self, keys):
        """

        :param keys:
        :return:
        """
        with self.red.pipeline(transaction=False) as p:
            for key in keys:
                p.exists(key)
            result = p.execute()
        return result

    def exists(self, keys):
        """

        :param keys:
        :return:
        """
        return self.red.exists(keys)

    def incr(self, key):
        return self.red.incr(key)

    #
    # def is_pexist(self, keys):
    #     if len(keys) > 0:
    #         with self.red.pipeline(transaction=False) as p:
    #             for key in keys:
    #                 p.exists(key)
    #             result = p.execute()
    #         return result
    #     else:
    #         return [False]
    #
    # def psmembers(self, keys):
    #     """
    #
    #     :param keys:
    #     :return:
    #     """
    #     with self.red.pipeline(transaction=False) as p:
    #         for key in keys:
    #             p.smembers(key)
    #         result = p.execute()
    #     return result
    #
    # def smembers(self, key):
    #     """
    #
    #     :param key:
    #     :return:
    #     """
    #     return self.red.smembers(key)
    #
    # def lock(self, name, value, time_out):
    #     s = self.red.setnx(name, value)
    #     if s == 1:
    #         self.red.expire(name, time_out)
    #     return s
    #
    # def srem(self, name, *values):
    #     return self.red.srem(name, *values)
    #
    # def sadd(self, name, *values):
    #     return self.red.sadd(name, *values)

    def delete(self, *names):
        return self.red.delete(*names)
