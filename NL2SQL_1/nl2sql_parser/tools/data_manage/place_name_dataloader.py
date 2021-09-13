# -*- coding: utf-8 -*-

import json

from nl2sql_parser.tools.data_manage.dao.utils.mysql_utils import Mysql
from nl2sql_parser.tools.common.flash_keyword import KeywordProcessor
from conf.file_conf import admincode_file_path

from django.core.cache import cache

class PlaceNameDataloader(object):
    """
    地区数据dataloader
    """
    def __init__(self):
        self.mysql_db = Mysql()
        self.admincode_dict = self.get_admincode()

    def get_admincode(self):
        """
        读取行政区划代码
        """
        with open(admincode_file_path, 'r', encoding='utf-8') as f:
            admincode_dict = json.loads(f.read())

        return admincode_dict

    def get_data_from_db(self):
        """
        从数据库获取数据
        :return:
        """
        try:
            sql = "select `province`, `province_abbre`, `city`, `city_abbre`, `district`, `district_abbre` from `administrative_divisions`"
            values = self.mysql_db.fetch_all(sql, [])
        except:
            values = []
            print("Error: 无法获取数据")
            raise Exception('查询失败,无法获取数据')

        def add_data_to_dict(dic, key, value):
            if key not in dic:
                dic[key] = [value]
            else:
                dic[key].append(value)

        name_full_name_dict = {} # 名称，和对应该名称的所有地区信息的字典
        full_name_name_dict = {}
        for province, province_abbre, city, city_abbre, district, district_abbre in values:
            if province:
                full_name = province
                add_data_to_dict(name_full_name_dict, province, full_name)
                admincode = self.admincode_dict.get(full_name, '')
                full_name_name_dict[full_name] = {'name': province, 'level':1, 'full_name': full_name, 'admincode': admincode}
            if province_abbre:
                full_name = province
                add_data_to_dict(name_full_name_dict, province_abbre, full_name)
                admincode = self.admincode_dict.get(full_name, '')
                full_name_name_dict[full_name] = {'name': province, 'level':1, 'full_name': full_name, 'admincode': admincode}
            if city:
                full_name = province+city
                add_data_to_dict(name_full_name_dict, city, full_name)
                admincode = self.admincode_dict.get(full_name, '')
                full_name_name_dict[full_name] = {'name': city, 'level':2, 'full_name': full_name, 'admincode': admincode}
            if city_abbre:
                full_name = province+city
                add_data_to_dict(name_full_name_dict, city_abbre, full_name)
                admincode = self.admincode_dict.get(full_name, '')
                full_name_name_dict[full_name] = {'name': city, 'level':2, 'full_name': full_name, 'admincode': admincode}
            if district:
                full_name = province+city+district
                add_data_to_dict(name_full_name_dict, district, full_name)
                admincode = self.admincode_dict.get(full_name, '')
                full_name_name_dict[full_name] = {'name': district, 'level':3, 'full_name': full_name, 'admincode': admincode}
            if district_abbre:
                full_name = province+city+district
                add_data_to_dict(name_full_name_dict, district_abbre, full_name)
                admincode = self.admincode_dict.get(full_name, '')
                full_name_name_dict[full_name] = {'name': district, 'level':3, 'full_name': full_name, 'admincode': admincode}

        new_name_full_name_dict = {}
        for k in name_full_name_dict:
            full_name_list = name_full_name_dict[k]
            new_full_name_list = sorted(full_name_list, key=lambda x: full_name_name_dict[x]['level'])
            new_name_full_name_dict[k] = new_full_name_list

        return new_name_full_name_dict, full_name_name_dict

    def make_cache(self):
        """
        地区描述字典放到缓存中
        :return:
        """
        name_full_name_dict, full_name_name_dict = self.get_data_from_db()
        cache.set('name_full_name_dict', name_full_name_dict)
        cache.set('full_name_name_dict', full_name_name_dict)

    def get_cache(self, cache_key):
        """
        从缓存中读取数据
        :return:
        """
        data = cache.get(cache_key)
        return data

    def get_keywordprocessor(self):
        """
        获取地区名称关键词搜索器
        :return:
        """
        keywordprocessor = KeywordProcessor()
        name_full_name_dict = self.get_cache('name_full_name_dict')
        keywordprocessor.add_keywords_from_list(list(name_full_name_dict.keys()))
        return keywordprocessor

PlaceNameDataloader().make_cache()
