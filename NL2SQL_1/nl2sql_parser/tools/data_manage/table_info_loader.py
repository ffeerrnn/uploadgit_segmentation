# -*- coding: utf-8 -*-

import json
import pandas as pd
from django.core.cache import cache

from nl2sql_parser.tools.data_manage.dao.utils.mysql_utils import Mysql
from nl2sql_parser.tools.data_manage.model.meta.table import Table
from nl2sql_parser.tools.data_manage.model.meta.field import Field
from nl2sql_parser.tools.common.flash_keyword import KeywordProcessor

from conf.file_conf import *

class TableInfoLoader(object):
    """
    从table_info表中加载表信息
    """

    def __init__(self):
        self.cache_prefix = 'meta'
        self.mysql_db = Mysql()
        self.table = Table()
        self.field = Field()

    def make_cache(self, key, data, except_message=''):
        try:
            cache_key = '_'.join([self.cache_prefix, key])
            cache.set(cache_key, data, timeout=None)
        except Exception as e:
            print(e)
            raise Exception(except_message)

    def get_cache(self, key):
        cache_key = '_'.join([self.cache_prefix, key])
        data = cache.get(cache_key, '')
        return data

    def cache_fresh(self):
        """
        刷新缓存
        """
        table_ids = self.get_table_ids_from_db()
        for table_id in table_ids:
            table_info = self.get_table_info_from_db(table_id)

        imagegsd_satellite_dict = self.get_imagegsd_satellite_dict()
        self.make_cache('imagegsd_satellite_dict', imagegsd_satellite_dict)

        oral_imagegsd_dict = self.get_oral_imagegsd_dict()
        self.make_cache('oral_imagegsd_dict', oral_imagegsd_dict)

    def get_table_ids_from_db(self):
        """

        :return:
        """
        sql = "select {field} from {table}".format(field=self.table.ID, table=self.table.TABLE)
        results = self.mysql_db.fetch_all(sql, [])
        results = [i[0] for i in results]

        key_str = 'table_ids'
        self.make_cache(key_str, results, 'table_ids缓存生成异常')
        return results

    def get_table_info_from_db(self, table_id):
        """

        :return:
        """
        sql = "select * from {table} where {field} = %s".format(table=self.table.TABLE, field=self.table.ID)
        result = self.mysql_db.fetch_one(sql, [str(table_id)])

        return_keys = self.table.get_return_keys()

        return_json_data = dict(zip(return_keys.replace("`", "").replace(" ", "").split(','), result))
        return_json_data['name'] = [i.upper() for i in return_json_data['name'].strip().split(',')]
        return_json_data['name'].append(return_json_data['origin_name'].upper())
        return_json_data['type_name'] = [i.upper() for i in return_json_data['type_name'].strip().split(',')]

        return_json_data["fields"] = self.get_fields_info_from_db(return_json_data['field_set'])

        key_str = 'table'+str(table_id)
        self.make_cache(key_str, return_json_data, '表{}表信息缓存生成异常'.format(str(table_id)))

        return return_json_data

    def get_table_info(self, table_id):
        """

        """
        key_str = 'table'+str(table_id)
        table_info = self.get_cache(key_str)
        if not table_info:
            table_info = self.get_table_info_from_db(table_id)

        return table_info

    def get_first_table_info(self, table_ids):
        """
        地球观测返回的table_ids可能有很多，在用其中一个表的信息做解析时，其他表的字段如果有枚举值补充上
        """
        first_table_info = self.get_table_info(table_ids[0])

        for table_id in table_ids[1:]:
            table_info = self.get_table_info(table_id)
            for f1 in first_table_info['fields']:
                for f2 in table_info['fields']:
                    if f1['alias'] == f2['alias']:
                        print('alias', f1['alias'])
                        print(f1['dataEnum'])
                        new_data_enum = f1['dataEnum']
                        new_data_enum.extend(f2['dataEnum'])
                        new_data_enum = list(set(new_data_enum))
                        f1['dataEnum'] = new_data_enum

        return first_table_info        

    def get_fields_info_from_db(self, field_ids_str):
        """
        获取某张表包含的所有字段信息
        :return:
        """
        field_ids_list = [str(i) for i in field_ids_str.split(',')]
        sql = "select * from {table} where {field} in ".format(table=self.field.TABLE, field=self.field.ID)
        ids_sql = "(" + ','.join(['%s' for i in field_ids_list]) + ")"
        sql += ids_sql

        results = self.mysql_db.fetch_all(sql, field_ids_list)
        return_keys = self.field.get_return_keys()

        fields = []
        for result in results:
            data = dict(zip(return_keys.replace("`", "").replace(" ", "").split(','), result))
            field = {
                'id': data['id'],
                'alias': data['name'],
                'name': data['origin_name'],
                'type': data['value_type'],
                'dataEnum': data['data_enum'].strip().split(',') if data['data_enum'] else []
            }
            key_str = 'field'+str(data['id'])
            self.make_cache(key_str, field, '字段{}信息缓存生成异常'.format(str(data['id'])))
            fields.append(field)

        return fields

    def get_field_info_from_db(self, field_id):
        """
        从数据库获取某个字段的信息
        :return:
        """
        sql = "select * from {table} where {field} = %s".format(table=self.field.TABLE, field=self.field.ID)

        result = self.mysql_db.fetch_one(sql, [str(field_id)])
        return_keys = self.field.get_return_keys()

        data = dict(zip(return_keys.replace("`", "").replace(" ", "").split(','), result))
        field = {
            'id': data['id'],
            'alias': data['name'],
            'name': data['origin_name'],
            'type': data['value_type'],
            'dataEnum': data['data_enum'].strip().split(',') if data['data_enum'] else []
        }
        key_str = 'field'+str(data['id'])
        self.make_cache(key_str, field, '字段{}信息缓存生成异常'.format(str(data['id'])))

        return field

    def get_field_info(self, field_id):
        """
        """
        key_str = 'field'+str(field_id)
        field = self.get_cache(key_str)
        if not field:
            field = self.get_field_info_from_db(field_id)
        
        return field

    def get_table_ids(self):
        """
        """
        key_str = 'table_ids'
        table_ids = self.get_cache(key_str)
        if not table_ids:
            table_ids = self.get_table_ids_from_db()
        return table_ids

    def table_name_id_dict(self):
        """
        获取表名和id对应的字典
        """
        all_table_ids = self.get_table_ids()

        name_id_dict = {}
        for table_id in all_table_ids:
            table_info = self.get_table_info(table_id)
            name_list = table_info['name']
            name_list.extend(table_info['type_name'])
            name_list = list(set(name_list))
            for name in name_list:
                if name not in name_id_dict:
                    name_id_dict[name] = [table_id]
                else:
                    name_id_dict[name].append(table_id)
        
        key_str = 'table_name_id_dict'
        self.make_cache(key_str, name_id_dict, '表名id字典信息缓存生成异常')
        
        return name_id_dict

    def table_name_keywordprocessor(self):
        """        
        """
        table_name_id_dict = self.table_name_id_dict()
        print(table_name_id_dict)
        keywordprocessor = KeywordProcessor()
        keywordprocessor.add_keywords_from_list(list(table_name_id_dict.keys()))
        return keywordprocessor

    def get_imagegsd_satellite_dict(self):
        """
        获取分辨率对应卫星的字典
        """
        df = pd.read_excel(sensor_imagegsd_file_path, dtype={"imagegsd":str})
        imagegsd_satellite_dict = {}
        
        for i, row in df.iterrows():
            imagegsd = str(row['imagegsd'])
            if imagegsd not in imagegsd_satellite_dict:
                imagegsd_satellite_dict[imagegsd] = [str(row['satellite'])]
            else:
                imagegsd_satellite_dict[imagegsd].append(str(row['satellite']))

        return imagegsd_satellite_dict

    def get_oral_imagegsd_dict(self):
        """
        读取口语与分辨率对应的解析关系
        """
        with open(oral_imagegsd_file_path) as f:
            oral_imagegsd_dict = json.loads(f.read())

        return oral_imagegsd_dict


TableInfoLoader().cache_fresh()

if __name__ == '__main__':

    from django.conf import settings
    import NL2SQL.settings as app_setting
    settings.configure(default_settings=app_setting)

    loader = TableInfoLoader()

    data = loader.get_table_info(1)
    print(data)