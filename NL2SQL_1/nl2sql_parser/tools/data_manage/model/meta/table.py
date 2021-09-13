# -*- coding: utf-8 -*-

class Table(object):
    # 表名
    TABLE = "table_info"

    # 字段名
    ID = "id"
    ORIGIN_NAME = "origin_name"
    NAME = "name"
    TYPE_NAME = "type_name"
    FIELD_SET = "field_set"

    @staticmethod
    def get_return_keys():
        return "id, origin_name, name, type_name, field_set"
