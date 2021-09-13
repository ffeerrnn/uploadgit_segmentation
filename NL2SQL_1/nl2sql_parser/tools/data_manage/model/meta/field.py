# -*- coding: utf-8 -*-

class Field:
    # 表名
    TABLE = "field_info"

    # 字段名
    ID = "id"
    ORIGIN_NAME = "origin_name"
    NAME = "name"
    ORIGIN_TYPE = "origin_type"
    VALUE_TYPE = "value_type"
    DATA_ENUM = "data_enum"
    TABLE_ID = "table_id"

    @staticmethod
    def get_return_keys():
        return "id, origin_name, name, origin_type, value_type, data_enum, table_id"
