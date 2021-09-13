# -*- coding: utf-8 -*-

import json
from django.db import connection
from django.core.paginator import Paginator

class Mysql(object):

    def __init__(self):
        # 打开数据库连接
        self.db = connection

    def execute_sql(self, sql, params):
        """
        执行单条SQL语句
        :param sql: SQL语句
        :return:
        """

        # 使用cursor()方法获取操作游标
        cursor = self.db.cursor()

        try:
            # 执行SQL语句
            cursor.execute(sql, params)
            # 提交修改
            self.db.commit()
        except:
            # 发生错误时回滚
            self.db.rollback()
            print("数据库执行失败,操作已回滚")
            raise Exception('数据库执行失败,操作已回滚')

    def execute_sqls(self, sqls):
        """
        执行多条SQL语句，支持事务操作
        :param sqls: 多条SQL语句
        :return:
        """

        # 使用cursor()方法获取操作游标
        cursor = self.db.cursor()

        try:
            for sql, params in sqls:
                # 执行SQL语句
                cursor.execute(sql, params)
            # 提交修改
            self.db.commit()
        except:
            # 发生错误时回滚
            self.db.rollback()
            print("数据库执行失败,操作已回滚")
            raise Exception('数据库执行失败,操作已回滚')

    def fetch_one(self, sql, params):
        """
        查询单条数据
        :param sql: SQL查询语句
        :return: 数据元组
        """
        # 使用cursor()方法获取操作游标
        cursor = self.db.cursor()

        try:
            # 执行SQL语句
            cursor.execute(sql, params)
            # 获取记录
            data = cursor.fetchone()
            if data is None:
                data = []
            data = self.trans_db_data_to_py_data(data)
            return data
        except:
            print("Error: 无法获取数据")
            raise Exception('查询失败,无法获取数据')

    def fetch_all(self, sql, params):
        """
        查询多条数据
        :param sql: SQL语句
        :return: 列表，多条数据元组
        """
        # 使用cursor()方法获取操作游标
        cursor = self.db.cursor()

        try:
            # 执行SQL语句
            cursor.execute(sql, params)
            # 获取所有记录列表
            results = cursor.fetchall()
            return results

        except:
            print("Error: 无法获取数据")
            raise Exception('查询失败,无法获取数据')

    # 新增一行数据的SQL
    @staticmethod
    def build_add_data_sql(table, data):
        """
        新增一行数据
        :param table: 数据表名
        :param data: 字典，key为字段名，value为对应字段数据
        :return:
        """
        keys = []
        values = []
        s = []
        for key, value in data.items():
            value_str = json.dumps(value, ensure_ascii=False) if isinstance(value, list) else str(value)
            keys.append("`{key}`".format(key=key))
            values.append(value_str)
            s.append('%s')

        key_sql = ','.join(keys)
        value_sql = ','.join(s)

        sql = "INSERT INTO `" + table + "` (" + key_sql + ")" + " VALUES(" + value_sql + ")"

        return sql, values

    # 新增一行数据
    def add_data(self, table, data):
        """
        新增一行数据
        :param table: 数据表名
        :param data: 字典，key为字段名，value为对应字段数据
        :return:
        """
        sql, params = self.build_add_data_sql(table, data)
        self.execute_sql(sql, params)

    # 基于ID更新一行数据的SQL
    @staticmethod
    def build_update_data_by_id_sql(table, id, data):
        """
        根据id修改指定行数据
        :param table: 数据表名，字串
        :param id: 数据主键id，数值
        :param data: 字典，key为字段名，value为对应字段数据
        :return:
        """

        sqls = []
        values = []
        for key, value in data.items():
            value_str = json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
            values.append(value_str)
            sqls.append("`{key}` = %s".format(key=key))

        update_sql = ','.join(sqls)
        sql = "UPDATE `{table}` SET {update_sql} WHERE `id` =%s".format(
            table=table,
            update_sql=update_sql,
        )
        return sql, values

    # 修改一行数据
    def edit_data_by_id(self, table, id, data):
        """
        根据id修改指定行数据
        :param table: 数据表名，字串
        :param id: 数据主键id，数值
        :param data: 字典，key为字段名，value为对应字段数据
        :return:
        """
        sql, values = self.build_update_data_by_id_sql(table, id, data)
        params = values + [str(id)]
        self.execute_sql(sql, params)

    # 删除一条指定数据
    def delete_data_by_id(self, table, id):
        """
        基于目标id删除一条指定数据
        :param table: 数据表名，字串
        :param id: 数据主键id，数值
        :return:
        """
        sql = "DELETE FROM `{table}` WHERE `id` = %s".format(table=table)
        self.execute_sql(sql, [str(id)])

    # 删除多条指定数据
    def delete_data_by_ids(self, table, ids):
        """
        基于id列表，删除多条指定数据
        :param table: 数据表名，字串
        :param ids: 数据主键id构成的列表，数值列表
        :return:
        """
        sql = "DELETE FROM `{table}` WHERE id IN ".format(table=table)
        ids_sql = "(" + ','.join(['%s' for i in ids]) + ")"
        sql += ids_sql
        ids = [str(i) for i in ids]

        self.execute_sql(sql, ids)

    def simple_query_builder(self, query):
        """
        SQL查询语句构建器
        :param query:
        字典，可解析的查询条件

        query = {
            'table': "表名",
            'field': ['查询字段名','查询字段名'],
            'order': [('排序字段', '排序模式'), ('排序字段', '排序模式')],
            'where': [('查询字段', '查询方式', '查询条件', '多个where语句间连接方式')]
        }

        例如：query = {
            'table': "sacked_official",
            'field': ['name','jobs'],
            'where': [("name", "LIKE", "'王%'", '')],
            'order': [('name', 'asc'), ('jobs', 'desc')],
        }
        :return:
        """
        field_sql = ','.join(query['field'])

        sql = "SELECT " + field_sql + " FROM `" + query['table'] + "`"

        if query['leftjoin']:
            sql += query['leftjoin']

        if query['where']:
            query['where'][-1][-1] = '' # where语句最后一个"and"或"or"去掉
            where_sql = ' WHERE '
            for field, query_mode, query_condition, join_mode in query['where']:
                where_sql += '`{field}` {query_mode} {query_condition} {join_mode}'.format(
                    field=field, query_mode=query_mode, query_condition=query_condition, join_mode=join_mode
                )
            sql += where_sql
        if query['order']:
            order_sql = " ORDER BY " + ','.join(
                ['`{field}` {order_query}'.format(field=field, order_query=order_query) for field, order_query in
                 query['order']]
            )
            sql += order_sql
        return sql

    def simple_query(self, query, params):
        """
        多条结果的简单查询
        :param query:
        字典，可解析的查询条件
        query = {
            'table': "表名",
            'field': ['查询字段名','查询字段名'],
            'order': [('排序字段', '排序模式'), ('排序字段', '排序模式')],
            'where': [('查询字段', '查询方式', '查询条件', '多个where语句间连接方式')]
        }
        :return:
        """
        sql = self.simple_query_builder(query)
        data_set = self.fetch_all(sql, params)
        return data_set

    # 浏览-支持：查询、排序、分页
    def get_page_data(self, params, page, query, page_size=10, passed_data_set=[]):
        """
        获取查询分页数据，每页数量固定为10
        :param page: 当前页码
        :param query: 查询结构体
        :return:
        """
        if query:
            data_set = self.simple_query(query, params)
            py_data_set = self.trans_multi_db_data_to_py_data(data_set)
        else:
            py_data_set = passed_data_set
        p = Paginator(py_data_set, page_size)
        try:
            page_obj = p.page(page)
            page_data = {
                "list": page_obj.object_list,
                "pageNum": page_obj.number,
                "pageSize": page_obj.paginator.per_page,
                "total": page_obj.paginator.count
            }
        except Exception as e:
            if page > 0 and type(page) == type(1):
                # 这里处理页码超过范围的情况，页码超过范围，返回list结果为空，pageNum和pageSize和正常情况下的一样，total从将页码设为1
                # 的对象中获得
                page_obj = p.page(1)
                page_data = {
                    "list": [],
                    "pageNum": page,
                    "pageSize": page_size,
                    "total": page_obj.paginator.count
                }
            else:
                raise e
        return page_data

    @staticmethod
    def is_json(goal_str):
        try:
            json.loads(goal_str)
        except:
            return False
        return True

    # 将单条查询数据整理成json可用格式
    def trans_db_data_to_py_data(self, data):
        py_data_list = []
        if data is None:#防止NoneType在for循环时报错
            return []
        for item in data:
            value = json.loads(item) if self.is_json(item) else str(item)
            py_data_list.append(value)
        return py_data_list

    # 将多条查询数据整理成json可用格式
    def trans_multi_db_data_to_py_data(self, data_set):
        py_data_list = []
        for data in data_set:
            py_data_list.append(self.trans_db_data_to_py_data(data))
        return py_data_list


if __name__ == '__main__':
    # 引入django的settings模块
    from django.conf import settings
    # 引入要加载的目标settings文件
    from NL2SQL import settings as app_settings

    # 激活settings配置
    settings.configure(default_settings=app_settings)

    from pprint import pprint

    mysql_db = Mysql()

    # query = {
    #     'table': "sacked_official",
    #     'field': ['*'],
    #     'where': [("name", "LIKE", "'王%'", '')],
    #     'order': [('name', 'asc'), ('id', 'desc')],
    # }
    # page_data = mysql_db.get_page_data(2, query)
    sql = "SELECT table_name FROM table_info where id = 1"
    table_name = mysql_db.fetch_one(sql, [])

    pprint(
        table_name
    )

