# -*- coding: utf-8 -*-

import logging

from nl2sql_parser.services.template_parser.template.zh_filter import ZhFilter as ZhTemplateFilter
from nl2sql_parser.services.template_parser.template.zh_target import ZhTarget as ZhTemplateTarget
from nl2sql_parser.services.template_parser.operates.time_operate import TimeOperate
from nl2sql_parser.tools.nlp.segment import Segment
from nl2sql_parser.services.template_parser.operates.filter import Filter
from nl2sql_parser.services.template_parser.operates.target import Target

class TemplateParser(object):
    """
    基于模板的解析器
    """

    def __init__(self):
        self.logger = logging.getLogger('develop')
        self.zh_filter = ZhTemplateFilter()
        self.zh_target = ZhTemplateTarget()
        self.time_operate = TimeOperate()
        self.segment = Segment()
        self.filter = Filter()
        self.target = Target()
        self.chars = ["/", ".", "。", ",", "，", ";", "?", "；", "？"]

    def keyword(self):
        def add(filter, target):
            ret = []
            ret.extend(filter.gt_list)
            ret.extend(filter.lt_list)
            ret.extend(filter.ge_list)
            ret.extend(filter.le_list)
            ret.extend(filter.ne_list)
            ret.extend(filter.eq_list)
            ret.extend(filter.and_list)
            ret.extend(filter.or_list)
            ret.extend(target.group_list)
            ret.extend(target.sum_list)
            ret.extend(target.avg_list)
            ret.extend(target.distinct_count_list)
            ret.extend(target.count_list)
            ret.extend(target.max_list)
            ret.extend(target.min_list)
            return ret
        return add(self.zh_filter, self.zh_target)

    def time_field(self, table_info):
        """
        获取时间字段 id
        :param table:
        :return:
        """
        for field in table_info["fields"]:
            if field["type"] == "DATE":
                return field["id"]
        return None

    def filter_input(self, table_info):
        """
        拼接过滤条件解析的输入
        :param table_info:
        :return:
        """
        result = []
        for item in table_info["fields"]:
            result.append(
                {"field": item["alias"], "enum": item.get("dataEnum", [])})
        return result

    def target_input(self, table_info):
        """
        拼接非过滤精确匹配的输入
        :param table_info:
        :return:
        """
        result = {"name": table_info["name"], "fields": []}
        for item in table_info["fields"]:
            result["fields"].append(
                {"alias": item["alias"], "type": item["type"], "name": item["name"]})
        return result

    def order(self, target):
        """
        对非过滤解析结果打分
        :param target:
        :return:
        """

        def get_score(item):
            score = 0
            if "select" not in item or not item["select"]:
                return 0
            for sel in item["select"]:
                if "agg" in sel and sel["agg"]:
                    score += 1
                if "field" in sel and sel["field"]:
                    score += 1
            if "group" in item:
                for gro in item["group"]:
                    if "field" in gro:
                        score += 1
                    if "func" in gro:
                        score += 1
            if "limit" in item:
                score += 1
            return score

        return sorted(target, key=lambda x: -get_score(x))

    def add_info_filter(self, table_id, alias_type_map, alias_id_map, filter):
        if "nodeType" in filter and filter["nodeType"] == "CONDITION":
            # filter["table"] = table_id
            filter["fieldId"] = alias_id_map[filter["field"]]
            # filter["fieldType"] = alias_type_map[filter["field"]]
            filter["operate"] = filter["operator"]
            del filter["operator"]
            del filter["field"]
            del filter["valueType"]
            
        elif filter["nodeType"] == "GROUP":
            for item in filter["child"]:
                self.add_info_filter(
                    table_id, alias_type_map, alias_id_map, item)

    def combine(self, target, filter, time_result_filter, time_result_group, table_info):
        """
        合并过滤条件和非过滤条件
        :param target:
        :param filter:
        :param time_filter:
        :param table_info:
        :return:
        """
        alias_id_map = {}
        alias_type_map = {}
        id_type_map = {}
        for item in table_info["fields"]:
            alias_id_map[self.target.lemma(item["alias"])] = item["id"]
            alias_id_map[item["alias"]] = item["id"]
            alias_type_map[self.target.lemma(item["alias"])] = item["type"]
            alias_type_map[item["alias"]] = item["type"]
            id_type_map[item["id"]] = item["type"]
        result = {}

        # 没有识别出 select, 作为无法识别的标志
        if len(target) == 0 or len(target[0]["select"]) == 0:
            return result, id_type_map

        group = []
        select = []

        if filter:
            tmp_filter = {
                "logicalType": "AND",
                "nodeType": "GROUP",
                "child": filter} if len(filter) > 1 else filter[0]
            self.add_info_filter(
                table_info["id"],
                alias_type_map,
                alias_id_map,
                tmp_filter)
            result["filter"] = tmp_filter

        # 补充上关于时间的过滤条件
        if time_result_filter:
            if "filter" in result:
                result["filter"]["child"].append(time_result_filter)
            else:
                result["filter"] = time_result_filter

        if time_result_group:
            time_result_group["table"] = table_info["id"]
            group.append(time_result_group)

        order_map = {}
        if "order" in target[0]:
            for item in target[0]["order"]:
                order_map[alias_id_map[item["field"]]
                          ] = item.get("type", "ASC")

        if "group" in target[0]:
            for item in target[0]["group"]:
                group_item = {
                    "table": table_info["id"], "fieldId": alias_id_map[item["field"]]}
                if group_item["fieldId"] in order_map:
                    group_item["order"] = order_map[group_item["fieldId"]]
                group.append(group_item)

        for item in target[0]["select"]:
            select_item = {"aggregate": item.get("agg", None),
                           "fieldId": alias_id_map[item["field"]]}
            # if select_item["fieldId"] in order_map:
            #     select_item["order"] = order_map[select_item["fieldId"]]
            select.append(select_item)

        result["group"] = group
        result["select"] = select

        if "limit" in target[0] and target[0]["limit"]:
            result["limit"] = target[0]["limit"]

        return result, id_type_map

    def transfer_format(self, result):
        """
        统一输出格式
        :return:
        """
        new_result = {'select': [], 'where': {}, 'group': [], 'order': [], 'limit': None, 'having': {}}
        if 'select' in result:
            new_result['select'] = result['select']
        if 'filter' in result:
            new_result['where'] = result['filter']
        if 'group' in result:
            new_result['group'] = result['group']
        if 'order' in result:
            new_result['order'].append(result['order'])
        if 'limit' in result:
            new_result['limit'] = result['limit']

        return new_result

    def parse(self, table_info, query):
        """

        :param table_id:
        :param question:
        :return:
        """
        # 字段别名，保留词, 枚举值作为合词列表
        fields = []
        enums = []
        for item in table_info["fields"]:
            fields.append(item["alias"])
            if "dataEnum" in item and item["dataEnum"]:
                for enum in item["dataEnum"]:
                    enums.append(enum)
        keywords = fields
        keywords.extend(self.keyword())
        keywords.extend(enums)
        keywords = list(set(keywords))

        # 分词、合词
        self.logger.info("keywords:{}".format(keywords))
        self.segment.update_index(keywords)

        # 处理特殊符号，防止分词时出错
        for item in self.chars:
            query = query.replace(item, " {} ".format(item))

        words = self.segment.segment(query)
        self.logger.debug("words after segment:{}".format(words))

        # 提取时间
        id = self.time_field(table_info)
        if id:
            standard_result = self.time_operate.standard(query)
            try:
                time_result_filter = self.time_operate.to_filter(
                    standard_result[0], id)
                time_result_group = self.time_operate.to_filter(
                    standard_result[1], id)
            except BaseException as e:
                time_result_filter = []
                time_result_group = []
                self.logger.info(repr(e))
            self.logger.debug("time_result_filter ret:{}".format(time_result_filter))
            self.logger.debug("time_result_group ret:{}".format(time_result_group))
            words = self.segment.segment(standard_result[2])
        else:
            time_result_filter = None
            time_result_group = None

        # 匹配过滤条件
        fields = self.filter_input(table_info)
        tmp_words = []
        for item in words:
            tmp_words.append("".join(item))
        self.logger.debug("filter input:{}".format(tmp_words))
        filter, query = self.filter.match(tmp_words, fields)
        self.logger.debug("query after filter:{}".format(query))
        self.logger.debug("filter ret:{}".format(filter))

        # 精确匹配非过滤条件
        target_table = self.target_input(table_info)
        words = self.segment.segment(query)
        tmp_words = []
        for item in words:
            tmp_words.append("".join(item))
        self.logger.debug("target input:{}".format(tmp_words))
        target = self.target.target(tmp_words, target_table)
        target = self.order(target)
        self.logger.debug("target ret:{}".format(target))
        if not target or "select" not in target[0] or not target[0]["select"]:
            target = []

        print(target, filter, time_result_filter, time_result_group, table_info)
        # 和不同的系统对接时，对时间类条件的格式可能不同，因此，这里先不合并时间类的结果
        result_without_time, id_type_map = self.combine(target, filter, [], [], table_info)
        result_without_time = self.transfer_format(result_without_time)
        
        return result_without_time, time_result_filter, time_result_group

if __name__ == '__main__':

    from django.conf import settings
    import NL2SQL.settings as app_setting
    settings.configure(default_settings=app_setting)

    tp = TemplateParser()

    query = '2019年9月最大云量'
    # query = 'PMS云量'
    table_id = 1
    result = tp.parse(table_id, query)

    print('result', result)