# -*- coding: utf-8 -*-


"""
   author: LinJie.Xu
   time: 2019/6/12
"""
import copy
import importlib
import json
import logging
import re

import numpy as np

from nl2sql_parser.services.template_parser.template import zh_filter
from nl2sql_parser.tools.common.tools import Tools
from nl2sql_parser.tools.common.stop_word import stop_word


class Filter:

    def __init__(self):
        self.logger = logging.getLogger("django")
        self.LOGICAL = ("EQ", "NEQ", "ELT", "EGT", "LT", "GT")
        self.delete_word = ["where", "in", "when", "which", "on", "at", "and", "of", "be"]

    def templates(self):
        """
        加载模板
        :param lang:
        :return:
        """

        # 模板热更新
        importlib.reload(zh_filter)
        from nl2sql_parser.services.template_parser.template.zh_filter import ZhFilter
        self.template = ZhFilter()

        self.gt = self.template.gt_list
        self.lt = self.template.lt_list
        self.ge = self.template.ge_list
        self.le = self.template.le_list
        self.ne = self.template.ne_list
        self.eq = self.template.eq_list
        self.and_list = self.template.and_list
        self.or_list = self.template.or_list

        self.compares = []
        self.compares.extend(self.gt)
        self.compares.extend(self.lt)
        self.compares.extend(self.ge)
        self.compares.extend(self.le)
        self.compares.extend(self.ne)
        self.compares.extend(self.eq)

        self.logical = []
        self.logical.extend(self.and_list)
        self.logical.extend(self.or_list)

        self.template = self.template.template

    def judge(self, ret, word, field_info):
        """
        判断问题中的某个词是否为 ${value} 槽，在比较和逻辑字符后面，或者被引号引起来，或者是非数字的枚举值
        :param ret:
        :param word:
        :param field_info:
        :return:
        """

        if ret and (ret[-1][:9] == "${compare" or ret[-1][:9] == "${logical") and word not in stop_word:
            return "COMMON", None
        elif len(ret) > 1 and ret[-1] in stop_word and (ret[-2][:9] == "${compare" or ret[-2][:9] == "${logical") and word not in stop_word:
            return "COMMON", None
        elif len(word) > 1 and word[0] == '"' and word[-1] == '"':
            return "COMMON", None
        for item in field_info:
            if word in item["enum"] and not Tools.is_number(word):
                return "ENUM", item["field"]
        return "OTHERS", None

    def merge_two_words(self, words):
        """
        两个相邻的词，如果拼起来以后在模板里，那就把这两个词拼起来
        :param words:
        :return:
        """
        if len(words) == 1:
            return words
        word_list = []
        word_list.extend(self.compares)
        word_list.extend(self.logical)
        new_words = []
        i = 0
        while i < len(words)-1:
            if " ".join(words[i:i + 2]) in word_list:
                new_words.append(" ".join(words[i:i + 2]))
                i += 2
            else:
                if i == len(words) - 2:
                    new_words.extend(words[i:i + 2])
                    i += 2
                else:
                    new_words.append(words[i])
                    i += 1
        return new_words

    def replace(self, words, field_info):
        """
        将问题中的词替换为对应的槽
        :param words:
        :param field_info:
        :return:
        """
        ret_map = {} # 槽对应的词的字典
        field_map = {} # 枚举值类value槽对应字段的字典
        ret = []
        num_field = 1
        num_compare = 1
        num_value = 1
        num_logical = 1
        fields = []
        for item in field_info:
            fields.append(item["field"])
        for word in words:
            if word in fields:
                item = "${{field{}}}".format(num_field)
                ret.append(item)
                ret_map[item] = word
                num_field += 1
            elif word in self.compares:
                item = "${{compare{}}}".format(num_compare)
                ret.append(item)
                ret_map[item] = word
                num_compare += 1
            elif word in self.logical:
                item = "${{logical{}}}".format(num_logical)
                ret.append(item)
                ret_map[item] = word
                num_logical += 1
            else:
                flag, field = self.judge(ret, word, field_info)
                if flag != "OTHERS":
                    item = "${{value{}}}".format(num_value)
                    ret.append(item)
                    ret_map[item] = word
                    if flag == "ENUM":
                        field_map[item] = field
                    num_value += 1
                else:
                    ret.append(word)
        ret = "".join(ret)
        return ret, ret_map, field_map

    def slots(self, ret, template, ret_map, field_map):
        """
        将结果映射和字段映射中的顺序进行重排
        :param ret:
        :param template:
        :param ret_map:
        :param field_map:
        :return:
        """
        new_ret_map = {}
        new_field_map = {}
        ret_item = re.findall(r"\${.*?}", ret)
        template_item = re.findall(r"\${.*?}", template)
        for item in zip(ret_item, template_item):
            new_ret_map[item[1]] = ret_map[item[0]]
            if item[0] in field_map:
                new_field_map[item[1]] = field_map[item[0]]
        return new_ret_map, new_field_map

    def judge_logical(self, word):
        if word in self.or_list:
            return "OR"
        elif word in self.and_list:
            return "AND"
        else:
            raise Exception("Error word {} in logical match".format(word))

    def judge_compare(self, word):
        if word in self.lt:
            return "LT"
        elif word in self.gt:
            return "GT"
        elif word in self.le:
            return "ELT"
        elif word in self.ge:
            return "EGT"
        elif word in self.ne:
            return "NEQ"
        elif word in self.eq:
            return "EQ"
        else:
            raise Exception("Error word {} in compare match.".format(word))

    def transfer(self, result, new_ret_map, new_field_map):
        """
        结果格式转化
        :param result:
        :param new_ret_map:
        :param new_field_map:
        :return:
        """

        if "logicalType" in result and result["logicalType"] not in ("AND", "OR"):
            if result["logicalType"] in new_ret_map:
                result["logicalType"] = self.judge_logical(
                    new_ret_map[result["logicalType"]])
            else:
                return False
        if result["nodeType"] == "CONDITION":
            if result["operator"] in new_ret_map:
                result["operator"] = self.judge_compare(
                    new_ret_map[result["operator"]])
            else:
                if result["operator"] not in self.LOGICAL:
                    return False
            if "field" in result:
                if result["field"] in new_ret_map:
                    result["field"] = new_ret_map[result["field"]]
                else:
                    return False
            else:
                if result["fieldValue"] in new_field_map and new_field_map[result["fieldValue"]]:
                    result["field"] = new_field_map[result["fieldValue"]]
                else:
                    return False
            if result["fieldValue"] in new_ret_map:
                result["fieldValue"] = new_ret_map[result["fieldValue"]]
            else:
                return False
            return True
        elif "child" in result:
            flag = True
            for child in result["child"]:
                flag = self.transfer(child, new_ret_map, new_field_map)
            return flag

    def delete(self, query, patterns, ret_map):
        """
        将解析到的过滤条件删除掉
        :param query:
        :param patterns:
        :param ret_map:
        :return:
        """

        for pattern in patterns:
            query = query.replace(pattern, "")
        return self.recovery(query, ret_map)

    def recovery(self, query, ret_map):
        """
        过滤条件匹配完成之后，需要把去除了过滤条件之后的问题中的槽还原成词
        :param query:
        :param ret_map:
        :return:
        """

        for key in ret_map.keys():
            query = query.replace(key, ret_map[key])
        return query

    def match(self, words, field_info):
        """
        提取过滤条件的入口，输入问题，返回过滤条件列表
        :param words:
        :param field_info:
        :param lang:
        :return:
        """

        self.templates()

        query, ret_map, field_map = self.replace(words, field_info)
        print(query)
        print(ret_map)
        results = []
        spans = []
        patterns = []
        for item in self.template:
            item_ = item["template"].replace("*", ".*?")
            item_ = re.sub(r"\${field[0-9]*}", "${field[0-9]*?}", item_)
            item_ = re.sub(r"\${compare[0-9]*}", "${compare[0-9]*?}", item_)
            item_ = re.sub(r"\${value[0-9]*}", "${value[0-9]*?}", item_)
            item_ = re.sub(r"\${logical[0-9]*}", "${logical[0-9]*?}", item_)
            item_ = item_.replace("$", r"\$")
            # rets = re.findall(item_, query)
            rets = re.finditer(item_, query)
            for ret in rets:
                sp = ret.span()
                ret = ret.group()
                if self.pos_overlap(sp, spans):
                    continue
                new_ret_map, new_field_map = self.slots(
                    ret, item["template"], ret_map, field_map)
                result = copy.deepcopy(item["result"])
                flag = self.transfer(result, new_ret_map, new_field_map)
                if not flag:
                    continue
                patterns.append(ret)
                results.append(result)
                spans.append(sp)
        results, idx = self.uniq(results)
        patterns = list(np.array(patterns)[idx])
        query = self.delete(query, patterns, ret_map)
        return results, query

    def element(self, result, set_):
        if result["nodeType"] == "CONDITION":
            set_.add(
                (result["field"],
                 result["operator"],
                 result["fieldValue"]))
        else:
            for item in result["child"]:
                self.element(item, set_)

    def judge_pass(self, item, items, idx):
        for i in range(len(items)):
            if i == idx:
                continue
            flag = False
            for j in item:
                if j not in items[i]:
                    flag = True
            if not flag:
                return False
        return True

    def uniq(self, result):
        """
        对过滤条件结果进行去重
        :param result:
        :return:
        """
        items = []
        for item in result:
            tmp = set()
            self.element(item, tmp)
            items.append(tmp)
        ret = []
        idx = []
        for i in range(len(items)):
            if self.judge_pass(items[i], items, i):
                ret.append(result[i])
                idx.append(i)
        return ret, idx

    def pos_overlap(self, pos_span, spans):
        """
        匹配到的过滤条件是否和已有的位置重叠
        :param pos_span:
        :param spans:
        :return:
        """
        for sp in spans:
            if sp[1] > pos_span[0] and sp[0] < pos_span[1]:
                return True
        return False


if __name__ == "__main__":
    filter = Filter()


    def test_unit(words, fields):
        filter.template("en_US")
        results, query = filter.match(words, fields)
        print(json.dumps(results))
        print(query)


    test_unit(["show",
               "the",
               "area",
               "and",
               "the total of",
               "age",
               "where",
               "age",
               "is",
               "18",
               "or",
               "19",
               "in",
               "BEIJING"],
              [{"field": "age",
                "enum": []},
               {"field": "single",
                "enum": []},
               {"field": "year",
                "enum": []},
               {"field": "area",
                "enum": ["BEIJING"]},
               {"field": "name",
                "enum": []}])
