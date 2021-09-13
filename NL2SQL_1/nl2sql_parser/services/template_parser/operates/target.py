# -*- coding: utf-8 -*-


"""
   author: LinJie.Xu
   time: 2019/6/12
"""

import importlib
import json
import logging
import re

from nl2sql_parser.services.template_parser.template import zh_target
from nl2sql_parser.services.template_parser.operates.match_field import MatchField
from nl2sql_parser.tools.common.tools import Tools


class Target:

    def __init__(self):
        self.logger = logging.getLogger("django")
        self.match_field = MatchField()
        self.stop_words = [" and ", ","]
        self.AGG = ["SUM", "COUNT", "DISTINCT_COUNT", "AVERAGE"]

    def classify(self, table):
        """
        将字段分为维度和指标
        :param table:
        :return:
        """

        dims = []
        indexs = []
        for item in table["fields"]:
            if item["type"] == "NUM":
                indexs.append(self.lemma(item["alias"]))
            elif item["type"] in ("TEXT", "DATE"):
                dims.append(self.lemma(item["alias"]))
        return dims, indexs

    def lemma(self, word):
        """
        提取词根
        :param word:
        :param lang:
        :return:
        """

        # if lang != "en_US":
        #     return word
        # name_n = str(self.lemmatizer(word, u"NOUN")[0])
        # name_v = str(self.lemmatizer(word, u"VERB")[0])
        # if name_n == word:
        #     return name_v
        return word

    def extract(self, words, dims, indexs, table):
        """
        从问句中提取维度和指标
        :param words:
        :param dims:
        :param indexs:
        :return:
        """

        semantic = {}
        # 提取出 query 中的维度、指标
        name = self.lemma(table["name"])
        for word in words:
            word = self.lemma(word)
            field = self.match_field.match_field(table, word, words)
            if field in dims:
                semantic[word] = {"type":"DIM", "field":field}
            elif field in indexs:
                semantic[word] = {"type":"INDEX", "field":field}
            elif word == name:
                semantic[word] = {"type":"TABLE"}
        self.logger.debug("semantic:{}".format(semantic))
        return semantic

    def merge_two_words(self, words):
        """
        两个相邻的词，如果拼起来以后在模板里，那就把这两个词拼起来
        :param words:
        :return:
        """
        if len(words) == 1:
            return words
        word_list = []
        word_list.extend(self.order)
        word_list.extend(self.agg)
        word_list.extend(self.group)
        word_list.extend(self.connect)
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

    def replace(self, words, semantic):
        """
        替换问题中的关键词
        :param words:
        :param semantic:
        :return:
        """
        new = []
        num_field = 1
        num_agg = 1
        num_order = 1
        num_num = 1
        ret_map = {}
        for item in words:
            item = self.lemma(item)
            if item in self.group:  # ${group}
                ret_map["${group}"] = item
                new.append("${group}")
            elif item in self.connect:
                ret_map["${connect}"] = item
                new.append("${connect}")
            elif item in semantic.keys():
                if semantic[item]["type"] in ("DIM", "INDEX"):  # ${field}
                    ret_map["${{field{}}}".format(num_field)] = semantic[item]["field"]
                    new.append("${{field{}}}".format(num_field))
                    num_field += 1
                else:  # ${table}
                    ret_map["${table}"] = item
                    new.append("${table}")
            elif Tools.is_number(item):  # ${num}
                ret_map["${{num{}}}".format(num_num)] = item
                new.append("${{num{}}}".format(num_num))
                num_num += 1
            elif item in self.agg:  # ${agg}
                ret_map["${{agg{}}}".format(num_agg)] = item
                new.append("${{agg{}}}".format(num_agg))
                num_agg += 1
            elif item in self.order:  # ${order}
                ret_map["${{order{}}}".format(num_order)] = item
                new.append("${{order{}}}".format(num_order))
                num_order += 1
            else:  # other
                new.append(item)

        new_sentence = "".join(new)
        # for stop_word in self.stop_words:
        #     new_sentence = new_sentence.replace(stop_word, "")
        return new_sentence, ret_map

    def match(self, sentence):
        """
        匹配模板
        :param sentence:
        :return:
        """

        matchs = []
        for template in self.templates:
            template_pattern = template["template"].replace(
                "-index", "").replace("-dim", "")
            template_pattern = template_pattern.replace(
                "*", r".*?").replace("$", r"\$")
            template_pattern = "^{}$".format(template_pattern)
            ret = re.match(template_pattern, sentence)
            if ret:
                matchs.append(template)
        return matchs

    def aggregate(self, word):
        """
        获取聚合函数
        :param word:
        :return:
        """

        if word in self.sum:
            return "SUM"
        elif word in self.avg:
            return "AVERAGE"
        elif word in self.count:
            return "COUNT"
        elif word in self.distinct_count:
            return "DISTINCT_COUNT"
        self.logger.exception("Error word {} in aggregate match".format(word))
        raise Exception("Error word {} in aggregate match".format(word))

    def get_order(self, word):
        """
        获取排序方式
        :param word:
        :return:
        """

        if word in self.max:
            return "MAX"
        elif word in self.min:
            return "MIN"
        self.logger.exception("Error word {} in order match".format(word))
        raise Exception("Error word {} in order match".format(word))

    def transfer(self, models, ret_map):
        """
        拼接结果
        :param matchs:
        :param ret_map:
        :return:
        """

        results = []
        for idx in range(len(models)):
            result = json.loads(
                json.dumps(
                    models[idx]["result"]).replace(
                    "-dim",
                    "").replace(
                    "-index",
                    ""))
            if "select" in result:
                select = result["select"]
            else:
                select = []
            if "group" in result:
                group = result["group"]
            else:
                group = []
            if "order" in result:
                order = result["order"]
            else:
                order = []
            for sel_idx in range(len(select)):
                select[sel_idx]["field"] = ret_map[select[sel_idx]["field"]]
                if "agg" in result["select"][sel_idx]:
                    if select[sel_idx]["agg"] not in self.AGG: # 比如${agg1}，这个是按模板里匹配到的聚合函数
                        agg_word = ret_map[select[sel_idx]["agg"]]
                    else: # 比如"SUM"，这是直接在模板的结果文件里指定的聚合函数
                        agg_word = select[sel_idx]["agg"]
                    try:
                        select[sel_idx]["agg"] = self.aggregate(agg_word)
                    except BaseException:
                        select[sel_idx]["agg"] = self.get_order(agg_word)
            for group_idx in range(len(group)):
                group[group_idx]["field"] = ret_map[group[group_idx]["field"]]
            if "limit" in result:
                try:
                    result["limit"] = ret_map[result["limit"]]
                except BaseException:
                    pass
            for order_idx in range(len(order)):
                order[order_idx]["field"] = ret_map[order[order_idx]["field"]]
                if "type" in order[order_idx]:
                    type = self.get_order(ret_map[order[order_idx]["type"]])
                    order[order_idx]["type"] = "DESC" if type == "MAX" else "ASC"
                if "agg" in order[order_idx]:
                    if order[order_idx]["agg"] not in self.AGG: # 比如${agg1}，这个是按模板里匹配到的聚合函数
                        agg_word = ret_map[order[order_idx]["agg"]]
                    else: # 比如"SUM"，这是直接在模板的结果文件里指定的聚合函数
                        agg_word = order[order_idx]["agg"]
                    try:
                        order[order_idx]["agg"] = self.aggregate(agg_word)
                    except BaseException:
                        order[order_idx]["agg"] = self.get_order(agg_word)
            results.append(result)
        return results

    def template(self):
        """
        读取模板
        :param lang:
        :return:
        """

        importlib.reload(zh_target)
        from nl2sql_parser.services.template_parser.template.zh_target import ZhTarget as TemplateZh
        template = TemplateZh()

        self.connect = template.connect_list
        self.group = template.group_list
        self.sum = template.sum_list
        self.avg = template.avg_list
        self.distinct_count = template.distinct_count_list
        self.count = template.count_list
        self.max = template.max_list
        self.min = template.min_list

        self.order = []
        self.order.extend(self.max)
        self.order.extend(self.min)

        self.agg = []
        self.agg.extend(self.sum)
        self.agg.extend(self.avg)
        self.agg.extend(self.distinct_count)
        self.agg.extend(self.count)

        self.templates = template.templates

    def target(self, words, table):
        """
        精确解析非过滤结果的入口
        :param words:
        :param table:
        :param lang:
        :return:
        """

        self.template()

        dims, indexs = self.classify(table)
        cols = []
        for item in table["fields"]:
            cols.append(item["alias"])
        semantic = self.extract(words, dims, indexs, table)
        # 提取出 query 中的维度、指标，返回是字典，键是词语，值是该词语匹配到的字段和字段类型
        sentence, ret_map = self.replace(words, semantic)
        print (sentence)
        self.logger.debug("target replaced sentence:{}".format(sentence))
        models = self.match(sentence)
        results = self.transfer(models, ret_map)
        # print(json.dumps(results, ensure_ascii=False))
        return results


if __name__ == "__main__":
    def test_unit(q):
        target1 = Target("en_US")
        print("*" * 10)
        print("sentence:{}".format(q))
        print("result:{}".format(target1.target(q, table_)))
        print("$" * 10)


    table_ = {"name": "singer", "fields": [
        {"alias": "singer_id", "type": "TEXT", "name": "singer_id"},
        {"alias": "name", "type": "TEXT", "name": "name"},
        {"alias": "country", "type": "TEXT", "name": "country"},
        {"alias": "song_name", "type": "TEXT", "name": "song_name"},
        {"alias": "year", "type": "TEXT", "name": "song_release_year"},
        {"alias": "age", "type": "NUM", "name": "age"},
        {"alias": "is_male", "type": "NUM", "name": "is_male"}]}

    test_unit(["What", "is", "the", "average", ",", "and",
               "sum", "age", "of", "all", "singers"])
    test_unit(
        "Show the name and the release year of the song by the youngest age".split())
    test_unit("Show the name of the song by the youngest age".split())
    test_unit("what is the most age of singer".split())
    test_unit("age for each country".split())
