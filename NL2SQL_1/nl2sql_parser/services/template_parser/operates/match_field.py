# -*- coding: utf-8 -*-


"""
   author: LinJie.Xu
   time: 2019/6/12
"""

import re

from nl2sql_parser.tools.common.tools import Tools

class MatchField(object):
    def __init__(self):
        self.pos = None

    def ner_match(self, word, datas, identify):
        """
        从预览数据中根据命名实体识别匹配字段,比如匹配地址，当
        预览数据中一般以上的数据都是地址，那就算匹配上了,当有
        多个都能匹配上时取第一个，但是会给出提示信息。
        :param word: 当前的需要识别的词
        :param datas: 预览数据
        :param identify: word 的词性，可能为 ns, nh 和 ni
        :return: 匹配到的字段，可能为 None
        """
        tmp = {}
        if len(datas) == 0:
            return None
        for item in list(datas[0].keys()):
            tmp[item] = 0
        for item in datas:
            for item_key in list(item.keys()):
                try:
                    tmp_words = Tools.segment(item[item_key])
                    pos = Tools.postag(tmp_words)
                except BaseException:
                    continue
                flag = True
                if Tools.is_empty(pos):
                    flag = False
                else:
                    for item_pos in pos:
                        if item_pos != identify:
                            flag = False
                            break
                if flag:
                    tmp[item_key] += 1
        length = len(datas)
        ret = None
        for item in list(tmp.keys()):
            if tmp[item] > (length / 2):
                if not ret:
                    ret = item
        return ret

    def match_enum(self, db_info, word):
        """
        通过枚举值匹配值和字段
        """
        for item in db_info["fields"]:
            if "dataEnum" in item and item["dataEnum"]:
                for v in item["dataEnum"]:
                    if word == v:
                        return item["alias"]
        return None

    def match(self, word, table_info):
        """
        对于某个单词，要确定它的数据库语义
        """

        # 字段名完全匹配
        for item in table_info["fields"]:
            if word == item["name"] or word == item["alias"]:
                return item["alias"]

        # 枚举值匹配
        field_name = self.match_enum(table_info, word)
        if field_name:
            return field_name

        # 匹配描述（描述用“,”分词）
        for item in table_info["fields"]:
            if "desc" in item:
                desc_split = re.split(r'[,，]', item["desc"].strip())
                for desc_word in desc_split:
                    if word == desc_word:
                        return item["alias"]

        # 下面的规则都是模糊匹配
        # 字段名部分匹配，取匹配上的最短字段
        tmp_fields = []
        word = word.strip('s')
        for item in table_info["fields"]:
            if (min(len(item["alias"]), len(word)) / max(len(word), len(item["alias"])) >= 0.3) and \
                    ((len(word) > 1) or len(word) > 2) and \
                    (item["alias"].find(word) != -1 or word.find(item["alias"]) != -1):
            # if ((len(word) > 1 and lang == LanguageType.ZH_CN) or len(word) > 2 or pos[0] == "n") and \
            #         (item["alias"].find(word) != -1 or word.find(item["alias"]) != -1) :
                tmp_fields.append(item["alias"])
        if tmp_fields:
            return min(tmp_fields, key=len)

        # # 通过命名实体识别，如果预览数据中的字段有一半以上是时间，就用那个字段
        # field_name = self.match_field_from_view_data_with_pos(
        #     word, db_info_json["data"], "nt")
        # if field_name:
        #     return ("value", "tablename", field_name, word)

        # # 用 word2vec 进行模糊匹配, word 不能为纯数字
        # if not Tools.is_number(word):
        #     ret = get_similarity(word2vec_model, word, cols)
        #     if ret:
        #         Tool.LOGGER.debug(word + "通过 word2vec 找到 " + ret)
        #         return ("attribute", "tableName", ret, word)

        return None

    def match_field(self, table_info, word, words, keyword=[]):
        """
        获取每个词的数据库语义
        """
        pos_list = ["n"]*len(words)

        if word in keyword:
            return None
        else:
            pos = pos_list[words.index(word)]
            return self.match(word, table_info)


if __name__ == "__main__":
    match_field = MatchField()
    print(match_field.match_field({"fields":[{"alias":"男性人口", "name":"dd"}]}, "人", ["有", "多少", "人"], "zh_CN"))
