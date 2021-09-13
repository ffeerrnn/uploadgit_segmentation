# -*- coding: utf-8 -*-

import logging
import jieba

from django.core.cache import cache

from nl2sql_parser.tools.inverted_index.build_index import build_index
from nl2sql_parser.services.template_parser.template.zh_filter import ZhFilter as ZhTemplateFilter
from nl2sql_parser.services.template_parser.template.zh_target import ZhTarget as ZhTemplateTarget
from nl2sql_parser.tools.common.stop_word import stop_word
from nl2sql_parser.tools.common.tools import Tools
from nl2sql_parser.tools.data_manage.table_info_loader import TableInfoLoader


class MatchTable(object):

    def __init__(self):
        self.logger = logging.getLogger("django")
        self.zh_filter = ZhTemplateFilter()
        self.zh_target = ZhTemplateTarget()
        self.tools = Tools()
        self.table_info_loader = TableInfoLoader()

    def _character(self, word):
        characters = list(word)
        return characters

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

    def time_in_field(self, word, index):
        words = list(word)

        for w in words:
            try:
                value = index[w]
            except:
                return True
        return False

    def match_table(self, query):

        words = list(jieba.cut(query))

        keyword = self.keyword()
        words_filter = list(filter(lambda w: w not in stop_word and w not in keyword and w != " ", words))

        # 字段匹配
        count = {}
        for word in words_filter:
            word = word.strip('s') if len(word) > 4 else word
            charaters = self._character(word)

            for charater in charaters:
                try:
                    # if self.tools.is_number(charater):
                    #     continue
                    table_ids = build_index.inverted_index[charater]
                    for table_id in table_ids:
                        count.setdefault(table_id, 0)
                        count[table_id] += 1
                except:
                    pass

        # 时间匹配
        has_time = False
        bfd_pos = self.tools.sw(query)
        for word, pos in bfd_pos:
            if pos == "t":
                has_time = True
                time_word = word
                break

        if has_time and self.time_in_field(time_word, build_index.inverted_index):
            try:
                table_ids = build_index.inverted_index["${DATE}"]
                for table_id in table_ids:
                    count.setdefault(table_id, 0)
                    count[table_id] += 50
            except:
                pass

        return count, words_filter

    def best_table(self, ids, words):
        """
        获取最好的一张表
        :param ids:
        :return:
        """

        def distince(alias):
            dis = 0
            for word in words:
                last = -1
                beg = 0
                for ch in word:
                    pos = alias.find(ch, beg)
                    if pos != -1:
                        if last == -1:
                            last = pos
                        beg = pos
                        dis += abs(pos - last)
            return dis

        res = []
        for id in ids:
            table_meta = read_data.query_meta_basic(id)
            alias_str = ""
            for field in table_meta["fields"]:
                alias_str = "{}_____{}".format(alias_str, field["alias"].replace(" ", ""))
            res.append((id, distince(alias_str)))
        res = sorted(res, key=lambda x: x[1])
        res = list(map(lambda x: x[0], res))
        return res

    def match(self, query):
        """
        找表的入口
        :param self:
        :param query:
        :param user_id:
        :return:
        """

        # table_ids = read_data.query_selected_tables(user_id)
        res, words_filter = self.match_table(query)
        # res = list(filter(lambda x: x[0] in table_ids, sorted(res.items(), key=lambda x: -x[1])))
        res = sorted(res.items(), key=lambda x: -x[1])

        result = []
        last_score = -1
        for item in res[:10]:
            if item[1] < last_score:
                break
            result.append(item[0])
            last_score = item[1]

        self.logger.info("match result:{}".format(res[:10]))

        # if len(result) > 1:
        #     result = self.best_table(result, words_filter)
        self.logger.info("match res:{}".format(result))

        return result

    def match_aogeo(self, query):
        """
        地球观测项目的找表函数，主要通过表名匹配
        """
        keyword_processor = self.table_info_loader.table_name_keywordprocessor()
        match_results = keyword_processor.extract_keywords_nlp(query, span_info=True)
        print(match_results)
        match_results = sorted(match_results, key=lambda x: len(x[0]), reverse=True)
 
        table_name_id_dict = self.table_info_loader.get_cache('table_name_id_dict')
        
        all_table_ids = self.table_info_loader.get_table_ids()

        if match_results:
            table_ids = table_name_id_dict[match_results[0][0]]
            return table_ids, True            
        else:
            return all_table_ids, False


if __name__ == "__main__":

    from django.conf import settings
    import NL2SQL.settings as app_setting
    settings.configure(default_settings=app_setting)

    build_index.init()
    print(build_index.inverted_index.keys())
    match = MatchTable()

    import time

    s = time.time()
    # print(match.match("开放日期", "1"))
    print (match.match("高分2019年9月最大云量"))
    # print(match.match("各年份人口", "1"))
    # print(match.match("order_time", "1"))
    # print(match.match("飞驰人生的票房", "1"))
    # print(match.match("豆瓣电影票房的样例数据", "1"))
    print(time.time() - s)
