import copy
import datetime
import re
import importlib
import time
from dateutil.relativedelta import relativedelta

import jieba

from nl2sql_parser.tools.common.tools import Tools
from nl2sql_parser.services.template_parser.template.time import Time

class TimeOperate(object):

    def __init__(self):
        self.time = Time()
        self.month_list = self.time.month_list
        self.constant = [
            "NOW_YEAR",
            "NOW_MONTH",
            "NOW_DAY",
            "NOW_HOUR",
            "NOW_MINUTE",
            "NOW_SECOND"]
        # 支持的槽：${month}, ${num}
        # 支持的常量：NOW_YEAR, NOW_MONTH, NOW_DAY, NOW_HOUR, NOW_MINUTE, NOW_SECOND
        self.month_map = {
            "jan.": 1,
            "january": 1,
            "feb.": 2,
            "february": 2,
            "mar.": 3,
            "march": 3,
            "apr.": 4,
            "april": 4,
            "may.": 5,
            "may": 5,
            "jun.": 6,
            "june": 6,
            "jul.": 7,
            "july": 7,
            "aug.": 8,
            "august": 8,
            "sept.": 9,
            "september": 9,
            "oct.": 10,
            "october": 10,
            "nov.": 11,
            "november": 11,
            "dec.": 12,
            "december": 12,
            "janvier": 1,
            "fev.": 2,
            "février": 2,
            "mars.": 3,
            "mars": 3,
            "avr.": 4,
            "avril": 4,
            "mai.": 5,
            "mai": 5,
            "juin.": 6,
            "juin": 6,
            "juillet.": 7,
            "juillet": 7,
            "aout.": 8,
            "août": 8,
            "septembre": 9,
            "octobre": 10,
            "novembre": 11,
            "décembre": 12
        }

        # 匹配“零一二三四五六七八九十千万亿”
        self.chinese_number = re.compile("[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u5343\u96f6\u4e07\u4ebf]+")

    def slots(self, ret, template, ret_map):
        """
        将结果映射和字段映射中的顺序进行重排
        :param ret:
        :param template:
        :param ret_map:
        :param field_map:
        :return:
        """
        new_ret_map = {}
        ret_item = re.findall(r"\${.*?}", ret)
        template_item = re.findall(r"\${.*?}", template)
        for item in zip(ret_item, template_item):
            new_ret_map[item[1]] = ret_map[item[0]]
        return new_ret_map

    def load_template(self):
        self.time = Time()
        self.group = self.time.group
        self.templates = self.time.templates

    def has_time(self, query):
        """
        判断某个问题中是否存在时间
        :param words:
        :param lang:
        :return:
        """

        self.load_template()
        query_copy = query
        # 必须先转化成阿拉伯数字再分词，否则会分错
        query = self.transfer_number(query)
        words = list(jieba.cut(query))

        query, ret_map = self.replace(words)

        for template in self.templates:
            pattern = re.sub(
                r"\${num[0-9]*}",
                "${num[0-9]*?}",
                template["template"])
            pattern = pattern.replace("$", r"\$")
            rets = re.findall(pattern, query)
            if rets:
                return True, rets[0]
        return False, None

    def transfer_number(self, query):
        """
        把句子中的所有中文数字转化成阿拉伯数字
        :param query:
        :return:
        """

        while (True):
            print(query)
            res = self.chinese_number.search(query)
            if res:
                idx = res.span()
                number = Tools.chinese_to_arabic(query[idx[0]:idx[1]])
                query = "{}{}{}".format(query[:idx[0]], number, query[idx[1]:])
            else:
                return query

    def standard(self, query):

        self.load_template()

        # 必须先转化成阿拉伯数字再分词，否则会分错
        query = self.transfer_number(query)
        words = list(jieba.cut(query))

        query, ret_map = self.replace(words)
        results = []
        patterns = []

        for template in self.templates:
            pattern = re.sub(
                r"\${num[0-9]*}",
                "${num[0-9]*?}",
                template["template"])
            pattern = pattern.replace("$", r"\$")
            rets = re.findall(pattern, query)
            for item in rets:
                new_ret_map = self.slots(item, template["template"], ret_map)
                result = self.transfer(new_ret_map, template["result"])
                results.append(result)
                patterns.append(item)
        query = self.delete(query, patterns, ret_map)
        results = self.sort(results)
        if results:
            filter_tmp = None
            group_tmp = None
            for item in results:
                if "group" not in item[1]:
                    filter_tmp = item[1]
                    break
            for item in results:
                if "group" in item[1]:
                    group_tmp = item[1]
                    break
            return filter_tmp, group_tmp, query
        else:
            return None, None, query

    def to_filter(self, result, field_id):

        if not result:
            return None

        def value(item):
            now_year = datetime.datetime.now().year
            now_month = datetime.datetime.now().month
            now_day = datetime.datetime.now().day

            year = now_year
            if "year" in item:
                if item["year"].find("NOW_YEAR") == -1:
                    year = int(item["year"])
                else:
                    year_str = str(
                        item["year"]).replace(
                        "NOW_YEAR",
                        str(now_year))
                    try:
                        year = eval(year_str)
                    except BaseException:
                        year = now_year
            month = 1
            if "month" in item:
                if item["month"].find("NOW_MONTH") == -1:
                    if item["month"].lower() in self.month_map:
                        month = int(self.month_map[item["month"].lower()])
                    else:
                        month = int(item["month"])
                else:
                    time_str = str(
                        item["month"]).replace(
                        "NOW_MONTH",
                        str(now_month))
                    try:
                        month = eval(time_str)
                    except BaseException:
                        month = now_month
            day = 1
            if "day" in item:
                if item["day"].find("NOW_DAY") == -1:
                    day = int(item["day"])
                else:
                    day_str = str(item["day"]).replace("NOW_DAY", str(now_day))
                    try:
                        day = eval(day_str)
                    except BaseException:
                        day = now_day

            # 处理 day、month 为负数的情况，比如问“近100天”
            if day <= 0 or month <= 0:
                if month <= 0:
                    year -= int((-month + 12) / 12)
                    month = 12 - (-month) % 12
                if day <= 0:
                    time_str = "{}-{}-{} 00:00:00".format(year, month, 1)
                    time_stamp = time.mktime(time.strptime(time_str, "%Y-%m-%d %H:%M:%S"))
                    time_array = time.localtime(time_stamp - (-day + 1) * 24 * 3600)
                    year, month, day = time_array.tm_year, time_array.tm_mon, time_array.tm_mday

            return "{}-{}-{} 00:00:00".format(year, '{:02d}'.format(month), '{:02d}'.format(day))

        results = []
        if "group" in result:
            if not result["group"]:
                return None
            return {
                "fieldId": field_id,
                "func": result["group"].get(
                    "func",
                    None)}
        elif "value" in result:
            for item in result["value"]:
                func = item.get("func", None)
                results.append({"fieldId": field_id,
                                "operate": "EQ",
                                # "fieldValue": Tools.str2timestamp(value(item),
                                #                                   func),
                                "fieldValue": value(item),
                                "func": func,
                                "nodeType": "CONDITION"})
            if not results:
                return None
            return {"logicalType": "OR", "child": results, "nodeType": "GROUP"} if len(results) > 1 else results[0]
        else:
            if "start" in result:
                func = result["start"].get("func", None)
                # results.append({"fieldId": field_id, "operate": "EGT", "fieldValue": Tools.str2timestamp(
                    # value(result["start"]), func), "func": func, "nodeType": "CONDITION"})
                results.append({"fieldId": field_id, "operate": "GT", "fieldValue":
                    value(result["start"]), "func": func, "nodeType": "CONDITION"})
            if "end" in result:
                func = result["end"].get("func", None)
                # results.append({"fieldId": field_id, "operate": "ELT", "fieldValue": Tools.str2timestamp(
                #     value(result["end"]), func), "func": func, "nodeType": "CONDITION"})
                results.append({"fieldId": field_id, "operate": "LT", "fieldValue": 
                    value(result["end"]), "func": func, "nodeType": "CONDITION"})
            if not results:
                return None
            return {"logicalType": "AND", "child": results, "nodeType": "GROUP"} if len(results) > 1 else results[0]

    def sort(self, results):

        def add_score(item):
            score = 0
            if "year" in item and item["year"] not in self.constant:
                score += 1
            if "month" in item and item["month"] not in self.constant:
                score += 1
            if "day" in item and item["day"] not in self.constant:
                score += 1
            if "hour" in item and item["hour"] not in self.constant:
                score += 1
            if "minute" in item and item["minute"] not in self.constant:
                score += 1
            if "second" in item and item["second"] not in self.constant:
                score += 1
            return score
        tmp = []
        for result in results:
            score = 0
            if "value" in result:
                for item in result["value"]:
                    score += add_score(item)
            else:
                if "start" in result:
                    score += 1
                    score += add_score(result["start"])
                if "end" in result:
                    score += 1
                    score += add_score(result["end"])
            tmp.append((score, result))
        results = sorted(tmp, key=lambda x: -x[0])
        return results

    def transfer(self, ret_map, result):
        def data(ret):

            if "year" in ret and ret["year"] in ret_map:
                ret["year"] = ret_map[ret["year"]]
            elif "year" in ret:
                items = re.findall(r"\${.*?}", ret["year"])
                for item in items:
                    if item in ret_map:
                        ret["year"] = re.sub(
                            r"\${num[0-9]*}", ret_map[item], ret["year"])

            if "month" in ret and ret["month"] in ret_map:
                tmp = ret["month"]
                ret["month"] = ret_map[ret["month"]]
                if ret["month"].find("$") != -1:
                    ret["month"] = re.sub(r"\${num[0-9]*}", tmp, ret["month"])
            elif "month" in ret:
                items = re.findall(r"\${.*?}", ret["month"])
                for item in items:
                    if item in ret_map:
                        ret["month"] = re.sub(
                            r"\${num[0-9]*}", ret_map[item], ret["month"])

            if "day" in ret and ret["day"] in ret_map:
                tmp = ret["month"]
                ret["day"] = ret_map[ret["day"]]
                if ret["day"].find("$") != -1:
                    ret["day"] = re.sub(r"\${num[0-9]*}", tmp, ret["day"])
            elif "day" in ret:
                items = re.findall(r"\${.*?}", ret["day"])
                for item in items:
                    if item in ret_map:
                        ret["day"] = re.sub(
                            r"\${num[0-9]*}", ret_map[item], ret["day"])

            if "hour" in ret and ret["hour"] in ret_map:
                ret["hour"] = ret_map[ret["hour"]]
            if "minute" in ret and ret["minute"] in ret_map:
                ret["minute"] = ret_map[ret["minute"]]
            if "second" in ret and ret["second"] in ret_map:
                ret["second"] = ret_map[ret["second"]]
        result_copy = copy.deepcopy(result)
        if "value" in result_copy:
            for idx in range(len(result_copy["value"])):
                data(result_copy["value"][idx])
        if "start" in result:
            data(result_copy["start"])
        if "end" in result:
            data(result_copy["end"])
        return result_copy

    def replace(self, words):
        """
        将问题中的词替换为对应的槽
        :param words:
        :param field_info:
        :return: 返回的是问题中的词语替换成槽后的问题，和槽与词语对应字典
        """

        ret_map = {}
        ret = []
        num_number = 1
        num_month = 1
        for word in words:
            if word in self.group:
                ret.append("${group}")
                ret_map["${group}"] = "${group}"
            elif Tools.is_number(word):
                item = "${{num{}}}".format(num_number)
                ret.append(item)
                ret_map[item] = word
                num_number += 1
            elif word.lower() in self.month_list:
                item = "${{month{}}}".format(num_month)
                ret.append(item)
                ret_map[item] = word
                num_month += 1
            else:
                ret.append(word)
        ret = "".join(ret)
        return ret, ret_map

    def delete(self, query, patterns, ret_map):
        """
        将解析到的时间删除掉
        :param query:
        :param patterns:
        :param ret_map:
        :return:
        """
        patterns = sorted(patterns, key=lambda x: -len(x))
        for pattern in patterns:
            query = query.replace(pattern, "")
        return self.recovery(query, ret_map)

    def recovery(self, query, ret_map):
        """
        时间匹配完成之后，需要把去除了时间之后的问题中的槽还原成词
        :param query:
        :param ret_map:
        :return:
        """

        for key in ret_map.keys():
            query = query.replace(key, ret_map[key])
        return query

    def trans_time_word(self, text):
        """
        把问题中的时间词与替换成时间表示，如今年换成2020年等
        """
        now_time = datetime.datetime.now()
        year = now_time.year
        month = now_time.month
        day = now_time.day
        weekday = now_time.weekday()

        last_year = year - 1
        before_last_year = year - 2

        last_month_time = now_time - relativedelta(months=1) 
        before_last_month_time = now_time - relativedelta(months=2)

        last_day_time = now_time - relativedelta(days=1) 
        before_last_day_time = now_time - relativedelta(days=2)

        Mon = now_time - relativedelta(days=weekday) # 本周一的日期
        Tue = Mon + relativedelta(days=1)
        Wed = Mon + relativedelta(days=2)
        Thur = Mon + relativedelta(days=3)
        Fri = Mon + relativedelta(days=4)
        Sat = Mon + relativedelta(days=5)
        Sun = Mon + relativedelta(days=6)

        la_Mon = Mon - relativedelta(days=7) # 上周一的日期
        la_Tue = la_Mon + relativedelta(days=1)
        la_Wed = la_Mon + relativedelta(days=2)
        la_Thur = la_Mon + relativedelta(days=3)
        la_Fri = la_Mon + relativedelta(days=4)
        la_Sat = la_Mon + relativedelta(days=5)
        la_Sun = la_Mon + relativedelta(days=6)

        next_Mon = Mon + relativedelta(days=7) # 下周一的日期
        
        replace_dict = {}
        replace_dict['今年'] = '{}年'.format(str(year))
        replace_dict['去年'] = '{}年'.format(str(last_year))
        replace_dict['前年'] = '{}年'.format(str(before_last_year))

        replace_dict['本月'] = '{year}年{mon}月'.format(year=str(year), mon=str(month))
        replace_dict['上个月'] = '{year}年{mon}月'.format(year=str(last_month_time.year), mon=str(last_month_time.month))
        replace_dict['上月'] = '{year}年{mon}月'.format(year=str(last_month_time.year), mon=str(last_month_time.month))
        
        replace_dict['今天'] = '{year}年{mon}月{day}日'.format(year=str(year), mon=str(month), day=str(day))
        replace_dict['昨天'] = '{year}年{mon}月{day}日'.format(year=str(last_day_time.year), mon=str(last_day_time.month), day=str(last_day_time.day))
        replace_dict['前天'] = '{year}年{mon}月{day}日'.format(year=str(before_last_day_time.year), mon=str(before_last_day_time.month), day=str(before_last_day_time.day))

        replace_dict['上周一'] = '{year}年{mon}月{day}日'.format(year=str(la_Mon.year), mon=str(la_Mon.month), day=str(la_Mon.day))
        replace_dict['上周二'] = '{year}年{mon}月{day}日'.format(year=str(la_Tue.year), mon=str(la_Tue.month), day=str(la_Tue.day))
        replace_dict['上周三'] = '{year}年{mon}月{day}日'.format(year=str(la_Wed.year), mon=str(la_Wed.month), day=str(la_Wed.day))
        replace_dict['上周四'] = '{year}年{mon}月{day}日'.format(year=str(la_Thur.year), mon=str(la_Thur.month), day=str(la_Thur.day))
        replace_dict['上周五'] = '{year}年{mon}月{day}日'.format(year=str(la_Fri.year), mon=str(la_Fri.month), day=str(la_Fri.day))
        replace_dict['上周六'] = '{year}年{mon}月{day}日'.format(year=str(la_Sat.year), mon=str(la_Sat.month), day=str(la_Sat.day))
        replace_dict['上周日'] = '{year}年{mon}月{day}日'.format(year=str(la_Sun.year), mon=str(la_Sun.month), day=str(la_Sun.day))
        replace_dict['上周'] = '{year1}年{mon1}月{day1}日到{year2}年{mon2}月{day2}日'.format(
            year1=str(la_Mon.year), mon1=str(la_Mon.month), day1=str(la_Mon.day), 
            year2=str(Mon.year), mon2=str(Mon.month), day2=str(Mon.day))

        replace_dict['周一'] = '{year}年{mon}月{day}日'.format(year=str(Mon.year), mon=str(Mon.month), day=str(Mon.day))
        replace_dict['周二'] = '{year}年{mon}月{day}日'.format(year=str(Tue.year), mon=str(Tue.month), day=str(Tue.day))
        replace_dict['周三'] = '{year}年{mon}月{day}日'.format(year=str(Wed.year), mon=str(Wed.month), day=str(Wed.day))
        replace_dict['周四'] = '{year}年{mon}月{day}日'.format(year=str(Thur.year), mon=str(Thur.month), day=str(Thur.day))
        replace_dict['周五'] = '{year}年{mon}月{day}日'.format(year=str(Fri.year), mon=str(Fri.month), day=str(Fri.day))
        replace_dict['周六'] = '{year}年{mon}月{day}日'.format(year=str(Sat.year), mon=str(Sat.month), day=str(Sat.day))
        replace_dict['周日'] = '{year}年{mon}月{day}日'.format(year=str(Sun.year), mon=str(Sun.month), day=str(Sun.day))
        replace_dict['这周'] = '{year1}年{mon1}月{day1}日到{year2}年{mon2}月{day2}日'.format(
            year1=str(Mon.year), mon1=str(Mon.month), day1=str(Mon.day), 
            year2=str(next_Mon.year), mon2=str(next_Mon.month), day2=str(next_Mon.day))

        # 季节的处理，考虑今年对应的季节是否到了，没到的话，取上一年对应季节
        if month >= 12:
            replace_dict['冬季'] = '{year1}年{mon1}月到{year2}年{mon2}月'.format(
            year1=str(year), mon1=str(12), year2=str(year+1), mon2=str(3))
        else:
            replace_dict['冬季'] = '{year1}年{mon1}月到{year2}年{mon2}月'.format(
            year1=str(year-1), mon1=str(12), year2=str(year), mon2=str(3))

        if month >= 9:
            replace_dict['秋季'] = '{year1}年{mon1}月到{year2}年{mon2}月'.format(
            year1=str(year), mon1=str(9), year2=str(year), mon2=str(12))
        else:
            replace_dict['秋季'] = '{year1}年{mon1}月到{year2}年{mon2}月'.format(
            year1=str(year-1), mon1=str(9), year2=str(year-1), mon2=str(12))

        if month >= 6:
            replace_dict['夏季'] = '{year1}年{mon1}月到{year2}年{mon2}月'.format(
            year1=str(year), mon1=str(6), year2=str(year), mon2=str(9))
        else:
            replace_dict['夏季'] = '{year1}年{mon1}月到{year2}年{mon2}月'.format(
            year1=str(year-1), mon1=str(6), year2=str(year-1), mon2=str(9))

        if month >= 3:
            replace_dict['春季'] = '{year1}年{mon1}月到{year2}年{mon2}月'.format(
            year1=str(year), mon1=str(3), year2=str(year), mon2=str(6))
        else:
            replace_dict['春季'] = '{year1}年{mon1}月到{year2}年{mon2}月'.format(
            year1=str(year-1), mon1=str(3), year2=str(year-1), mon2=str(6))
            
        replace_dict['春天'] = replace_dict['春季']
        replace_dict['夏天'] = replace_dict['夏季']
        replace_dict['秋天'] = replace_dict['秋季']
        replace_dict['冬天'] = replace_dict['冬季']
        
        text = text
        for k in replace_dict:
            text = text.replace(k, replace_dict[k])
        
        # 针对常州空天院的特殊需求，将“近3年内7月”转成最近的一个7月
        special_mon = re.findall('近\d+年[内|间]?(\d+)月', text)
        if special_mon:
            spe_mon = int(special_mon[0])
            if spe_mon > month:
                text = re.sub('近\d+年[内|间]?', replace_dict['去年'], text)
            else:
                text = re.sub('近\d+年[内|间]?', replace_dict['今年'], text)
        
        return text


if __name__ == "__main__":
    time_operator = TimeOperate()
    # ret = time_operator.standard("2020年8月云量")
    # print(ret)
    # time_result_filter = time_operator.to_filter(ret[0], 3)
    # print(time_result_filter)
    # result = time_operator.to_filter(ret[0], "1")
    # print(result)
    #
    # ret = time_operator.standard(
    #     "Sales target for 2019 / 2018", "en_US")
    # print(ret)
    # result = time_operator.to_filter(ret[0], "1")
    # print(result)
    #
    # ret = time_operator.standard(
    #     "Sales target for each year", "en_US")
    # print(ret)
    # result = time_operator.to_filter(ret[0], "1")
    # print(result)
    text  = '周六的数据'
    text = time_operator.trans_time_word(text)
    print(text)