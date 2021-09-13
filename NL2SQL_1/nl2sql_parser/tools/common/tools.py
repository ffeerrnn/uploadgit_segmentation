# -*- coding: utf-8 -*-


"""
   author: LinJie.Xu
   time: 2019/6/12
"""

import hashlib
import json
import logging
import time
import uuid
import requests
from functools import wraps
from NL2SQL.settings import SW_SERVER

class Tools(object):
    analysis_number = 0  # 智能挖掘占用数量
    logger = logging.getLogger("main")

    @staticmethod
    def print_cool_title(name):
        """
        打印标题
        :param name:
        :return:
        """

        Tools.logger.info('--' * 10 + '{name}'.format(name=name) + '--' * 10)

    @staticmethod
    def print_request(name, request_data):
        """
        打印请求数据

        :param name: 请求 API 名称
        :param request_data: 请求对象
        :return: 转换成 json 后的请求数据
        """

        Tools.print_cool_title(name)
        Tools.logger.debug('{name} input----{request_data}'.format(name=name, request_data=request_data))

    @staticmethod
    def data_result(data=None):
        if not data:
            return ""
        else:
            return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def common_result(code, msg, error_msg=None):
        if not error_msg:
            error_msg = msg
        return json.dumps({"code": code, "msg": msg, "errMsg": error_msg}, ensure_ascii=False)

    @staticmethod
    def has_key(map_, key):
        """
        判断 key 是否存在
        """

        if key not in map_:
            return False

        if map_[key] is None:
            return False

        return True

    @staticmethod
    def has_not_key(map_, key):
        """
        判断 key 是否不存在
        """

        if key not in map_:
            return True

        if map_[key] is None:
            return True

        return False

    @staticmethod
    def escape(str_):
        """
        对 str_ 进行转义
        """

        if not str_:
            return str_

        if not isinstance(str_, bytes) and not isinstance(str_, str):
            return str_

        return str_.replace("'", "\\\'").replace('"', "\\\"")

    @staticmethod
    def occupy_analysis():
        """
        占用智能挖掘
        """

        Tools.analysis_number += 1

    @staticmethod
    def can_analysis():
        """
        查看当前是否能够进行智能挖掘
        """

        return Tools.analysis_number < LoadConfig.get_config_by_name(
            "mining", "analysis_number")

    @staticmethod
    def release_analysis():
        """
        释放智能挖掘
        """

        Tools.analysis_number -= 1

    @staticmethod
    def str2timestamp(date_str, func):
        """
        字符串转化为时间戳，或者数字
        """

        try:
            if func in ("TO_SECOND", "TO_MINUTE", "TO_HOUR",
                        "TO_DAY", "TO_WEEK", "TO_QUARTERLY", "TO_MONTH",
                        "TO_YEAR"):
                format_str = "%Y-%m-%d %H:%M:%S"
                print(date_str)
                time_array = time.strptime(date_str, format_str)
                print(time_array)
                time_stamp = int(time.mktime(time_array))
                # 需要转化成毫秒
                return time_stamp * 1000
            else:
                return int(date_str)
        except BaseException:
            raise Exception(
                "str2timestamp error, str: {0}, func: {1}".format(
                    date_str, func))

    @staticmethod
    def timestamp2str(timestamp, func):
        """
        时间戳转化为字符串
        """

        try:
            if func in ("TO_SECOND", "TO_MINUTE", "TO_HOUR",
                        "TO_DAY", "TO_WEEK", "TO_QUARTERLY", "TO_MONTH",
                        "TO_YEAR"):
                time_array = time.localtime(timestamp / 1000)
                time_str = time.strftime("%Y-%m-%d %H:%M:%S", time_array)
                return time_str
            else:
                return str(timestamp)
        except BaseException:
            raise Exception(
                "timestamp2str error, str: {0}, func: {1}".format(
                    timestamp, func))

    @staticmethod
    def md5(str_):
        """
        获取 md5 加密结果
        """

        m = hashlib.md5()
        m.update(str_.encode("utf-8"))
        return m.hexdigest()

    @staticmethod
    def id_generator(pre):
        """
        生成 id
        """

        id = uuid.uuid1()
        md5_code = Tools.md5(str(id))
        return "{0}_{1}{2}".format(pre, md5_code[0:8], md5_code[-8:])

    @staticmethod
    def sql_query2str(data):
        """
        将sql查询结果中的int，float 转为str
        """

        if not data:
            return data
        ndata = []

        for it in data:
            nit = []
            for i in it:
                if isinstance(i, int) or isinstance(i, float):
                    nit.append(str(i))
                else:
                    nit.append(i)
            ndata.append(nit)

        return ndata

    @staticmethod
    def execute_time(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)

            Tools.logger.debug("{} execute cost time: {}".format(func.__name__, time.time() - start))

            return result

        return wrapper

    @staticmethod
    def judge_language(word):
        """
        包含汉字的就认为是中文，否则认为是英文
        """

        for w in word:
            if "\u4e00" <= w <= "\u9fa5":
                return LanguageType.ZH_CN
        return LanguageType.EN_US

    @staticmethod
    def is_number(s):
        """
        判断是不是数字
        """

        try:
            float(s)
            return True
        except ValueError:
            pass

        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass

        return False

    @staticmethod
    def chinese_to_arabic(cn):
        """
        中文数值转化为阿拉伯数字
        """

        cn_num = {
            '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '零': 0
        }

        cn_unit = {
            '十': 10,
            '百': 100,
            '千': 1000,
            '万': 10000,
            '亿': 100000000,
        }

        unit = 0
        ldig = []
        for cndig in reversed(cn):
            if cndig in cn_unit:
                unit = cn_unit.get(cndig)
                if unit == 10000 or unit == 100000000:
                    ldig.append(unit)
                    unit = 1
            else:
                dig = cn_num.get(cndig)
                if unit:
                    dig *= unit
                    unit = 0
                ldig.append(dig)
        if unit == 10:
            ldig.append(10)
        val, tmp = 0, 0
        for x in reversed(ldig):
            if x == 10000 or x == 100000000:
                val += tmp * x
                tmp = 0
            else:
                tmp += x
        val += tmp
        return val

    def sw(self, text):
        """
        调用百分点分词服务
        :param text:
        :return:
        """
        data = json.dumps({'text': [text]})
        sentencesList = requests.post(SW_SERVER, data=data).json()['result']
        contents = sentencesList[0].strip().split('\t')
        result = []
        for content in contents:
            if "||" in content:
                word, postag = content[:-2], content[-1]
            else:
                word, postag = content.split('|', 1)
            result.append([word, postag])
        return result

if __name__ == "__main__":
    print(Tools.id_generator("eee"))
    print(Tools.md5("ddd"))
    print(Tools().sw('今天是3月5日'))
