# -*- coding: utf-8 -*-

class Time(object):

    def __init__(self):
        self.month_list = [
            "jan.",
            "january",
            "feb.",
            "february",
            "mar.",
            "march",
            "apr.",
            "april",
            "may.",
            "may",
            "jun.",
            "june",
            "jul.",
            "july",
            "aug.",
            "august",
            "sept.",
            "september",
            "oct.",
            "october",
            "nov.",
            "november",
            "dec.",
            "december",
            "janvier",
            "fev.",
            "février",
            "mars.",
            "mars",
            "avr.",
            "avril",
            "mai.",
            "mai",
            "juin.",
            "juin",
            "juillet.",
            "juillet",
            "aout."
            "août",
            "septembre",
            "octobre",
            "novembre",
            "décembre"
        ]
        self.group = ["each", "by", "every", "chaque", "par", "tous"]
        self.templates = [
            {"template": "${group} year",
             "result": {"group": {"func": "TO_YEAR"}}},
            {"template": "${group} month",
             "result": {"group": {"func": "TO_MONTH"}}},
            {"template": "${group} day",
             "result": {"group": {"func": "TO_DAY"}}},
            {"template": "in ${num1}",
             "result": {"value": [{"year": "${num1}",
                                   "func": "TO_YEAR"}]}},
            {"template": "for ${num1} / ${num2}",
             "result": {"value": [{"year": "${num1}",
                                   "func": "TO_YEAR"},
                                  {"year": "${num2}",
                                   "func": "TO_YEAR"}]}},
            {"template": "in ${num1} / ${num2}",
             "result": {"value": [{"year": "${num1}",
                                   "func": "TO_YEAR"},
                                  {"year": "${num2}",
                                   "func": "TO_YEAR"}]}},
            {"template": "${month1} ${num1}",
             "result": {"value": [{"year": "${num1}",
                                   "month": "${month1}",
                                   "func": "TO_MONTH"}]}},
            {"template": "${month1} ${num1} , ${num2}",
             "result": {"value": [{"year": "${num2}",
                                   "month": "${month1}",
                                   "day": "${num1}",
                                   "func": "TO_DAY"}]}},
            {"template": "${month1} ${num1} / ${num2}",
             "result": {"value": [{"year": "${num1}",
                                   "month": "${month1}",
                                   "func": "TO_MONTH"},
                                  {"year": "${num2}",
                                   "month": "${month1}",
                                   "func": "TO_MONTH"}]}},
            {"template": "${month1}",
             "result": {"value": [{"year": "NOW_YEAR",
                                   "month": "${month1}",
                                   "func": "TO_MONTH"}]}},
            {"template": "[每|各]年",
             "result": {"group": {"func": "TO_YEAR"}}},
            {"template": "[每|各]月",
             "result": {"group": {"func": "TO_MONTH"}}},
            {"template": "[每|各][天|日]",
             "result": {"group": {"func": "TO_DAY"}}},
            {"template": "近${num1}年",
             "result": {"start": {"year": "NOW_YEAR-${num1}+1", "func":"TO_YEAR"}}},
            {"template": "近${num1}[天|日]",
             "result": {"start": {"year": "NOW_YEAR", "month": "NOW_MONTH", "day": "NOW_DAY-${num1}", "func": "TO_DAY"}}},
            {"template": "近${num1}个?月",
             "result": {"start": {"year": "NOW_YEAR", "month": "NOW_MONTH-${num1}", "func": "TO_MONTH"}}},
            {"template": "${num1}年",
             "result": {"value": [{"year": "${num1}", "func": "TO_YEAR"}]}},
            {"template": "${num1}月",
             "result": {"value": [{"year": "NOW_YEAR", "month":"${num1}", "func": "TO_MONTH"}]}},
            {"template": "${num1}年${num2}月",
             "result": {"value": [{"year": "${num1}", "month": "${num2}", "func": "TO_MONTH"}]}},
            {"template": "${num1}年${num2}月${num3}日",
             "result": {"value": [{"year": "${num1}", "month": "${num2}", "day": "${num3}", "func": "TO_DAY"}]}},
            {"template": "${num1}月?[到|至]${num2}月",
             "result": {"start": {"year": "NOW_YEAR", "month":"${num1}", "func": "TO_MONTH"}, "end": {"year": "NOW_YEAR", "month":"${num2}", "func": "TO_MONTH"}}},
            {"template": "${num1}年?[到|至]${num2}年",
             "result": {"start": {"year": "${num1}", "func": "TO_YEAR"}, "end": {"year": "${num2}", "func": "TO_YEAR"}}},
            {"template": "${num1}年${num2}月[到|至]${num3}年${num4}月",
             "result": {"start": {"year": "${num1}", "month":"${num2}", "func": "TO_MONTH"},
                        "end": {"year": "${num3}", "month":"${num4}", "func": "TO_MONTH"}}},
            {"template": "${num1}年${num2}月${num3}日[到|至]${num4}年${num5}月${num6}日",
             "result": {"start": {"year": "${num1}", "month":"${num2}", "day": "${num3}", "func": "TO_DAY"},
                        "end": {"year": "${num4}", "month":"${num5}", "day": "${num6}", "func": "TO_DAY"}}},
            {"template": "${group} an(née)?",
             "result": {"group": {"func": "TO_YEAR"}}},
            {"template": "${group} mois",
             "result": {"group": {"func": "TO_MONTH"}}},
            {"template": "${group} jour(née)?",
             "result": {"group": {"func": "TO_DAY"}}},
            {"template": "en ${num1}",
             "result": {"value": [{"year": "${num1}",
                                   "func": "TO_YEAR"}]}},
            {"template": "en ${num1} , ${num2}",
             "result": {"value": [{"year": "${num1}",
                                   "func": "TO_YEAR"},
                                  {"year": "${num2}",
                                   "func": "TO_YEAR"}]}},
            {"template": "en ${num1} ou en ${num2}",
             "result": {"value": [{"year": "${num1}",
                                   "func": "TO_YEAR"},
                                  {"year": "${num2}",
                                   "func": "TO_YEAR"}]}},
            {"template": "dernière année|année dernière",
             "result": {"start": {"year": "NOW_YEAR-1", "func": "TO_YEAR"}}},
            {"template": "${num1} dernières année",
             "result": {"start": {"year": "NOW_YEAR-${num1}", "func": "TO_YEAR"}}},
            {"template": "dernier mois|mois dernier",
             "result": {"start": {"year": "NOW_YEAR", "month": "NOW_MONTH-1", "func": "TO_MONTH"}}},
            {"template": "${num1} derniers mois",
             "result": {"start": {"year": "NOW_YEAR", "month": "NOW_MONTH-${num1}", "func": "TO_MONTH"}}},
            {"template": "dernier jours|d'hier",
             "result": {"start": {"year": "NOW_YEAR", "month": "NOW_MONTH", "day": "NOW_DAY-1", "func": "TO_DAY"}}},
            {"template": "${num1} derniers jours",
             "result": {"start": {"year": "NOW_YEAR", "month": "NOW_MONTH", "day": "NOW_DAY-${num1}", "func": "TO_DAY"}}},
            {"template": "en ${month1} ${num1}",
             "result": {"value": [{"year": "${num1}",
                                   "month": "${month1}",
                                   "func": "TO_MONTH"}]}},
            {"template": "en ${month1} ${num1} , ${num2}",
             "result": {"value": [{"year": "${num2}",
                                   "month": "${month1}",
                                   "day": "${num1}",
                                   "func": "TO_DAY"}]}},
            {"template": "${num1} ${month1} ${num2}",
             "result": {"value": [{"year": "${num2}",
                                   "month": "${month1}",
                                   "day": "${num1}",
                                   "func": "TO_DAY"}]}},
            {"template": "en ${month1} ${num1} , ${month2} ${num2}",
             "result": {"value": [{"year": "${num1}",
                                   "month": "${month1}",
                                   "func": "TO_MONTH"},
                                  {"year": "${num2}",
                                   "month": "${month2}",
                                   "func": "TO_MONTH"}]}},
            {"template": "en ${month1} ${num1} ou en ${month2} ${num2}",
             "result": {"value": [{"year": "${num1}",
                                   "month": "${month1}",
                                   "func": "TO_MONTH"},
                                  {"year": "${num2}",
                                   "month": "${month2}",
                                   "func": "TO_MONTH"}]}},
            {"template": "en ${month1}",
             "result": {"value": [{"year": "NOW_YEAR",
                                   "month": "${month1}",
                                   "func": "TO_MONTH"}]}},
        ]
