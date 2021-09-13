# -*- coding: utf-8 -*-


"""
   author: LinJie.Xu
   time: 2019/6/12
"""


class ZhTarget(object):

    def __init__(self):
        self.group_list = ["各", "各个", "每", "每个", "不同"]
        self.connect_list = ["和", ",", "、"]
        self.sort_list = []
        self.sum_list = ["总", "求和", "总数", "量"]
        self.avg_list = ["平均", "平均值", "平均数"]
        self.distinct_count_list = ["不同数量", "去重计数"]
        self.count_list = ["计数", "数量", "多少", "有多少"]
        self.max_list = ["最高", "最多", "最贵", "最牛", "最强", "最老", "最大", "最失调"]
        self.min_list = ["最低", "最少", "最便宜", "最没用", "最弱", "最年轻", "最小", "最均衡"]
        self.templates = [
                        {"template": "*${agg1}${field1}*",
                         "result": {"select": [{"field": "${field1}", "agg": "${agg1}"}], "group": []}},
                        {"template": "*${order1}${field1}*",
                         "result": {"select": [{"field": "${field1}", "agg": "${order1}"}], "group": []}},
                        {"template": "*${field1}*",
                        "result": {"select": [{"field": "${field1}"}], "group": []}},
                        {"template": "*${field1}${connect}${field2}*",
                         "result": {"select": [{"field": "${field1}"},{"field": "${field2}"}], "group": []}},
                        {"template": "*${group}${field1-dim}*${field2-index}*",
                           "result": {"select": [{"field": "${field2}"}], "group": [{"field": "${field1}"}]}},
                        {"template": "*${group}${field1-dim}*${agg1}${field2-index}*",
                         "result": {"select": [{"field": "${field2}", "agg":"${agg1}"}], "group": [{"field": "${field1}"}]}},
                        {"template": "*${group}${field1-dim}*${order1}*${field2-index}*",
                         "result": {"select": [{"field": "${field2}", "agg": "${order1}"}], "group": [{"field": "${field1}"}]}},
                        {"template": "*${group}${field1-dim}*${field2-index}${agg1}*",
                         "result": {"select": [{"field": "${field2}", "agg": "${agg1}"}], "group": [{"field": "${field1}"}]}},
                        {"template": "*${field1-index}${order1}*${num1}*${field2-dim}*",
                           "result": {"select": [{"field": "${field1}"}], "limit": "${num1}",
                                      "order": [{"field": "${field1}", "agg": "SUM", "type": "${order1}"}],
                                      "group": [{"field": "${field2}"}]}},
                        {"template": "*${field1}${order1}的*${field2}*",
                           "result": {"select": [{"field": "${field2}"}, {"field":"${field1}"}], "group": [{"field": "${field2}"}], "order":[{"field":"${field1}", "type":"${order1}"}]}},
                        {"template": "*${group}${field1}${connect}${field2}*${field3}*",
                            "result": {"select": [{"field": "${field3}"}], "group": [{"field": "${field1}"}, {"field": "${field2}"}]}},
                        {"template": "*${group}${field1}*${group}${field2}*${field3}*",
                         "result": {"select": [{"field": "${field3}"}], "group": [{"field": "${field1}"}, {"field": "${field2}"}]}},
                        {"template": "*${group}${field1}*${field2}${connect}${field3}*",
                         "result": {"select": [{"field": "${field2}"}, {"field": "${field3}"}], "group": [{"field": "${field1}"}]}},
                        {"template": "*${group}${field1}${connect}${field2}*${agg1}${field3}${connect}${field4}*",
                         "result": {"select": [{"field": "${field3}", "agg":"${agg1}"}, {"field": "${field3}", "agg":"${agg1}"}], "group": [{"field": "${field1}"}, {"field": "${field2}"}]}}

        ]

