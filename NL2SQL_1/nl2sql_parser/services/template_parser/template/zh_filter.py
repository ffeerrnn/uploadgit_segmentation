# -*- coding: utf-8 -*-


"""
   author: LinJie.Xu
   time: 2019/6/12
"""


class ZhFilter(object):

    def __init__(self):
        self.gt_list = ["大于", ">", "大", "超过"]
        self.lt_list = ["小于", "<", "小", "低于"]
        self.ge_list = ["至少", ">=", "大于等于", "不低于"]
        self.le_list = ["至多", "<=", "小于等于", "不超过"]
        self.ne_list = ["不等于", "!="]
        self.eq_list = ["等于", "是", "="]
        self.and_list = ["并且", "且"]
        self.or_list = ["或者", "或"]
        self.template = [
            {"template": "${value1}",
             "result": {"nodeType": "CONDITION", "operator": "EQ", "fieldValue": "${value1}",
                        "valueType": "STRING"}},
            {"template": "${field1}${compare1}${value1}",
             "result": {"nodeType": "CONDITION", "field": "${field1}", "operator": "${compare1}",
                        "fieldValue": "${value1}", "valueType": "STRING"}},
            {"template": "${field1} ${verb1} ${value1}",
             "result": {"nodeType": "CONDITION", "operator": "${compare}",
                        "fieldValue": "${value}", "valueType": "STRING"}},
            {"template": "${field1}${compare1}${value1}*${logical1}*${field2}${compare2}${value2}",
             "result": {"logicalType": "${logical1}", "nodeType": "GROUP", "child": [
                 {"nodeType": "CONDITION", "field": "${field1}", "operator": "${compare1}",
                  "fieldValue": "${value1}", "valueType": "STRING"},
                 {"nodeType": "CONDITION", "field": "${field2}", "operator": "${compare2}",
                  "fieldValue": "${value2}", "valueType": "STRING"}]}},
            {"template": "${field1}${compare1}${value1}*${logical1}*${value2}",
             "result": {"logicalType": "OR", "nodeType": "GROUP", "child": [
                 {"nodeType": "CONDITION", "field": "${field1}", "operator": "${compare1}",
                  "fieldValue": "${value1}", "valueType": "STRING"},
                 {"nodeType": "CONDITION", "field": "${field1}", "operator": "${compare1}",
                  "fieldValue": "${value2}", "valueType": "STRING"}]}}


        ]
