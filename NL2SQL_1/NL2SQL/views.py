# -*- coding: utf-8 -*-

import json
import logging, traceback

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

from nl2sql_parser.services.nl2sqlparser import Parser

parser = Parser()
logger = logging.getLogger('develop')

# 通用自然语言转sql接口
@csrf_exempt
def nl2sql(request):

    try:
        request_data = json.loads(request.body, strict=False)
        question = request_data['question']
        table_ids = request_data.get('table_ids', [])
        
        table_ids, sql, _ = parser.parse(table_ids, question)
        res = {"code": 1, "msg": "OK", "table_ids": table_ids, "sql": sql, "question": question}    
    except:
        logger.error(traceback.format_exc())
        res = {"code": 0, "message": "服务异常"}

    response = HttpResponse(json.dumps(res), content_type="application/json")
    return response


# 通用自然语言转sql接口
@csrf_exempt
def aogeo(request):

    try:
        request_data = json.loads(request.body, strict=False)
        question = request_data['question']
        
        result = parser.parse_for_aogeo(question)
        if result:
            res = {"code": 1, "msg": "语义解析成功", "result": result}
        else:
            res = {"code": 0, "message": "解析失败"}    
    except:
        logger.error(traceback.format_exc())
        res = {"code": 0, "message": "服务异常"}

    response = HttpResponse(json.dumps(res), content_type="application/json")
    return response