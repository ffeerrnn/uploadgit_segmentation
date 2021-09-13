# -*- coding: utf-8 -*-
'''
@文件    :model_data_processor.py
@说明    :
@时间    :2020/08/29
@作者    :chenxu
@版本    :1.0
'''

class ModelDataProcessor(object):
    """
    处理模型的输入、输出
    """
    def __init__(self):
        pass

    def process_input_data(self, table_info, query):
        """
        把数据处理成模型的输入格式
        """

        type_exchange_dict = {'TEXT': 'text', 'DATE': 'real', 'NUM':'real'}

        table_name = table_info['origin_name']

        col_type = {}
        col_value = {}
        name_id_dict = {}
        for field in table_info['fields']:
            field_name = field['alias'] if field['alias'] else field['name']
            col_type[field['alias']] = type_exchange_dict[field['type']]
            col_value[field['alias']] = field['dataEnum']
            name_id_dict[field_name] = field['id']

        sample_data = [query, col_type, col_value, table_name]

        return sample_data, name_id_dict

    def process_output_data(self, task1_result, one_data, name_id_dict):
        """
        把模型的输出，统一成接口的格式
        """
        
        agg_dict = {0: None, 1: 'AVG', 2: 'MAX', 3: 'MIN', 4: 'COUNT', 5: 'SUM'}
        cond_op_dict = {0: 'GT', 1: 'LT', 2: 'EQ', 3: 'NEQ'}
        cond_conn_op = {1: 'AND', 2: 'OR'}

        column_names = list(one_data[1].keys())
        task1_result = task1_result[0]

        result = {'select': [], 'where': {}, 'group': [], 'order': [], 'limit': None, 'having': {}}
        
        for i in range(len(task1_result['sel'])):
            select_column_name = column_names[task1_result['sel'][i]]
            select_column_id = name_id_dict[select_column_name]
            select_item = {'aggregate': agg_dict[task1_result['agg'][i]], 'fieldId': select_column_id}
            result['select'].append(select_item)

        where_info = {}
        if len(task1_result['conds']) == 1:
            where_info['nodeType'] = 'CONDITION'

            cond = task1_result['conds'][0]
            where_column_name = column_names[cond[0]]
            where_column_id = name_id_dict[where_column_name]
            where_operate = cond_op_dict[cond[1]]
            where_value = cond[2]
            where_info['fieldId'] = where_column_id
            where_info['operate'] = where_operate
            where_info['fieldValue'] = where_value

        elif len(task1_result['conds']) > 1:
            where_info['nodeType'] = 'GROUP'
            where_info['logicalType'] = cond_conn_op[task1_result['cond_conn_op']]

            child = []
            for cond in task1_result['conds']:
                child_info = {}
                child_info['nodeType'] = 'CONDITION'

                where_column_name = column_names[cond[0]]
                where_column_id = name_id_dict[where_column_name]
                where_operate = cond_op_dict[cond[1]]
                where_value = cond[2]
                child_info['fieldId'] = where_column_id
                child_info['operate'] = where_operate
                child_info['fieldValue'] = where_value

                child.append(child_info)
            where_info['child'] = child
        
        result['where'] = where_info

        return result