# -*- coding: utf-8 -*-

# from django.conf import settings
# import NL2SQL.settings as app_setting
# settings.configure(default_settings=app_setting)

import logging
import time
import datetime
import re
from dateutil.relativedelta import relativedelta

from nl2sql_parser.tools.inverted_index.build_index import build_index
from nl2sql_parser.services.template_parser.operates.match_table import MatchTable
from nl2sql_parser.services.template_parser.operates.time_operate import TimeOperate
from nl2sql_parser.tools.data_manage.table_info_loader import TableInfoLoader
from nl2sql_parser.tools.data_manage.model_data_processor import ModelDataProcessor
from nl2sql_parser.services.template_parser.parser import TemplateParser
# from nl2sql_parser.services.model_parser.nl2sql_integration.src.nl2sql_main import Nl2SQL
from nl2sql_parser.tools.nlp.place_name_extract import PlaceNameExtractor
from nl2sql_parser.tools.common.place_coordinate import CHINA

build_index.init()

class Parser(object):
    """
    解析器
    """

    def __init__(self):
        self.logger = logging.getLogger('develop')
        self.table_matcher = MatchTable()
        self.time_operate = TimeOperate()
        self.table_info_loader = TableInfoLoader()
        self.model_data_processor = ModelDataProcessor()
        self.template_parser = TemplateParser()
        # self.model_parer = Nl2SQL()
        self.place_name_extractor = PlaceNameExtractor()

    def merge_result(self, template_result, model_result):
        """
        合并模板解析的结果和模型解析的结果
        :param template_result:
        :param model_result:
        :return:
        """
        result = model_result

        # 合并模板的select部分
        select_dict = {}
        for item in result['select']:
            if item['fieldId'] not in select_dict:
                select_dict[item['fieldId']] = [item['aggregate']]
            else:
                select_dict[item['fieldId']].append(item['aggregate'])
        for item in template_result['select']:
            if item['fieldId'] not in select_dict:
                result['select'].append(item)
            elif item['fieldId'] in select_dict and item['aggregate'] and item['aggregate'] not in select_dict[item['fieldId']]:
                result['select'].append(item)
        
        # 合并模板的where部分
        where = result['where']
        temp_where = template_result['where']
        result['where'] = self.merge_where(where, temp_where)
        
        return result

    def merge_where(self, where_dict1, where_dict2):
        """
        不同步骤得到的where部分合并，以where_dict1为主，都是GROUP时，若logicalType不一致，只保留where_dict1的
        """
        new_where = where_dict1

        if where_dict1 and where_dict2:
            if where_dict1['nodeType'] == 'CONDITION' and where_dict2['nodeType'] == 'CONDITION':
                if where_dict1 != where_dict2:
                    new_where = {'nodeType': 'GROUP', 'logicalType': 'AND', 'child': [where_dict1, where_dict2]}
            elif where_dict2['nodeType'] == 'CONDITION' and where_dict1['nodeType'] == 'GROUP':
                if where_dict2 not in new_where['child']:
                    new_where['child'].append(where_dict2)
            elif where_dict2['nodeType'] == 'GROUP' and where_dict1['nodeType'] == 'CONDITION':
                new_where = where_dict2
                if where_dict1 not in new_where['child']:
                    new_where['child'].append(where_dict1)
            elif where_dict1['nodeType'] == 'GROUP' and where_dict2['nodeType'] == 'GROUP':
                if where_dict1['logicalType'] == where_dict2['logicalType']:
                    for child in where_dict2['child']:
                        if child not in new_where['child']:
                            new_where['child'].append(child)

        elif not where_dict1 and where_dict2:
            new_where = where_dict2

        return new_where

    def add_time_info(self, result_without_time, time_filter, time_group):
        """
        结果中增加时间信息
        """
        time_filter = self.expand_time(time_filter) if time_filter else {}
        result_with_time = result_without_time
        result_with_time['where'] = self.merge_where(result_without_time['where'], time_filter)
        
        return result_with_time

    def expand_time(self, time_filter):
        """
        把解析出的类似"时间"="2019年9月"这样的条件转化成大于9月1日0点，小于10月1日0点
        """
        if time_filter['nodeType'] == 'CONDITION':

            if time_filter['operate'] == 'EQ':
                if time_filter['func'] not in ['TO_YEAR', 'TO_MONTH', 'TO_DAY']:
                    return time_filter

                timeArray = time.strptime(time_filter['fieldValue'], "%Y-%m-%d %H:%M:%S")
                if time_filter['func'] == 'TO_YEAR':
                    start_year = timeArray.tm_year
                    end_year = start_year + 1
                    start_time = '{}-01-01 00:00:00'.format(str(start_year))
                    end_time = '{}-01-01 00:00:00'.format(str(end_year))
                elif time_filter['func'] == 'TO_MONTH':
                    year = timeArray.tm_year
                    start_month = timeArray.tm_mon
                    end_month = start_month + 1
                    start_time = '{year}-{mon}-01 00:00:00'.format(year=str(year), mon='{:02d}'.format(start_month))
                    end_time = '{year}-{mon}-01 00:00:00'.format(year=str(year), mon='{:02d}'.format(end_month))

                elif time_filter['func'] == 'TO_DAY':
                    year = timeArray.tm_year
                    mon = timeArray.tm_mon
                    start_day = timeArray.tm_mday
                    end_day = start_day + 1
                    start_time = '{year}-{mon}-{day} 00:00:00'.format(year=str(year), mon='{:02d}'.format(mon), day='{:02d}'.format(start_day))
                    end_time = '{year}-{mon}-{day} 00:00:00'.format(year=str(year), mon='{:02d}'.format(mon), day='{:02d}'.format(end_day))

                childs = []
                childs.append({'nodeType': 'CONDITION', 'fieldId': time_filter['fieldId'], 'operate': 'GT', 'fieldValue': start_time})
                childs.append({'nodeType': 'CONDITION', 'fieldId': time_filter['fieldId'], 'operate': 'LT', 'fieldValue': end_time})

                return {'nodeType': 'GROUP', 'logicalType': 'AND', 'child': childs}
                
            else:
                return time_filter
        
        elif time_filter['nodeType'] == 'GROUP':
            childs = []
            for child in time_filter['child']:
                new_child = self.expand_time(child)
                childs.append(new_child)
            time_filter['child'] = childs
        
        return time_filter

    def parse(self, table_ids, query, time_mode='default'):
        """

        :return:
        """
        self.logger.info('-------------------------开始解析一条数据----------------------- ')
        self.logger.info('输入: table_ids: {table_ids}, query: {query}'.format(table_ids=table_ids, query=query))

        # 把一些时间实体转换成具体时间
        query = self.time_operate.trans_time_word(query)
        self.logger.info('时间转化后的query: {}'.format(query))

        # 匹配表
        table_found = True
        if not table_ids:
            table_ids, table_found = self.table_matcher.match_aogeo(query)
        self.logger.info('match tables: {}'.format(table_ids))

        # 匹配到多张表的情况，用第一张表的信息做后续解析
        first_table_info = self.table_info_loader.get_first_table_info(table_ids)
        self.logger.info('use table info: {}'.format(first_table_info))
        
        template_result_without_time, time_filter, time_group = self.template_parser.parse(first_table_info, query)
        self.logger.info('template_result_without_time: {}'.format(template_result_without_time))
        self.logger.info('time_filter: {}'.format(time_filter))
        self.logger.info('time_group: {}'.format(time_group))
        """
        model_input_data, name_id_dict = self.model_data_processor.process_input_data(first_table_info, query)
        column_result, value_result, finall_sql = self.model_parer.nl2sql(model_input_data)
        self.logger.info('model output: column_result: {column}, value_result: {value}, finall_sql: {sql}'
        .format(column=column_result, value=value_result, sql=finall_sql))
        model_result = self.model_data_processor.process_output_data(column_result, model_input_data, name_id_dict)
        self.logger.info('model_result: {}'.format(model_result))
        
        result_without_time = self.merge_result(template_result_without_time, model_result)
        """
        result_without_time = template_result_without_time
        # 增加时间信息
        if time_mode == 'bfd_bi':
            result = result_without_time
        else:
            result = self.add_time_info(result_without_time, time_filter, time_group)

        return table_ids, result, table_found

    def parse_for_aogeo(self, query):
        """
        地球观测的问题解析
        """
        def parse_where(where):
            """
            转换where条件，由于目前客户定义条件之间的关系都是and，所以按都是and转换，正常不应该这么转换，而是输出sql成分去拼接sql
            """
            if where['nodeType'] == 'GROUP':
                result = []
                for child in where['child']:
                    result.extend(parse_where(child))
                return result
            else:
                return [where]

        def imagegsd_by_rule(imagegsd, query, oral_imagegsd_dict):
            """
            需要规则处理的分辨率
            """
            for k in oral_imagegsd_dict:
                if k in query:
                    imagegsd['property'] = 'imageGsd'
                    imagegsd['minValue'] = oral_imagegsd_dict[k]['minValue']
                    imagegsd['maxValue'] = oral_imagegsd_dict[k]['maxValue']
                    return imagegsd

            return imagegsd

        result = {'region': {}, 'satelliteList': [], 'conditions': [], 'type': '1', 'time': {}}
        query = query.upper() # 按客户需求，将英文都转成大写后处理
        origin_query = query
        # 解析地区信息
        place_name = self.place_name_extractor.extract(query)
        place_replace_dict = {'北京': '北京市', '天津': '天津市', '上海': '上海市', '重庆': '重庆市'}
        admincode_dict = {'北京市': '110000', '天津市': '120000', '上海市': '310000', '重庆市': '500000'}
        if place_name:
            result['region'] = {'type': 0, 'level': place_name[0]['level'], 'geom': place_name[0]['name'], 'topology': 1,
            "admincode": place_name[0]['admincode']}
            # 行政区划表里“北京”是省一级，“北京市”是市一级，这里“北京”替换成“北京市”，“北京市”改成省一级
            if result['region']['geom'] in place_replace_dict.keys():
                result['region']['geom'] = place_replace_dict[result['region']['geom']]
    
            if result['region']['geom'] in place_replace_dict.values():
                result['region']['level'] = 1
                result['region']['admincode'] = admincode_dict[result['region']['geom']]
        else:
            result['region'] = {'type': 1, 'level': 3, 'geom': CHINA, 'topology': 1, "admincode": ''}      
 
        # 问题中如果没有“数据”，匹配不到select字段内容，会造成其他模块异常
        query = query+'数据' if '数据' not in query else query 
        
        # 问题中的“3米”这样的前面加上“分辨率”
        match_meters = re.findall(r'\d+\.?\d*[米|M|m]', query) 
        for meter in match_meters:
            query = query.replace(meter, '分辨率是'+meter)
        
        table_ids, sql, table_found = self.parse([], query)        
        
        sensorIds = []
        cloudpercent = {'property': 'cloudpercent', 'minValue': '0', 'maxValue': '20'}
        imagegsd = {}
        conds = parse_where(sql['where']) if sql['where'] else []
        for cond in conds:
            field_name = self.table_info_loader.get_field_info(cond['fieldId'])['name']
            if field_name == 'receivetime' and cond['operate'] == 'GT':
                result['time']['starttime'] = cond['fieldValue']
            elif field_name == 'receivetime' and cond['operate'] == 'LT':
                result['time']['endtime'] = cond['fieldValue']

            elif field_name == 'sensor' and cond['operate'] == 'EQ':
                sensorIds.append(cond['fieldValue'])

            elif field_name == 'cloudpercent' and cond['operate'] in ['GT', 'EGT']:
                cloudpercent['minValue'] = cond['fieldValue']
            elif field_name == 'cloudpercent' and cond['operate'] in ['LT', 'ELT']:
                cloudpercent['maxValue'] = cond['fieldValue']
            elif field_name == 'cloudpercent' and cond['operate'] == 'EQ':
                cloudpercent['minValue'] = cond['fieldValue']
                cloudpercent['maxValue'] = cond['fieldValue']

            elif field_name == 'imagegsd' and cond['operate'] in ['GT', 'EGT']:
                imagegsd['property'] = 'imageGsd'
                imagegsd['minValue'] = cond['fieldValue']
            elif field_name == 'imagegsd' and cond['operate'] in ['LT', 'ELT']:
                imagegsd['property'] = 'imageGsd'
                imagegsd['maxValue'] = cond['fieldValue']
            elif field_name == 'imagegsd' and cond['operate'] == 'EQ':
                imagegsd['property'] = 'imageGsd'
                imagegsd['minValue'] = cond['fieldValue']
                imagegsd['maxValue'] = cond['fieldValue']
        
        if imagegsd:
            if 'minValue' not in imagegsd:
                imagegsd['minValue'] = ''
            if 'maxValue' not in imagegsd:
                imagegsd['maxValue'] = ''

        oral_imagegsd_dict = self.table_info_loader.get_cache('oral_imagegsd_dict')

        imagegsd = imagegsd_by_rule(imagegsd, query, oral_imagegsd_dict)
        if imagegsd:
            result['conditions'].append(imagegsd)
        result['conditions'].append(cloudpercent)
        
        
        # 解析到PMS和WFV，要补充其包含的传感器类型
        if 'PMS' in sensorIds:
            sensorIds.extend(['PMS1', 'PMS2'])
        else:
            detail_id_in = False
            for id in ['PMS1', 'PMS2']:
                if id in origin_query:
                    detail_id_in = True
            if 'PMS' in origin_query and not detail_id_in:
                sensorIds.extend(['PMS1', 'PMS2'])

        if 'WFV' in sensorIds:
            sensorIds.extend(['WFV1', 'WFV2', 'WFV3', 'WFV4'])
        else:
            detail_id_in = False
            for id in ['WFV1', 'WFV2', 'WFV3', 'WFV4']:
                if id in origin_query:
                    detail_id_in = True
            if 'WFV' in origin_query and not detail_id_in:
                sensorIds.extend(['WFV1', 'WFV2', 'WFV3', 'WFV4'])

        sensorIds_set = set(sensorIds)

        radar_satellite_name = ['GF3', 'Sentinel1', 'Sentinel2', 'Sentinel3'] # 雷达卫星，要在类别里过滤一下
        # 解析卫星类型信息
        if '雷达' in origin_query:
            result['type'] = '2'
        if '光学' in origin_query:
            result['type'] = '1'
        # 如果只有一张表，并且问题中没有“光学”、“雷达”字样，那卫星类型就是这张表对应卫星的类型
        if len(table_ids) == 1 and '雷达' not in origin_query and '光学' not in origin_query:
            table_info = self.table_info_loader.get_table_info(table_ids[0])
            table_name = table_info['origin_name']
            if table_name in radar_satellite_name:
                result['type'] = '2'
                
        need_radar = False if result['type'] == '1' else True

        imagegsd_satellite_dict = self.table_info_loader.get_cache('imagegsd_satellite_dict')
        table_name_filted_by_imagegsd = []
        if imagegsd:
            min_value = float(imagegsd['minValue'])
            max_value = float(imagegsd['maxValue'])
            for k in imagegsd_satellite_dict:
                if float(k) >= min_value and float(k) <= max_value:
                    table_name_filted_by_imagegsd.extend(imagegsd_satellite_dict[k])
        
        for table_id in table_ids:
            table_info = self.table_info_loader.get_table_info(table_id)
            table_name = table_info['origin_name']
            if table_name_filted_by_imagegsd and table_name not in table_name_filted_by_imagegsd:
                continue
            is_radar = table_name in radar_satellite_name
            if need_radar != is_radar:
                continue
            fields = table_info['fields']
            field_names = [f['name'] for f in fields]
            # 获取这个卫星对应的传感器
            table_sensorIds = []
            if 'sensor' in field_names:
                for field in fields:
                    if field['name'] == 'sensor':
                        table_sensorIds = field['dataEnum']
                        break
            sensorIds_intersection = sensorIds_set.intersection(set(table_sensorIds))
            if not sensorIds or sensorIds_intersection: # 如果没有解析到传感器，则所有表都返回，如果解析到传感器，返回对应的表
                result['satelliteList'].append({'satelliteId': table_name, 'sensorIds': list(sensorIds_intersection)})
        
        # 如果没有解析到相关信息，就返回None
        conds_related_words = re.search(r'数据|影像|\d+米|分辨率|传感器|国内|云量|国外|雷达|光学', origin_query)
        
        if not conds_related_words and not table_found and not place_name and not sensorIds and not result['time']:
            return None
        
        # 如果没有解析到时间，添加默认时间
        now_time = datetime.datetime.now()
        start_time =  now_time - relativedelta(months=3)
        now_time_str = now_time.strftime("%Y-%m-%d %H:%M:%S")
        start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
        if 'starttime' not in result['time']:
            result['time']['starttime'] = start_time_str
        if 'endtime' not in result['time']:
            result['time']['endtime'] = now_time_str
        result['time']['order'] = 1

        return result 
        
if __name__ == '__main__':

    parser = Parser()
    query = '高分2019年9月最大云量'
    result = parser.parse([], query)
    print('result', result)