# _*_coding:utf-8_*_
# 作者： 万方名
# 创建日期： 2020/8/21   16:23
# 文件： nl2sql_main.py

import json
import pandas as pd
import numpy as np

from nl2sql_parser.services.model_parser.nl2sql_integration.src.utils.data_process import DataProcessCol
from nl2sql_parser.services.model_parser.nl2sql_integration.src.utils.data_process import QueryTokenizer
from nl2sql_parser.services.model_parser.nl2sql_integration.src.utils.data_process import SqlLabelEncoder
from nl2sql_parser.services.model_parser.nl2sql_integration.src.utils.data_process import DataProcessValue
from nl2sql_parser.services.model_parser.nl2sql_integration.src.utils.data_process import DataProcessValueSeq
from nl2sql_parser.services.model_parser.nl2sql_integration.src.utils.data_process import CandidateCondsExtractor
from nl2sql_parser.services.model_parser.nl2sql_integration.src.utils.data_process import FullSampler

from nl2sql_parser.services.model_parser.nl2sql_integration.src.column_model import ColumnModel
from nl2sql_parser.services.model_parser.nl2sql_integration.src.value_model import ValueModel

from collections import defaultdict
from keras_bert import load_vocabulary, get_checkpoint_paths

from conf.model_conf import *


def process_result(task1_result, task2_result):
    for query_id, pred_sql in enumerate(task1_result):
        cond = list(task2_result.get(query_id, []))
        pred_sql['conds'] = cond

        json_str = json.dumps(pred_sql, ensure_ascii=False)
        return json_str


def outputs_to_sqls(preds_cond_conn_op, preds_sel_agg, preds_cond_op, header_lens, label_encoder):
    """
    Generate sqls from model outputs
    """
    preds_cond_conn_op = np.argmax(preds_cond_conn_op, axis=-1)
    preds_cond_op = np.argmax(preds_cond_op, axis=-1)

    sqls = []

    for cond_conn_op, sel_agg, cond_op, header_len in zip(preds_cond_conn_op, preds_sel_agg, preds_cond_op,
                                                          header_lens):
        sel_agg = sel_agg[:header_len]
        # force to select at least one column for agg
        sel_agg[sel_agg == sel_agg[:, :-1].max()] = 1
        sel_agg = np.argmax(sel_agg, axis=-1)

        sql = label_encoder.decode(cond_conn_op, sel_agg, cond_op)
        sql['conds'] = [cond for cond in sql['conds'] if cond[0] < header_len]

        sel, agg = [], []
        for col_id, agg_op in zip(sql['sel'], sql['agg']):
            if col_id < header_len:
                sel.append(col_id)
                agg.append(agg_op)

        sql['sel'] = sel
        sql['agg'] = agg
        sqls.append(sql)
    return sqls


def merge_result(qc_pairs, result, threshold):
    select_result = defaultdict(set)
    for pair, score in zip(qc_pairs, result):
        if score > threshold:
            select_result[pair.query_id].update([pair.cond_sql])
    return dict(select_result)


def result2sql(task1_result, one_data):
    agg_dict = {1: 'AVG', 2: 'MAX', 3: 'MIN', 4: 'COUNT', 5: 'SUM'}
    cond_op_dict = {0: '>', 1: '<', 2: '==', 3: '!='}
    cond_conn_op = {1: 'and', 2: 'or'}

    column_names = list(one_data[1].keys())
    task1_result = task1_result[0]

    # 前半段sql
    if task1_result['agg'][0] == 0:
        sql_select = 'select {} from {} '.format(column_names[task1_result['sel'][0]], one_data[3])
    else:
        sql_select = 'select {}({}) from {} '.format(agg_dict[task1_result['agg'][0]],
                                                     column_names[task1_result['sel'][0]], one_data[3])

    # 后半段sql
    sql_where = ' where '
    if len(task1_result['conds']) == 1:
        sql_where = 'where {0} {1} {2}'.format(column_names[task1_result['conds'][0][0]],
                                               cond_op_dict[task1_result['conds'][0][1]], task1_result['conds'][0][2])
    elif len(task1_result['conds']) > 1:
        for cond in task1_result['conds']:
            sql_where += column_names[cond[0]]
            sql_where += ' '
            sql_where += cond_op_dict[cond[1]]
            sql_where += ' '
            sql_where += cond[2]
            sql_where += ' '
            sql_where += cond_conn_op[task1_result['cond_conn_op']]
            sql_where += ' '
        sql_where = sql_where[:-4] if task1_result['cond_conn_op'] == 2 else sql_where[:-5]

    finall_sql = sql_select + sql_where
    return finall_sql


class Nl2SQL(object):
    def __init__(self):
        paths = get_checkpoint_paths(bert_model_path)
        token_dict = load_vocabulary(paths.vocab)
        self.qt = QueryTokenizer(token_dict)
        self.cm = ColumnModel()
        self.vm = ValueModel()
        self.label_encoder = SqlLabelEncoder()
        self.col_model = self.cm.get_col_model()
        self.value_model, self.tokenizer = self.vm.get_value_model()
        self.nl2sql(self.get_sample_data())

    def nl2sql(self, one_data):
        print("\n\nlalanl2sql")
        # 把一条数据转成可以训练的数据，使用index获取的数据的时候自动转换成矩阵的形式
        col_test_data = DataProcessCol(
            data=one_data,
            tokenizer=self.qt,
            shuffle_header=False,
            max_len=160
        )

        # 获取预测列的模型

        # 预测列
        header_lens = np.sum(col_test_data[0]['input_header_mask'], axis=-1)
        preds_cond_conn_op, preds_sel_agg, preds_cond_op = self.col_model.predict_on_batch(col_test_data[0])
        print("preds_cond_conn_op", preds_cond_conn_op.shape, preds_cond_conn_op)
        print("preds_sel_agg", preds_sel_agg.shape, preds_sel_agg)
        print("preds_cond_op", preds_cond_op.shape, preds_cond_op)

        m1_result = [preds_cond_conn_op, preds_sel_agg, preds_cond_op]
        task1_result = outputs_to_sqls(preds_cond_conn_op, preds_sel_agg, preds_cond_op, header_lens,
                                       self.label_encoder)
        print('task1_result:{}'.format(task1_result))

        # 获取预测值的模型和数据

        value_data = DataProcessValue(one_data,
                                      candidate_extractor=CandidateCondsExtractor(share_candidates=True),
                                      has_label=False, model_1_outputs=task1_result)

        value_data_seq = DataProcessValueSeq(value_data, self.tokenizer, sampler=FullSampler(), shuffle=False,
                                             batch_size=128)
        print("value_data_seq", value_data_seq)

        # 预测值
        value_result = self.value_model.predict_generator(value_data_seq, verbose=1)
        print('value_result:{}'.format(value_result))
        task2_result = merge_result(value_data, value_result, threshold=0.95)
        print('task2_result:{}'.format(task2_result))

        final_output_file = 'final_output_8_26.json'
        with open(final_output_file, 'w') as f:
            for query_id, pred_sql in enumerate(task1_result):
                cond = list(task2_result.get(query_id, []))
                pred_sql['conds'] = cond
                json_str = json.dumps(pred_sql, ensure_ascii=False)
                f.write(json_str + '\n')

        sql = result2sql(task1_result, one_data)

        return task1_result, task2_result, sql

    def get_sample_data(self):
        """
        在模型初始化后调用一次预测，解决ValueError("Tensor %s is not an element of this graph." % obj)问题
        """
        question_test = 'PE2011大于11或者EPS2011大于11的公司有哪些'
        table = {"证券代码": "text", "公司名称": "text", "股价": "real", "EPS2011": "real", "EPS2012E": "real", "EPS2013E": "real",
                "PE2011": "real", "PE2012E": "real", "PE2013E": "real", "NAV": "text", "折价率": "text", "PB2012Q1": "real",
                "评级": "text"}
        col_value = {"证券代码": ["600340.SH", "000402.SZ", "600823.SH", "600716.SH", "000608.SZ", "002285.SZ"],
                    "公司名称": ['华夏幸福', '金融街', '世茂股份', '凤凰股份', '阳光股份', '世联地产'],
                    "股价": ['17.49', '6.53', '11.79', '5.54', '4.79', '15.07'],
                    "EPS2011": ['1.54', '0.67', '1.01', '0.32', '0.23', ' 0.48'],
                    "EPS2012E": ['2.03', '0.78', '1.13', '0.45', '0.29', '0.84'],
                    "EPS2013E": ['2.67', '0.91', '1.39', '0.66', '0.32', '1.05'],
                    "PE2011": ['11.36', '9.8', '11.66', '17.51', '20.76', '31.27'],
                    "PE2012E": ['8.61', '8.41', '10.4', '12.45', '16.31', '18.04'],
                    "PE2013E": ['6.56', '7.2', '8.47', '8.39', '14.75', '14.34'],
                    "NAV": ['None', '-38.7', '22.09', '7.46', '6.71', 'None'],
                    "折价率": ['None', '-38.7', '-46.6', '-25.8', '-28.7', 'None'],
                    "PB2012Q1": ['5.8', '1.0', '1.3', '2.5', '1.4', '3.6'],
                    "评级": ['推荐', '谨慎推荐', '无', '谨慎推荐', '谨慎推荐', '无']}
        table_name = '表3：2019年第4周（2019.01.28 - 2019.02.03）全国电影票房TOP10'
        sample_data = [question_test, table, col_value, table_name]
        
        return sample_data




def get_table_col(table_id, table_rows, table_names, table_headers, table_types):
    table_id = 'Table_' + table_id
    index = table_names.index(table_id)
    table_columns = table_headers[index]
    columns_types = table_types[index]
    columns_values = table_rows[index]

    rows = np.asarray(table_rows[index])

    table = {}
    col_value = {}
    for i in range(len(table_columns)):
        col_name = table_columns[i]
        col_type = columns_types[i]
        table[col_name] = col_type

        col_value[col_name] = rows[:, i].tolist()

    return table, col_value


def random_create_model2_input(num, train_path, table_info_path):
    train_data = pd.read_json(train_path, lines=True, encoding='utf-8')
    table_data = pd.read_json(table_info_path, lines=True, encoding='utf-8')

    questions = train_data['question'].tolist()
    table_ids = train_data['table_id'].tolist()

    table_rows = table_data['rows'].tolist()
    table_names = table_data['name'].tolist()
    table_types = table_data['types'].tolist()
    table_headers = table_data['header'].tolist()

    result_2 = []
    table_id = table_ids[num]
    table, col_value = get_table_col(table_id, table_rows, table_names, table_headers, table_types)
    result_2.append(questions[num])
    result_2.append(table)
    result_2.append(col_value)

    return result_2


if __name__ == '__main__':
    # train_path = "../data/train/train.json"
    # table_info = "../data/train/train.tables.json"
    # num_questions = 2
    # inputs_dict = random_create_model2_input(num_questions, train_path, table_info)

    question_test = 'PE2011大于11或者EPS2011大于11的公司有哪些'
    table = {"证券代码": "text", "公司名称": "text", "股价": "real", "EPS2011": "real", "EPS2012E": "real", "EPS2013E": "real",
             "PE2011": "real", "PE2012E": "real", "PE2013E": "real", "NAV": "text", "折价率": "text", "PB2012Q1": "real",
             "评级": "text"}
    col_value = {"证券代码": ["600340.SH", "000402.SZ", "600823.SH", "600716.SH", "000608.SZ", "002285.SZ"],
                 "公司名称": ['华夏幸福', '金融街', '世茂股份', '凤凰股份', '阳光股份', '世联地产'],
                 "股价": ['17.49', '6.53', '11.79', '5.54', '4.79', '15.07'],
                 "EPS2011": ['1.54', '0.67', '1.01', '0.32', '0.23', ' 0.48'],
                 "EPS2012E": ['2.03', '0.78', '1.13', '0.45', '0.29', '0.84'],
                 "EPS2013E": ['2.67', '0.91', '1.39', '0.66', '0.32', '1.05'],
                 "PE2011": ['11.36', '9.8', '11.66', '17.51', '20.76', '31.27'],
                 "PE2012E": ['8.61', '8.41', '10.4', '12.45', '16.31', '18.04'],
                 "PE2013E": ['6.56', '7.2', '8.47', '8.39', '14.75', '14.34'],
                 "NAV": ['None', '-38.7', '22.09', '7.46', '6.71', 'None'],
                 "折价率": ['None', '-38.7', '-46.6', '-25.8', '-28.7', 'None'],
                 "PB2012Q1": ['5.8', '1.0', '1.3', '2.5', '1.4', '3.6'],
                 "评级": ['推荐', '谨慎推荐', '无', '谨慎推荐', '谨慎推荐', '无']}
    table_name = '表3：2019年第4周（2019.01.28 - 2019.02.03）全国电影票房TOP10'
    sample_data = [question_test, table, col_value, table_name]

    nl2sql = Nl2SQL()

    column_result, value_result, finall_sql = nl2sql.nl2sql(sample_data)
    print('task1_result的结果为：{0}\ntask2_result的结果为：{1}\n最终的sql语句为：{2}'.format(column_result, value_result, finall_sql))
