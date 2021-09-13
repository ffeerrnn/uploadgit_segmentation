import numpy as np
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

# col_test_data = DataProcessCol(
#             data=one_data,
#             tokenizer=self.qt,
#             shuffle_header=False,
#             max_len=160
#         )
TOKEN_IDS, SEGMENT_IDS = [], []
HEADER_IDS, HEADER_MASK = [], []
question = sample_data[0]
print(question)
col_names = list(sample_data[1].keys())
col_types = list(sample_data[1].values())
print(col_names)
print(col_types)
col_orders = np.arange(len(sample_data[1]))
print(col_orders)

# token_ids, segment_ids, header_ids = self.tokenizer.encode(question, col_names, col_types, col_orders)
_token_cls = '[CLS]'
question_tokens = ['[CLS]', 'p', 'e', '2', '0', '1', '1', '大', '于', '1', '1', '或', '者', 'e', 'p', 's', '2', '0', '1', '1', '大', '于', '1', '1', '的', '公', '司', '有', '哪', '些']
header_tokens = []

import re
def remove_brackets(s):
    # 去括号
    return re.sub(r'[\(\（].*[\)\）]', '', s)
def _tokenize(text):
    r = []
    for c in text.lower():
        r.append(c)
        # if c in self._token_dict:
        #     r.append(c)
        # elif self._is_space(c):
        #     r.append(self.SPACE_TOKEN)
        # else:
        #     r.append(self._token_unk)
    return r
col_type_token_dict = {'text': '[unused11]', 'real': '[unused12]'}
for col_name, col_type in zip(col_names, col_types):

    col_type_token = col_type_token_dict[col_type]
    col_name = remove_brackets(col_name)
    col_name_tokens = _tokenize(col_name)
    col_tokens = [col_type_token] + col_name_tokens
    header_tokens.append(col_tokens)
    print("col_name, col_type", col_name, col_type, col_type_token, col_name_tokens, col_tokens)

all_tokens = [question_tokens] + header_tokens
print("all_tokens", all_tokens)

sents_of_tokens = all_tokens
packed_sents = []
packed_sents_lens = []
_token_sep = '[SEP]'
for tokens in sents_of_tokens:
    packed_tokens = tokens + [_token_sep]
    packed_sents += packed_tokens
    packed_sents_lens.append(len(packed_tokens))

    print("packed_tokens, len(packed_tokens)", packed_tokens, len(packed_tokens))

print("packed_sents", packed_sents)
print("packed_sents_lens", packed_sents_lens)
tokens, tokens_lens = packed_sents, packed_sents_lens
TOKEN_UNK = '[UNK]'  # Token for unknown words
# token_ids = self._convert_tokens_to_ids(tokens)
token_ids =  [101, 158, 147, 123, 121, 122, 122, 1920, 754, 122, 122, 2772, 5442, 147, 158, 161, 123, 121, 122, 122, 1920, 754, 122, 122, 4638, 1062, 1385, 3300, 1525, 763, 102, 11, 6395, 1171, 807, 4772, 102, 11, 1062, 1385, 1399, 4917, 102, 12, 5500, 817, 102, 12, 147, 158, 161, 123, 121, 122, 122, 102, 12, 147, 158, 161, 123, 121, 122, 123, 147, 102, 12, 147, 158, 161, 123, 121, 122, 124, 147, 102, 12, 158, 147, 123, 121, 122, 122, 102, 12, 158, 147, 123, 121, 122, 123, 147, 102, 12, 158, 147, 123, 121, 122, 124, 147, 102, 11, 156, 143, 164, 102, 11, 2835, 817, 4372, 102, 12, 158, 144, 123, 121, 122, 123, 159, 122, 102, 11, 6397, 5277, 102]
segment_ids =  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
header_ids =  [ 31,  37,  43,  47,  56,  66,  76,  84, 93, 102, 107, 112, 122]
print("token_ids", token_ids)
print("segment_ids", segment_ids)
print("header_ids", header_ids)
header_ids = [hid for hid in header_ids if hid < 160]
print("header_ids", header_ids)
header_mask = [1] * len(header_ids)
print("header_mask", header_mask)
TOKEN_IDS, SEGMENT_IDS = [], []
HEADER_IDS, HEADER_MASK = [], []
TOKEN_IDS.append(token_ids)
SEGMENT_IDS.append(segment_ids)
HEADER_IDS.append(header_ids)
HEADER_MASK.append(header_mask)
print("TOKEN_IDS", TOKEN_IDS)
print("SEGMENT_IDS", SEGMENT_IDS)
print("HEADER_IDS", HEADER_IDS)
print("HEADER_MASK", HEADER_MASK)

input_token_ids = np.array([[ 101,  158,  147,  123,  121,  122,  122, 1920,  754,  122,  122,
        2772, 5442,  147,  158,  161,  123,  121,  122,  122, 1920,  754,
         122,  122, 4638, 1062, 1385, 3300, 1525,  763,  102,   11, 6395,
        1171,  807, 4772,  102,   11, 1062, 1385, 1399, 4917,  102,   12,
        5500,  817,  102,   12,  147,  158,  161,  123,  121,  122,  122,
         102,   12,  147,  158,  161,  123,  121,  122,  123,  147,  102,
          12,  147,  158,  161,  123,  121,  122,  124,  147,  102,   12,
         158,  147,  123,  121,  122,  122,  102,   12,  158,  147,  123,
         121,  122,  123,  147,  102,   12,  158,  147,  123,  121,  122,
         124,  147,  102,   11,  156,  143,  164,  102,   11, 2835,  817,
        4372,  102,   12,  158,  144,  123,  121,  122,  123,  159,  122,
         102,   11, 6397, 5277,  102,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0]], dtype=np.int32)

input_segment_ids = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0]], dtype=np.int32)

input_header_ids = np.array([[ 31,  37,  43,  47,  56,  66,  76,  84,  93, 102, 107, 112, 122]],dtype=np.int32)
input_header_mask = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.int32)

TOKEN_IDS = input_token_ids
SEGMENT_IDS = input_segment_ids
HEADER_IDS = input_header_ids
HEADER_MASK = input_header_mask

print('input_token_ids', TOKEN_IDS.shape)
print('input_segment_ids',SEGMENT_IDS.shape)
print('input_header_ids',HEADER_IDS.shape)
print('input_header_mask',HEADER_MASK.shape)

inputs = {
            'input_token_ids': TOKEN_IDS,
            'input_segment_ids': SEGMENT_IDS,
            'input_header_ids': HEADER_IDS,
            'input_header_mask': HEADER_MASK
        }
col_test_data = inputs
import math
length = math.ceil(len(sample_data) / 1)
# 预测列
header_lens = np.sum(col_test_data['input_header_mask'], axis=-1)
print("header_lens", header_lens)
x = col_test_data
all_inputs = []


data = x
names = ['input_token_ids', 'input_segment_ids', 'input_header_ids', 'input_header_mask']
shapes = [(None, None), (None, None), (None, None), (None, None)]
check_batch_axis = False
exception_prefix = 'input'

# (1, 3)
preds_cond_conn_op = np.array([[1.1983177e-06,4.9303969e-07,9.9999833e-01]])
# (1, 13, 7)
preds_sel_agg = np.array([[[7.16833824e-07,3.62023798e-07,2.65719450e-07,7.21573315e-07,1.13928729e-06,1.23212573e-07,9.99996662e-01],
                  [9.99828815e-01,6.39063364e-05,1.14988688e-05,1.40415796e-06,7.46040751e-05,2.81184930e-06,1.69247105e-05],
                  [4.36230977e-07,2.23951730e-07,1.50546498e-07,5.10679499e-08,9.37693798e-08,4.82763518e-08,9.99999046e-01],
                  [2.54048100e-05,2.47787170e-06,8.77683851e-06,3.56288012e-07,8.61068656e-07,5.82684061e-06,9.99956250e-01],
                  [3.68819769e-07,1.54150555e-07,1.32718142e-07,4.81875766e-08,1.40918189e-07,5.96953740e-08,9.99999046e-01],
                  [1.84247722e-06,5.17945523e-07,3.40694356e-07,1.14017467e-07,3.65295620e-07,1.90229741e-07,9.99996662e-01],
                  [1.15100065e-05,3.54690224e-06,2.59059243e-06,2.91471679e-07,6.20083540e-07,5.68794121e-06,9.99975801e-01],
                  [3.93957578e-07,1.94025446e-07,1.07887359e-07,6.82555807e-08,1.54153938e-07,5.38957288e-08,9.99999046e-01],
                  [5.33017271e-07,3.02137437e-07,1.57079356e-07,8.27759479e-08,1.36090534e-07,7.40748760e-08,9.99998689e-01],
                  [6.50288996e-07,2.35490745e-07,1.26978136e-07,7.38663459e-08,1.56277565e-07,5.88117999e-08,9.99998689e-01],
                  [5.52429356e-07,1.64175063e-07,1.32449330e-07,5.23462447e-08,6.62986892e-08,4.26352749e-08,9.99999046e-01],
                  [4.40667350e-07,1.98678336e-07,1.66662474e-07,8.27671300e-08,1.34968033e-07,5.54668347e-08,9.99998927e-01],
                  [5.45189721e-07,1.18502676e-07,1.59347394e-07,1.77747921e-07,2.90192219e-07,3.66919686e-08,9.99998689e-01]]])
# (1, 13, 5)
preds_cond_op = np.array([[[2.2677888e-07,5.8867279e-08,3.1306499e-07,6.8794996e-08,9.9999928e-01],
                  [2.2290051e-05,2.4400779e-06,7.5301631e-07,1.7706209e-06,9.9997270e-01],
                  [1.0219911e-06,6.3037156e-08,4.8295224e-08,1.6780461e-08,9.9999881e-01],
                  [9.9995506e-01,3.4396915e-06,1.5627313e-07,1.9797794e-07,4.1114024e-05],
                  [9.2176460e-06,4.8621524e-07,6.4865276e-07,1.1451968e-07,9.9998951e-01],
                  [1.7748409e-05,4.7054596e-07,6.7188631e-07,1.5396469e-07,9.9998093e-01],
                  [9.9864596e-01,6.1760758e-05,8.0640751e-07,1.8719234e-06,1.2895485e-03],
                  [2.0592634e-06,1.7589734e-07,2.2239728e-07,6.0747055e-08,9.9999750e-01],
                  [1.7896641e-06,1.9021583e-07,1.0889695e-07,4.7046576e-08,9.9999785e-01],
                  [3.6893462e-07,3.3651791e-08,5.2707442e-08,1.5738170e-08,9.9999952e-01],
                  [3.9003353e-07,8.9794590e-08,1.8857607e-07,2.7236455e-08,9.9999928e-01],
                  [9.3939752e-07,7.5338910e-08,8.2398074e-08,2.7672797e-08,9.9999893e-01],
                  [2.7575601e-07,1.1639111e-07,6.1958832e-07,1.0967749e-07,9.9999893e-01]]])
print("preds_cond_conn_op", preds_cond_conn_op.shape, preds_cond_conn_op)
print("preds_sel_agg", preds_sel_agg.shape, preds_sel_agg)
print("preds_cond_op", preds_cond_op.shape, preds_cond_op)
# task1_result = outputs_to_sqls(preds_cond_conn_op, preds_sel_agg, preds_cond_op, header_lens,self.label_encoder)

preds_cond_conn_op, preds_sel_agg, preds_cond_op, header_lens = preds_cond_conn_op, preds_sel_agg, preds_cond_op, header_lens

preds_cond_conn_op = np.argmax(preds_cond_conn_op, axis=-1)
preds_cond_op = np.argmax(preds_cond_op, axis=-1)

print("preds_cond_conn_op", preds_cond_conn_op)
print("preds_cond_op", preds_cond_op)
sqls = []
for cond_conn_op, sel_agg, cond_op, header_len in zip(preds_cond_conn_op, preds_sel_agg, preds_cond_op, header_lens):
    print("cond_conn_op", cond_conn_op)
    print("sel_agg", sel_agg.shape, sel_agg)
    print("cond_op", cond_op)
    print("header_len", header_len)
    sel_agg = sel_agg[:header_len]



    # force to select at least one column for agg
    sel_agg[sel_agg == sel_agg[:, :-1].max()] = 1
    print("sel_agg", sel_agg.shape, sel_agg)
    sel_agg = np.argmax(sel_agg, axis=-1)
    print("sel_agg", sel_agg.shape, sel_agg)

    # sql = label_encoder.decode(cond_conn_op, sel_agg, cond_op)
    print("\ncond_conn_op", cond_conn_op)
    print("sel_agg", sel_agg.shape, sel_agg)
    print("cond_op", cond_op)
    cond_conn_op_label, sel_agg_label, cond_op_label = cond_conn_op, sel_agg, cond_op
    cond_conn_op = int(cond_conn_op_label)
    sel, agg, conds = [], [], []
    agg_sql_dict = {0: "", 1: "AVG", 2: "MAX", 3: "MIN", 4: "COUNT", 5: "SUM"}
    op_sql_dict = {0: ">", 1: "<", 2: "==", 3: "!="}
    for col_id, (agg_op, cond_op) in enumerate(zip(sel_agg_label, cond_op_label)):
        print("col_id, (agg_op, cond_op)", col_id, (agg_op, cond_op), len(agg_sql_dict), len(op_sql_dict))
        if agg_op < len(agg_sql_dict):
            sel.append(col_id)
            agg.append(int(agg_op))
        if cond_op < len(op_sql_dict):
            conds.append([col_id, int(cond_op)])
    print("sel, agg, conds", sel, agg, conds)

    sql = {'sel': sel,
            'agg': agg,
            'cond_conn_op': cond_conn_op,
            'conds': conds}
    print("sql", sql)
    sql['conds'] = [cond for cond in sql['conds'] if cond[0] < header_len]
    print("sql", sql)

    sel, agg = [], []
    for col_id, agg_op in zip(sql['sel'], sql['agg']):
        if col_id < header_len:
            sel.append(col_id)
            agg.append(agg_op)
    print("sql", sql)
    #
    sql['sel'] = sel
    sql['agg'] = agg
    sqls.append(sql)
    print("sql", sql)
print("sqls", sqls)
task1_result = sqls
print('task1_result:{}'.format(task1_result))


value_result = np.array([[1.4364719e-05],
 [2.5033951e-06],
 [1.3113022e-06],
 [9.9494243e-01],
 [4.4196844e-04],
 [4.6789646e-06],
 [2.5033951e-06],
 [7.1525574e-07],
 [6.5932534e-07],
 [9.9116206e-01],
 [1.0676298e-04],
 [9.6113297e-07]])
print('value_result:{}'.format(value_result))