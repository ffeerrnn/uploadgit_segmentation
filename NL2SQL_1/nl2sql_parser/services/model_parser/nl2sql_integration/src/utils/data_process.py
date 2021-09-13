# _*_coding:utf-8_*_
# 作者： 万方名
# 创建日期： 2020/8/21   11:00
# 文件： data_process.py

import re
import math
import cn2an
import numpy as np
from collections import defaultdict

from keras.preprocessing.sequence import pad_sequences
from keras.utils.data_utils import Sequence

from nl2sql_parser.services.model_parser.nl2sql_integration.src.utils import SQL, MultiSentenceTokenizer


def str_to_year(string):
    year = string.replace('年', '')
    year = cn_to_an(year)
    if is_float(year) and float(year) < 1900:
        year = int(year) + 2000
        return str(year)
    else:
        return None


def remove_brackets(s):
    # 去括号
    return re.sub(r'[\(\（].*[\)\）]', '', s)


def an_to_cn(string):
    try:
        return str(cn2an.an2cn(string))
    except ValueError:
        return string


def cn_to_an(string):
    try:
        return str(cn2an.cn2an(string, 'normal'))
    except ValueError:
        return string


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def str_to_num(string):
    try:
        float_val = float(cn_to_an(string))
        if int(float_val) == float_val:
            return str(int(float_val))
        else:
            return str(float_val)
    except ValueError:
        return None


class DataProcessCol(Sequence):
    """
    Generate training data in batches

    """

    def __init__(self,
                 data,
                 tokenizer,
                 max_len=160,
                 batch_size=1,
                 shuffle=True,
                 shuffle_header=True,
                 global_indices=None):

        self.data = data
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        self.shuffle_header = shuffle_header
        self.max_len = max_len

        if global_indices is None:
            self._global_indices = np.arange(len(data))
        else:
            self._global_indices = global_indices

        if shuffle:
            np.random.shuffle(self._global_indices)

    def get_header(self, names, types):
        return ' | '.join(['{}({})'.format(n, t) for n, t in zip(names, types)])

    def _pad_sequences(self, seqs, max_len=None):
        padded = pad_sequences(seqs, maxlen=max_len, padding='post', truncating='post')
        if max_len is not None:
            padded = padded[:, :max_len]
        return padded

    def __getitem__(self, batch_id):
        TOKEN_IDS, SEGMENT_IDS = [], []
        HEADER_IDS, HEADER_MASK = [], []

        question = self.data[0]

        col_names = list(self.data[1].keys())
        col_types = list(self.data[1].values())

        header = self.get_header(col_names, col_types)

        col_orders = np.arange(len(self.data[1]))
        if self.shuffle_header:    # 随机打乱
            np.random.shuffle(col_orders)

        token_ids, segment_ids, header_ids = self.tokenizer.encode(question, col_names, col_types, col_orders)
        print("token_ids", token_ids)
        print("segment_ids", segment_ids)
        print("header_ids", header_ids)

        header_ids = [hid for hid in header_ids if hid < self.max_len]
        header_mask = [1] * len(header_ids)
        col_orders = col_orders[: len(header_ids)]

        TOKEN_IDS.append(token_ids)
        SEGMENT_IDS.append(segment_ids)
        HEADER_IDS.append(header_ids)
        HEADER_MASK.append(header_mask)

        TOKEN_IDS = self._pad_sequences(TOKEN_IDS, max_len=self.max_len)
        SEGMENT_IDS = self._pad_sequences(SEGMENT_IDS, max_len=self.max_len)
        HEADER_IDS = self._pad_sequences(HEADER_IDS)
        HEADER_MASK = self._pad_sequences(HEADER_MASK)


        inputs = {
            'input_token_ids': TOKEN_IDS,
            'input_segment_ids': SEGMENT_IDS,
            'input_header_ids': HEADER_IDS,
            'input_header_mask': HEADER_MASK
        }

        print("inputs", inputs)
        return inputs

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self._global_indices)


class CandidateCondsExtractor:
    """
    params:
        - share_candidates: 在同 table 同 column 中共享 real 型 candidates
    """
    CN_NUM = '〇一二三四五六七八九零壹贰叁肆伍陆柒捌玖貮两'
    CN_UNIT = '十拾百佰千仟万萬亿億兆点'

    def __init__(self, share_candidates=True):
        self.share_candidates = share_candidates
        self._cached = False

    def build_candidate_cache(self, queries):
        self.cache = defaultdict(set)
        print('building candidate cache')
        value_in_question = self.extract_values_from_text(queries[0])
        names = list(queries[1].keys())
        types = list(queries[1].values())
        cond_values_list = []

        for col_id, (col_name, col_type) in enumerate(zip(names, types)):
            value_in_column = self.extract_values_from_column(queries, col_id)
            if col_type == 'text':
                cond_values = value_in_column
            elif col_type == 'real':
                if len(value_in_column) == 1:
                    cond_values = value_in_column + value_in_question
                else:
                    cond_values = value_in_question
            cond_values_list.append(cond_values)
        self._cached = True
        return cond_values_list

    def get_cache_key(self, query_id, query, col_id):
        if self.share_candidates:
            return (query.table.id, col_id)
        else:
            return (query_id, query.table.id, col_id)

    def extract_year_from_text(self, text):
        values = []
        num_year_texts = re.findall(r'[0-9][0-9]年', text)
        values += ['20{}'.format(text[:-1]) for text in num_year_texts]
        cn_year_texts = re.findall(r'[{}][{}]年'.format(self.CN_NUM, self.CN_NUM), text)
        cn_year_values = [str_to_year(text) for text in cn_year_texts]
        values += [value for value in cn_year_values if value is not None]
        return values

    def extract_num_from_text(self, text):
        values = []
        num_values = re.findall(r'[-+]?[0-9]*\.?[0-9]+', text)
        values += num_values

        cn_num_unit = self.CN_NUM + self.CN_UNIT
        cn_num_texts = re.findall(r'[{}]*\.?[{}]+'.format(cn_num_unit, cn_num_unit), text)
        cn_num_values = [str_to_num(text) for text in cn_num_texts]
        values += [value for value in cn_num_values if value is not None]

        cn_num_mix = re.findall(r'[0-9]*\.?[{}]+'.format(self.CN_UNIT), text)
        for word in cn_num_mix:
            num = re.findall(r'[-+]?[0-9]*\.?[0-9]+', word)
            for n in num:
                word = word.replace(n, an_to_cn(n))
            str_num = str_to_num(word)
            if str_num is not None:
                values.append(str_num)
        return values

    def extract_values_from_text(self, text):
        values = []
        values += self.extract_year_from_text(text)
        values += self.extract_num_from_text(text)
        return list(set(values))

    def extract_values_from_column(self, querys, col_ids):
        question = querys[0]
        question_chars = set(question)
        unique_col_values = set(list(querys[2].values())[col_ids])
        select_col_values = [v for v in unique_col_values
                             if (question_chars & set(str(v)))]
        return select_col_values


class FullSampler:
    """
    不抽样，返回所有的 pairs

    """

    def sample(self, data):
        return data


class DataProcessValue:
    """
    question - cond pairs 数据集
    """
    OP_PATTERN = {
        'real':
            [
                {'cond_op_idx': 0, 'pattern': '{col_name}大于{value}'},
                {'cond_op_idx': 1, 'pattern': '{col_name}小于{value}'},
                {'cond_op_idx': 2, 'pattern': '{col_name}是{value}'}
            ],
        'text':
            [
                {'cond_op_idx': 2, 'pattern': '{col_name}是{value}'}
            ]
    }

    def __init__(self, queries, candidate_extractor, has_label=True, model_1_outputs=None):
        self.candidate_extractor = candidate_extractor
        self.has_label = has_label
        self.model_1_outputs = model_1_outputs
        self.data = self.build_dataset(queries)

    def build_dataset(self, queries):
        if not self.candidate_extractor._cached:
            self.candidate_extractor.build_candidate_cache(queries)
        names = list(queries[1].keys())
        types = list(queries[1].values())

        pair_data = []

        select_col_id = self.get_select_col_id(0)
        for col_id, (col_name, col_type) in enumerate(zip(names, types)):
            if col_id not in select_col_id:
                continue

            values = self.candidate_extractor.build_candidate_cache(queries)[col_id]
            pattern = self.OP_PATTERN.get(col_type, [])
            pairs = self.generate_pairs(queries[0], col_id, col_name, values, pattern)
            pair_data += pairs
        return pair_data

    def get_select_col_id(self, query_id):
        if self.model_1_outputs:
            select_col_id = [cond_col for cond_col, *_ in self.model_1_outputs[query_id]['conds']]
        else:
            print('无model_1_outputs')
        return select_col_id

    def generate_pairs(self, queston, col_id, col_name, values, op_patterns):
        pairs = []
        for value in values:
            for op_pattern in op_patterns:
                cond = op_pattern['pattern'].format(col_name=col_name, value=value)
                cond_sql = (col_id, op_pattern['cond_op_idx'], value)
                real_sql = {}

                label = 1 if cond_sql in real_sql else 0
                pair = QuestionCondPair(0, queston,
                                        cond, cond_sql, label)
                pairs.append(pair)
        return pairs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataProcessValueSeq(Sequence):
    def __init__(self, dataset, tokenizer, is_train=True, max_len=120,
                 sampler=None, shuffle=False, batch_size=32):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.max_len = max_len
        self.sampler = sampler
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.on_epoch_end()

    def _pad_sequences(self, seqs, max_len=None):
        return pad_sequences(seqs, maxlen=max_len, padding='post', truncating='post')

    def __getitem__(self, batch_id):
        batch_data_indices = self.global_indices[batch_id * self.batch_size: (batch_id + 1) * self.batch_size]
        batch_data = [self.data[i] for i in batch_data_indices]

        X1, X2 = [], []
        Y = []

        for data in batch_data:
            x1, x2 = self.tokenizer.encode(first=data.question.lower(),
                                           second=data.cond_text.lower())
            X1.append(x1)
            X2.append(x2)
            if self.is_train:
                Y.append([data.label])

        X1 = self._pad_sequences(X1, max_len=self.max_len)
        X2 = self._pad_sequences(X2, max_len=self.max_len)
        inputs = {'input_x1': X1, 'input_x2': X2}
        if self.is_train:
            Y = self._pad_sequences(Y, max_len=1)
            outputs = {'output_similarity': Y}
            return inputs, outputs
        else:
            return inputs

    def on_epoch_end(self):
        self.data = self.sampler.sample(self.dataset)
        self.global_indices = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.global_indices)

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)


class SqlLabelEncoder:
    """
    Convert SQL object into training labels.
    """

    def encode(self, sql: SQL, num_cols):
        cond_conn_op_label = sql.cond_conn_op

        sel_agg_label = np.ones(num_cols, dtype='int32') * len(SQL.agg_sql_dict)
        for col_id, agg_op in zip(sql.sel, sql.agg):
            if col_id < num_cols:
                sel_agg_label[col_id] = agg_op

        cond_op_label = np.ones(num_cols, dtype='int32') * len(SQL.op_sql_dict)
        for col_id, cond_op, _ in sql.conds:
            if col_id < num_cols:
                cond_op_label[col_id] = cond_op

        return cond_conn_op_label, sel_agg_label, cond_op_label

    def decode(self, cond_conn_op_label, sel_agg_label, cond_op_label):
        cond_conn_op = int(cond_conn_op_label)
        sel, agg, conds = [], [], []

        for col_id, (agg_op, cond_op) in enumerate(zip(sel_agg_label, cond_op_label)):
            if agg_op < len(SQL.agg_sql_dict):
                sel.append(col_id)
                agg.append(int(agg_op))
            if cond_op < len(SQL.op_sql_dict):
                conds.append([col_id, int(cond_op)])
        return {
            'sel': sel,
            'agg': agg,
            'cond_conn_op': cond_conn_op,
            'conds': conds
        }


class QueryTokenizer(MultiSentenceTokenizer):
    """
    Tokenize query (question + table header) and encode to integer sequence.
    Using reserved tokens [unused11] and [unused12] for classification
    """
    col_type_token_dict = {'text': '[unused11]', 'real': '[unused12]'}

    def tokenize(self, question, col_names, col_types, col_orders=None):
        """
        Tokenize quesiton and columns and concatenate.

        Parameters:
        query (Query): A query object contains question and table
        col_orders (list or numpy.array): For re-ordering the header columns

        Returns:
        token_idss: token ids for bert encoder
        segment_ids: segment ids for bert encoder
        header_ids: positions of columns
        """
        print("self._token_cls", self._token_cls)
        question_tokens = [self._token_cls] + self._tokenize(question)
        print("question_tokens", question_tokens)

        header_tokens = []

        for col_name, col_type in zip(col_names, col_types):
            print("col_name, col_type", col_name, col_type)
            col_type_token = self.col_type_token_dict[col_type]
            col_name = remove_brackets(col_name)
            col_name_tokens = self._tokenize(col_name)
            col_tokens = [col_type_token] + col_name_tokens
            header_tokens.append(col_tokens)

        all_tokens = [question_tokens] + header_tokens
        return self._pack(*all_tokens)

    def encode(self, question, col_names, col_types, col_orders=None):
        tokens, tokens_lens = self.tokenize(question, col_names, col_types, col_orders)
        token_ids = self._convert_tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        header_indices = np.cumsum(tokens_lens)    # 累加
        return token_ids, segment_ids, header_indices[:-1]


class QuestionCondPair:
    def __init__(self, query_id, question, cond_text, cond_sql, label):
        self.query_id = query_id
        self.question = question
        self.cond_text = cond_text
        self.cond_sql = cond_sql
        self.label = label

    def __repr__(self):
        repr_str = ''
        repr_str += 'query_id: {}\n'.format(self.query_id)
        repr_str += 'question: {}\n'.format(self.question)
        repr_str += 'cond_text: {}\n'.format(self.cond_text)
        repr_str += 'cond_sql: {}\n'.format(self.cond_sql)
        repr_str += 'label: {}\n'.format(self.label)
        return repr_str


class QuestionCondPairsDataset:
    """
    question - cond pairs 数据集
    """
    OP_PATTERN = {
        'real':
            [
                {'cond_op_idx': 0, 'pattern': '{col_name}大于{value}'},
                {'cond_op_idx': 1, 'pattern': '{col_name}小于{value}'},
                {'cond_op_idx': 2, 'pattern': '{col_name}是{value}'}
            ],
        'text':
            [
                {'cond_op_idx': 2, 'pattern': '{col_name}是{value}'}
            ]
    }

    def __init__(self, queries, candidate_extractor, has_label=True, model_1_outputs=None):
        self.candidate_extractor = candidate_extractor
        self.has_label = has_label
        self.model_1_outputs = model_1_outputs
        self.data = self.build_dataset(queries)

    def build_dataset(self, queries):
        if not self.candidate_extractor._cached:
            self.candidate_extractor.build_candidate_cache(queries)

        pair_data = []
        for query_id, query in enumerate(queries):
            select_col_id = self.get_select_col_id(query_id, query)
            for col_id, (col_name, col_type) in enumerate(query.table.header):
                if col_id not in select_col_id:
                    continue

                cache_key = self.candidate_extractor.get_cache_key(query_id, query, col_id)
                values = self.candidate_extractor.cache.get(cache_key, [])
                pattern = self.OP_PATTERN.get(col_type, [])
                pairs = self.generate_pairs(query_id, query, col_id, col_name,
                                            values, pattern)
                pair_data += pairs
        return pair_data

    def get_select_col_id(self, query_id, query):
        if self.model_1_outputs:
            select_col_id = [cond_col for cond_col, *_ in self.model_1_outputs[query_id]['conds']]
        elif self.has_label:
            select_col_id = [cond_col for cond_col, *_ in query.sql.conds]
        else:
            select_col_id = list(range(len(query.table.header)))
        return select_col_id

    def generate_pairs(self, query_id, query, col_id, col_name, values, op_patterns):
        pairs = []
        for value in values:
            for op_pattern in op_patterns:
                cond = op_pattern['pattern'].format(col_name=col_name, value=value)
                cond_sql = (col_id, op_pattern['cond_op_idx'], value)
                real_sql = {}
                if self.has_label:
                    real_sql = {tuple(c) for c in query.sql.conds}
                label = 1 if cond_sql in real_sql else 0
                pair = QuestionCondPair(query_id, query.question.text,
                                        cond, cond_sql, label)
                pairs.append(pair)
        return pairs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
