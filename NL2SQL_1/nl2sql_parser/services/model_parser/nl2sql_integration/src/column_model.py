# _*_coding:utf-8_*_
# 作者： 万方名
# 创建日期： 2020/8/23   11:08
# 文件： column_model.py
import os
import keras.backend as K
from keras_bert import load_trained_model_from_checkpoint, get_checkpoint_paths
from keras.layers import Input, Dense, Lambda, Multiply, Masking, Concatenate
from keras.models import Model
from nl2sql_parser.services.model_parser.nl2sql_integration.src.utils.optimizer import RAdam

from conf.model_conf import *

os.environ["CUDA_VISIBLE_DEVICES"] = ""
bert_model_path = "../bert_model/chinese_wwm_L-12_H-768_A-12/"
col_model_path = "../data/model/task1_best_model_2020_8_20.h5"

class ColumnModel(object):
    def __init__(self):
        self.op_sql_dict = {0: ">", 1: "<", 2: "==", 3: "!="}
        self.agg_sql_dict = {0: "", 1: "AVG", 2: "MAX", 3: "MIN", 4: "COUNT", 5: "SUM"}
        self.conn_sql_dict = {0: "", 1: "and", 2: "or"}
        self.bert_model_path = bert_model_path

    def seq_gather(self, x):
        seq, idxs = x
        idxs = K.cast(idxs, 'int32')
        return K.tf.batch_gather(seq, idxs)

    def get_col_model(self):
        self.paths = get_checkpoint_paths(self.bert_model_path)
        self.bert_model = load_trained_model_from_checkpoint(self.paths.config, self.paths.checkpoint, seq_len=None)

        # output sizes
        num_sel_agg = len(self.agg_sql_dict) + 1
        num_cond_op = len(self.op_sql_dict) + 1
        num_cond_conn_op = len(self.conn_sql_dict)
        for l in self.bert_model.layers:
            l.trainable = True

        inp_token_ids = Input(shape=(None,), name='input_token_ids', dtype='int32')
        inp_segment_ids = Input(shape=(None,), name='input_segment_ids', dtype='int32')
        inp_header_ids = Input(shape=(None,), name='input_header_ids', dtype='int32')
        inp_header_mask = Input(shape=(None,), name='input_header_mask')

        x = self.bert_model([inp_token_ids, inp_segment_ids])  # (None, seq_len, 768)

        # predict cond_conn_op
        x_for_cond_conn_op = Lambda(lambda x: x[:, 0])(x)  # (None, 768)
        p_cond_conn_op = Dense(num_cond_conn_op, activation='softmax', name='output_cond_conn_op')(x_for_cond_conn_op)

        # predict sel_agg
        x_for_header = Lambda(self.seq_gather, name='header_seq_gather')([x, inp_header_ids])  # (None, h_len, 768)
        header_mask = Lambda(lambda x: K.expand_dims(x, axis=-1))(inp_header_mask)  # (None, h_len, 1)

        x_for_header = Multiply()([x_for_header, header_mask])
        x_for_header = Masking()(x_for_header)

        p_sel_agg = Dense(num_sel_agg, activation='softmax', name='output_sel_agg')(x_for_header)

        x_for_cond_op = Concatenate(axis=-1)([x_for_header, p_sel_agg])
        p_cond_op = Dense(num_cond_op, activation='softmax', name='output_cond_op')(x_for_cond_op)

        column_model = Model(
            [inp_token_ids, inp_segment_ids, inp_header_ids, inp_header_mask],
            [p_cond_conn_op, p_sel_agg, p_cond_op]
        )

        learning_rate = 1e-5

        column_model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=RAdam(lr=learning_rate)
        )

        column_model.load_weights(col_model_path)
        return column_model
