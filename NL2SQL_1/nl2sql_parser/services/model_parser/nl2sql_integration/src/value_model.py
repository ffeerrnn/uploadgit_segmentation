# _*_coding:utf-8_*_
# 作者： 万方名
# 创建日期： 2020/8/23   14:33
# 文件： value_model.py
import os
from keras_bert import get_checkpoint_paths, load_vocabulary, Tokenizer, load_trained_model_from_checkpoint

from keras.layers import Input, Lambda, Dense
from keras.models import Model
from keras.optimizers import Adam

from conf.model_conf import *
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class SimpleTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')
            else:
                R.append('[UNK]')
        return R


class ValueModel(object):
    def __init__(self):
        self.model_path = value_model_path
        self.bert_model_path = bert_model_path

    def get_value_model(self):
        paths = get_checkpoint_paths(self.bert_model_path)
        token_dict = load_vocabulary(paths.vocab)
        tokenizer = SimpleTokenizer(token_dict)

        bert_model = load_trained_model_from_checkpoint(
            paths.config, paths.checkpoint, seq_len=None)
        for l in bert_model.layers:
            l.trainable = True

        x1_in = Input(shape=(None,), name='input_x1', dtype='int32')
        x2_in = Input(shape=(None,), name='input_x2')
        x = bert_model([x1_in, x2_in])
        x_cls = Lambda(lambda x: x[:, 0])(x)
        y_pred = Dense(1, activation='sigmoid', name='output_similarity')(x_cls)

        model = Model([x1_in, x2_in], y_pred)

        model.compile(loss={'output_similarity': 'binary_crossentropy'},
                      optimizer=Adam(1e-5),
                      metrics={'output_similarity': 'accuracy'})
        model.load_weights(self.model_path)
        return model, tokenizer
