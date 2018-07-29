# !/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score, log_loss
import time
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from DataParse import DataParse
from tqdm import tqdm
np.random.seed(2018)

class DCN(BaseEstimator, TransformerMixin):
    def __init__(self, category_size,
                 category_feature_size,
                 continuous_feature_size,
                 embedding_size=8,
                 dropout_fm=[1.0, 1.0],
                 deep_layers=[32, 32],
                 cross_layer_num = 3,
                 dropout_deep=[1.0, 1.0, 1.0],
                 deep_layers_activation=tf.nn.relu,
                 epochs=10,
                 batch_size=128,
                 learning_rate=0.001,
                 optimizer_type='adam',
                 batch_norm=0,
                 batch_norm_decay=0.995,
                 verbose=True,
                 random_seed=2016,
                 loss_type='logloss',
                 metric_type='auc',
                 l2_reg=0.0):
        self.category_size = category_size
        self.category_feature_size = category_feature_size
        self.continuous_feature_size = continuous_feature_size
        self.embedding_size = embedding_size
        self.dropout_fm = dropout_fm
        self.deep_layers = deep_layers
        self.cross_layer_num = cross_layer_num
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
        self.l2_reg = l2_reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay
        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.metric_type = metric_type

        self.field_size = self.category_feature_size + self.continuous_feature_size
        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.cate_index = tf.placeholder(tf.int32, [None, self.category_feature_size], name='category_index')
            self.cont_feats = tf.placeholder(tf.float32, [None, self.continuous_feature_size], name='continuous_feature')
            self.feature_value = tf.placeholder(tf.float32, [None, self.field_size], name='feature_value')
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name='label')
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name='dropout_keep_deep')
            self.train_phase = tf.placeholder(tf.bool, name='train_phase')

            weights = {}
            biases = {}

            with tf.name_scope('Embedding_Layer'):
                weights['category_embedding'] = tf.Variable(tf.random_normal([self.category_size, self.embedding_size], mean=0.0, stddev=0.01))
                # category -> Embedding
                self.embedding = tf.nn.embedding_lookup(weights['category_embedding'], self.cate_index) # [None, category_feature_size, embedding_size]
                self.embedding = tf.reshape(self.embedding, shape=[-1, self.category_feature_size * self.embedding_size]) # [None, category_feature_size*embedding_size]
                # concat Embedding Vector & continuous -> Dense Vector
                self.x0 = tf.concat([self.embedding, self.cont_feats], axis=1)  # [None, category_feature_size*embedding_size + continuous_feature_size]

            with tf.name_scope('deep_network'):
                num_layer = len(self.deep_layers)
                input_size = self.x0.shape.as_list()[1]
                glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))
                weights['deep_layer_0'] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])),
                    dtype=np.float32)
                biases['deep_layer_bias_0'] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])),
                    dtype=np.float32)

                for i in range(1, num_layer):
                    glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
                    weights['deep_layer_%s' % i] = tf.Variable(
                        np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i - 1], self.deep_layers[i])),
                        dtype=np.float32)
                    biases['deep_layer_bias_%s' % i] = tf.Variable(
                        np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                        dtype=np.float32)

                self.deep_out = self.x0
                for i in range(0, num_layer):
                    self.deep_out = tf.add(tf.matmul(self.deep_out, weights['deep_layer_%s' % i]),
                                           biases['deep_layer_bias_%s' % i])
                    if self.batch_norm:
                        self.deep_out = self.batch_norm_layer(self.deep_out, train_phase=self.train_phase,
                                                              scope_bn='bn_%s' % i)
                    self.deep_out = self.deep_layers_activation(self.deep_out)
                    self.deep_out = tf.nn.dropout(self.deep_out, self.dropout_keep_deep[i + 1])

            with tf.name_scope('cross_network'):
                input_size = self.x0.shape.as_list()[1]
                glorot = np.sqrt(2.0 / (input_size + input_size))
                for i in range(self.cross_layer_num):
                    weights['cross_layer_%s' % i] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(input_size, 1)), dtype=np.float32)
                    biases['cross_bias_%s' % i] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(input_size, 1)), dtype=np.float32)

                self.xl = tf.reshape(self.x0, shape=[-1, input_size, 1])
                self._x0 = self.xl
                for i in range(self.cross_layer_num):
                    self.xl = tf.tensordot(tf.matmul(self._x0, self.xl, transpose_b=True),
                                           weights['cross_layer_%s' % i], axes=1) + biases['cross_bias_%s' % i] + self.xl # [None, category_feature_size*embedding_size + continuous_feature_size, 1]

                self.cross_out = tf.reshape(self.xl, shape=[-1, input_size])

            with tf.name_scope('DCN_out'):
                input_size = self.cross_out.shape.as_list()[1] + self.deep_layers[-1]
                glorot = np.sqrt(2.0 / (input_size + 1))
                weights['concat_projection'] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(input_size, 1)), dtype=np.float32)
                biases['concat_bias'] = tf.Variable(tf.constant(0.01), dtype=np.float32)
                self.out = tf.concat([self.cross_out, self.deep_out], axis=1)
                self.out = tf.add(tf.matmul(self.out, weights['concat_projection']), biases['concat_bias'])

            # loss
            if self.loss_type == 'logloss':
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)

            elif self.loss_type == 'mse':
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))

            # l2 regularization on weights
            if self.l2_reg > 0:
                self.loss = self.loss + tf.contrib.layers.l2_regularizer(self.l2_reg)(weights['concat_projection'])
                for i in range(len(self.deep_layers)):
                    self.loss = self.loss + tf.contrib.layers.l2_regularizer(self.l2_reg)(
                        weights['deep_layer_%s' % i]
                    )
                for i in range(self.cross_layer_num):
                    self.loss = self.loss + tf.contrib.layers.l2_regularizer(self.l2_reg)(
                        weights['cross_layer_%s' % i]
                    )
            # optimizer
            if self.optimizer_type == 'adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

            elif self.optimizer_type == 'adagrad':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)

            elif self.optimizer_type == 'gd':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(
                    self.loss)

            elif self.optimizer_type == 'momentum':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                            momentum=0.95).minimize(
                    self.loss)

            elif self.optimizer_type == 'rmsprop':
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def fit(self, category_index, continuous_val, feat_val, label, valid_category_index=None, valid_continuous_val=None, valid_feat_val=None, valid_label=None):
        '''
        category_index for embedding
        concat category_embedding & continuous -> Dense input

        :param category_index: [[idx1_1, idx1_2,...], [idx2_1, idx2_2,...],...]
                                 idxi_j is the category index of category field j of sample i in the training set
        :param continuous_val: [[con_value1_1, con_value1_1,...], [con_value2_1, con_value2_2,...], ...]
                                 con_valuei_j is the continuous value of continuous field j of sample i in the training set
        :param feat_val: [[value1_1, value1_2,...], [value2_1, value2_2,...]...]
                           valuei_j is the feature value of feature field j of sample i in the training set
        :param label: [[label1], [label2], [label3], [label4],...]
        :param valid_feat_val: list of list of feature indices of each sample in the validation set
        :param valid_label: label of each sample in the validation set
        :param valid_category_index: list of list of feature indices of each sample in the validation set
        :return: None
        '''
        has_valid = valid_category_index is not None
        total_time = 0
        for epoch in range(self.epochs):
            start_time = time.time()
            for i in tqdm(range(0, len(category_index), self.batch_size), ncols=100):
                cate_index_batch = category_index[i: i + self.batch_size]
                cont_val_batch = continuous_val[i: i + self.batch_size]
                feat_val_batch = feat_val[i: i + self.batch_size]
                batch_y = label[i: i + self.batch_size]

                feed_dict = {
                    self.cate_index: cate_index_batch,
                    self.cont_feats: cont_val_batch,
                    self.feature_value: feat_val_batch,
                    self.label: batch_y,
                    self.dropout_keep_deep: self.dropout_deep,
                    self.train_phase: True
                }
                cost, opt = self.sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
            train_metric = self.evaluate(category_index, continuous_val, feat_val, label)
            end_time = time.time()
            if self.verbose:
                if has_valid:
                    valid_metric = self.evaluate(valid_category_index, valid_continuous_val, valid_feat_val, valid_label)
                    print('[%s] train-%s=%.4f, valid-%s=%.4f [%.1f s]' % (
                        epoch + 1, self.metric_type, train_metric, self.metric_type, valid_metric,
                        end_time - start_time))
                else:
                    print('[%s] train-%s=%.4f [%.1f s]' % (
                        epoch + 1, self.metric_type, train_metric, end_time - start_time))
            total_time = total_time + end_time - start_time

        print('cost total time=%.1f s' % total_time)

    def predict(self, category_index, continuous_val, feat_val):
        feed_dict = {
            self.cate_index: category_index,
            self.cont_feats: continuous_val,
            self.feature_value: feat_val,
            self.dropout_keep_deep: [1.0] * len(self.dropout_deep),
            self.train_phase: False
        }
        y_pred = self.sess.run(self.out, feed_dict=feed_dict)
        return y_pred

    def evaluate(self, category_index, continuous_val, feat_val, label):
        y_pred = self.predict(category_index, continuous_val, feat_val)
        if self.metric_type == 'auc':
            return roc_auc_score(label, y_pred)
        elif self.metric_type == 'logloss':
            return log_loss(label, y_pred)

if __name__ == '__main__':
    print('read dataset...')
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    y_train = pd.read_csv('data/y_train.csv')
    y_val = pd.read_csv('data/y_val.csv')

    continuous_feature = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    category_feature = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex',
                        'native_country']

    dataParse = DataParse(continuous_feature=continuous_feature, category_feature=category_feature)
    dataParse.FeatureDictionary(train, test)
    train_cat_index = dataParse.parse(train)
    test_cat_index = dataParse.parse(test)

    y_train = y_train.values.reshape(-1, 1)
    y_val = y_val.values.reshape(-1, 1)

    model = DCN(category_size=dataParse.category_size,
                category_feature_size=len(category_feature),
                continuous_feature_size=len(continuous_feature))

    model.fit(train_cat_index, train[continuous_feature], train, y_train)
    test_metric = model.evaluate(test_cat_index, test[continuous_feature], test, y_val)
    print('test-auc=%.4f' % test_metric)
