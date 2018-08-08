# !/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
import time
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from DataParse import DataParse
from tqdm import tqdm
np.random.seed(2018)

class PNN(BaseEstimator, TransformerMixin):
    def __init__(self, feature_size,
                 field_size,
                 embedding_size=8,
                 deep_init_size=50,
                 deep_layers=[32, 32],
                 dropout_deep=[1.0, 1.0, 1.0],
                 deep_layers_activation=tf.nn.relu,
                 epochs=10,
                 batch_size=128,
                 learning_rate=0.001,
                 optimizer_type='adam',
                 batch_norm=False,
                 batch_norm_decay=0.995,
                 verbose=True,
                 random_seed=2018,
                 loss_type='logloss',
                 metric_type='auc',
                 l2_reg=0.0,
                 use_inner=True):
        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.deep_init_size = deep_init_size
        self.deep_layers = deep_layers
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
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
        self.l2_reg = l2_reg
        self.use_inner = use_inner

        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.feature_index = tf.placeholder(tf.int32, [None, self.field_size], name='feature_index')
            self.feature_value = tf.placeholder(tf.float32, [None, self.field_size], name='feature_value')
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name='label')
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name='dropout_keep_deep')
            self.train_phase = tf.placeholder(tf.bool, name='train_phase')

            weights = {}
            biases = {}

            with tf.name_scope('Embedding_Layer'):
                weights['feature_embeddings'] = tf.Variable(tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01), name='feature_embeddings')
                self.embeddings = tf.nn.embedding_lookup(weights['feature_embeddings'], self.feature_index)  # [None, field_size, embedding_size]
                feat_value = tf.reshape(self.feature_value, shape=[-1, self.field_size, 1])  # [None, field_size, 1]
                self.embeddings = tf.multiply(self.embeddings, feat_value)  # [None, field_size, embedding_size]

            with tf.name_scope('Product_Layer'):
                if self.use_inner:
                    weights['product-quadratic-inner'] = tf.Variable(tf.random_normal([self.deep_init_size, self.field_size], 0.0, 0.01))
                else:
                    weights['product-quadratic-outer'] = tf.Variable(tf.random_normal([self.deep_init_size, self.embedding_size, self.embedding_size], 0.0, 0.01))
                weights['product-linear'] = tf.Variable(tf.random_normal([self.deep_init_size, self.field_size, self.embedding_size], 0.0, 0.01))
                biases['product-bias'] = tf.Variable(tf.random_normal([self.deep_init_size],0,0,1.0))

                # linear signals z
                linear_output = []
                for i in range(self.deep_init_size):
                    self.linear_product = tf.multiply(self.embeddings, weights['product-linear'][i])
                    self.linear_product = tf.reduce_sum(self.linear_product, axis=[1, 2])
                    self.linear_product = tf.reshape(self.linear_product, shape=[-1, 1])
                    linear_output.append(self.linear_product)

                self.lz = tf.concat(linear_output, axis=1)  # [N, deep_init_size]

                # quadratic signals p
                quadratic_output = []
                if self.use_inner:
                    for i in range(self.deep_init_size):
                        theta = tf.multiply(self.embeddings, tf.reshape(weights['product-quadratic-inner'][i], [1, -1, 1])) # [None, field_size, embedding_size]
                        quadratic_output.append(tf.reshape(tf.norm(tf.reduce_sum(theta, axis=1), axis=1), shape=[-1, 1]))  # [None, 1]
                else:
                    embedding_sum = tf.reduce_sum(self.embeddings, axis=1)
                    p = tf.matmul(tf.expand_dims(embedding_sum, 2), tf.expand_dims(embedding_sum, 1)) # [None, embedding_size, embedding_size]
                    for i in range(self.deep_init_size):
                        theta = tf.multiply(p, tf.expand_dims(weights['product-quadratic-outer'][i], 0))  # [None, embedding_size, embedding_size]
                        quadratic_output.append(tf.reshape(tf.reduce_sum(theta, axis=[1, 2]), shape=(-1, 1)))  # [None, 1]

                self.lp = tf.concat(quadratic_output, axis=1)  # [N, deep_init_size]
                z_add_p = tf.add(self.lp, self.lz)
                self.deep_out = tf.nn.relu(tf.add(z_add_p, biases['product-bias']))
                self.deep_out = tf.nn.dropout(self.deep_out, self.dropout_keep_deep[0]) # [N, deep_init_size]

                # z_p = tf.concat([self.lp, self.lz], axis=1) # z与p进行concat
                # self.deep_out = tf.nn.dropout(z_p, self.dropout_keep_deep[0])

            with tf.name_scope('Hidden_Layer'):
                num_layer = len(self.deep_layers)
                input_size = self.deep_out.shape.as_list()[1]
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

                for i in range(0, num_layer):
                    self.deep_out = tf.add(tf.matmul(self.deep_out, weights['deep_layer_%s' % i]),
                                           biases['deep_layer_bias_%s' % i])
                    if self.batch_norm:
                        self.deep_out = self.batch_norm_layer(self.deep_out, train_phase=self.train_phase,
                                                              scope_bn='bn_%s' % i)
                    self.deep_out = self.deep_layers_activation(self.deep_out)
                    self.deep_out = tf.nn.dropout(self.deep_out, self.dropout_keep_deep[i + 1])

                glorot = np.sqrt(2.0 / (self.deep_layers[-1] + 1))
                weights['deep_out'] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[-1], 1)),
                    dtype=np.float32)
                biases['deep_out_bias'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, 1)),
                                                      dtype=np.float32)
                self.out = tf.add(tf.matmul(self.deep_out, weights['deep_out']), biases['deep_out_bias'])

            # loss
            if self.loss_type == 'logloss':
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)

            elif self.loss_type == 'mse':
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))

            # l2 regularization on weights
            if self.l2_reg > 0:
                for i in range(len(self.deep_layers)):
                    self.loss = self.loss + tf.contrib.layers.l2_regularizer(self.l2_reg)(
                        weights['deep_layer_%s' % i])

            # optimizer
            if self.optimizer_type == 'adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

            elif self.optimizer_type == 'adagrad':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)

            elif self.optimizer_type == 'gd':
                self.optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate=self.learning_rate).minimize(
                    self.loss)

            elif self.optimizer_type == 'momentum':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                            momentum=0.95).minimize(self.loss)

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

    def fit(self, feat_index, feat_val, label, valid_feat_index=None, valid_feat_val=None, valid_label=None):
        '''
        :param feat_index: [[idx1_1, idx1_2,...], [idx2_1, idx2_2,...],...]
                            idxi_j is the feature index of feature field j of sample i in the training set
        :param feat_val: [[value1_1, value1_2,...], [value2_1, value2_2,...]...]
                        valuei_j is the feature value of feature field j of sample i in the training set
        :param label: [[label1], [label2], [label3], [label4],...]
        :return: None
        '''
        has_valid = valid_feat_index is not None
        total_time = 0
        for epoch in range(self.epochs):
            start_time = time.time()
            for i in tqdm(range(0, len(feat_index), self.batch_size), ncols=100):
                feat_index_batch = feat_index[i: i + self.batch_size]
                feat_val_batch = feat_val[i: i + self.batch_size]
                batch_y = label[i: i + self.batch_size]

                feed_dict = {
                    self.feature_index: feat_index_batch,
                    self.feature_value: feat_val_batch,
                    self.label: batch_y,
                    self.dropout_keep_deep: self.dropout_deep,
                    self.train_phase: True
                }
                cost, opt = self.sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
            train_metric = self.evaluate(feat_index, feat_val, label)
            end_time = time.time()
            if self.verbose:
                if has_valid:
                    valid_metric = self.evaluate(valid_feat_index, valid_feat_val, valid_label)
                    print('[%s] train-%s=%.4f, valid-%s=%.4f [%.1f s]' % (
                        epoch + 1, self.metric_type, train_metric, self.metric_type, valid_metric,
                        end_time - start_time))
                else:
                    print('[%s] train-%s=%.4f [%.1f s]' % (
                    epoch + 1, self.metric_type, train_metric, end_time - start_time))
            total_time = total_time + end_time - start_time

        print('cost total time=%.1f s' % total_time)

    def predict(self, feat_index, feat_val):
        feed_dict = {
            self.feature_index: feat_index,
            self.feature_value: feat_val,
            self.dropout_keep_deep: [1.0] * len(self.dropout_deep),
            self.train_phase: False
        }
        y_pred = self.sess.run(self.out, feed_dict=feed_dict)
        return y_pred

    def evaluate(self, feat_index, feat_val, label):
        y_pred = self.predict(feat_index, feat_val)
        if self.metric_type == 'auc':
            return roc_auc_score(label, y_pred)
        elif self.metric_type == 'logloss':
            return log_loss(label, y_pred)
        elif self.metric_type == 'acc':
            predict_items = []
            for item in y_pred:
                if item > 0.5:
                    predict_items.append(1)
                else:
                    predict_items.append(0)
            return accuracy_score(label, np.array(predict_items).reshape(-1, 1))

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
    train_feature_index, train_feature_val = dataParse.parse(train)
    test_feature_index, test_feature_val = dataParse.parse(test)

    y_train = y_train.values.reshape(-1, 1)
    y_val = y_val.values.reshape(-1, 1)

    n = len(continuous_feature) + len(category_feature)
    n = int(n * (n - 1) / 2)

    model = PNN(feature_size=dataParse.feature_size,
                field_size=dataParse.field_size,
                deep_init_size=n)

    model.fit(train_feature_index, train_feature_val, y_train)
    test_metric = model.evaluate(test_feature_index, test_feature_val, y_val)
    print('test-auc=%.4f' % test_metric)
