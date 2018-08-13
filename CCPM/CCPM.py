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
import math
np.random.seed(2018)

class CCPM(BaseEstimator, TransformerMixin):
    def __init__(self, feature_size,
                 field_size,
                 embedding_size=8,
                 convolution_layers_num=3,
                 convolution_layers_filters=[64, 64, 64],
                 epochs=10,
                 batch_size=128,
                 learning_rate=0.001,
                 optimizer_type='adam',
                 verbose=True,
                 random_seed=2018,
                 loss_type='logloss',
                 metric_type='auc',
                 l2_reg=0.0):
        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.convolution_layers_num = convolution_layers_num
        self.convolution_layers_filters = convolution_layers_filters
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.metric_type = metric_type
        self.l2_reg = l2_reg

        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.feature_index = tf.placeholder(tf.int32, [None, self.field_size], name='feature_index')
            self.feature_value = tf.placeholder(tf.float32, [None, self.field_size], name='feature_value')
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name='label')

            with tf.name_scope('Embedding_Layer'):
                self.embeddings = tf.keras.layers.Embedding(input_dim=self.feature_size,output_dim=self.embedding_size)(self.feature_index)  # [[None, field_size, embedding_size]]
                feat_value = tf.reshape(self.feature_value, shape=[-1, self.field_size, 1])  # [None, field_size, 1]
                self.embeddings = tf.multiply(self.embeddings, feat_value)  # [None, field_size, embedding_size]
                self.embeddings = tf.expand_dims(self.embeddings, axis=-1) # [None, field_size, embedding_size, 1]
                self.l = tf.transpose(self.embeddings, perm=[0, 2, 1, 3]) # [None, embedding_size, field_size, 1]

            with tf.name_scope('Convlution_Layer'):
                self.l = tf.keras.layers.Conv2D(filters=self.convolution_layers_filters[0], kernel_size=(3, 3), padding='same',
                                                activation=tf.nn.relu,
                                                input_shape=(self.embedding_size, self.field_size, 1))(self.l)
                self.l = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(self.l)

                for i in range(1, self.convolution_layers_num):
                    self.l = tf.keras.layers.Conv2D(filters=self.convolution_layers_filters[i], kernel_size = (3, 3), padding = 'same', activation = tf.nn.relu)(self.l)
                    self.l = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(self.l)

                self.out = tf.keras.layers.Flatten()(self.l)
                self.out = tf.keras.layers.Dense(1, activation=None)(self.out)

                # loss
                if self.loss_type == 'logloss':
                    self.out = tf.nn.sigmoid(self.out)
                    self.loss = tf.losses.log_loss(self.label, self.out)

                elif self.loss_type == 'mse':
                    self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))

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

    def get_p_max_pooling(self, l):

        assert l <= self.convolution_layers_num, 'p-max pooling error'

        if l == self.convolution_layers_num:
            return 3
        else:
            return (1 - math.pow(int(l / self.convolution_layers_num), self.convolution_layers_num - l)) * self.field_size

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
                    self.label: batch_y
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
            self.feature_value: feat_val
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

    model = CCPM(feature_size=dataParse.feature_size,
                field_size=dataParse.field_size,
                 metric_type='acc')

    model.fit(train_feature_index, train_feature_val, y_train)
    test_metric = model.evaluate(test_feature_index, test_feature_val, y_val)
    print('test-auc=%.4f' % test_metric)
