# !/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import tensorflow as tf
import gc
import warnings
warnings.filterwarnings('ignore')

class Embedding:
    def __init__(self, category_feature, continuous_feature, ignore_feature=[], category_dict={}, category_size=0, category_field_size=0, embedding_size=8, random_seed=2018):
        self.category_feature = category_feature
        self.ignore_feature = ignore_feature
        self.continuous_feature = continuous_feature
        self.category_dict = category_dict
        self.category_size = category_size
        self.category_field_size = category_field_size
        self.embedding_size = embedding_size
        self.random_seed = random_seed

    def to_embedding_vector(self, category_index, df, isPrintEmbeddingInfo=False):
        tf.set_random_seed(self.random_seed)

        self.cate_index = tf.placeholder(tf.int32, [None, self.category_field_size])
        self.continuous = tf.placeholder(tf.float32, [None, len(self.continuous_feature)])

        weights = {}

        weights['category_embedding'] = tf.Variable(tf.random_normal([self.category_size, self.embedding_size], mean=0.0, stddev=0.01))

        # category -> Embedding
        self.embedding = tf.nn.embedding_lookup(weights['category_embedding'], ids=self.cate_index) # [None, category_field_size, embedding_size]
        self.embedding = tf.reshape(self.embedding, shape=[-1, self.category_field_size * self.embedding_size]) # [None, category_field_size * embedding_size]
        # concat Embedding Vector & continuous -> Dense Vector
        self.dense_vector = tf.concat([self.embedding, self.continuous], axis=1)

        if isPrintEmbeddingInfo:
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                feed_dict = {
                    self.cate_index: category_index,
                    self.continuous: df[self.continuous_feature].values.tolist()
                }
                dense_vector = sess.run(self.dense_vector, feed_dict=feed_dict)
                print('value=', dense_vector)
                print('shape=', dense_vector.shape)

        return self.embedding

if __name__ == '__main__':
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    ignore_feature = ['id', 'target']
    category_feature = ['feat_cat_1', 'feat_cat_2']
    continuous_feature = ['feat_num_1', 'feat_num_2']

    embedding = Embedding(ignore_feature=ignore_feature, category_feature=category_feature, continuous_feature=continuous_feature)
    
    dataParse = DataParse(continuous_feature=continuous_feature, category_feature=category_feature)
    dataParse.FeatureDictionary(train, test)
    category_index = dataParse.parse(train)
    
    embedding.to_embedding_vector(category_index, train, isPrintEmbeddingInfo=True)
