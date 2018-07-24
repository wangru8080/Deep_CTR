# !/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import tensorflow as tf
import gc
import warnings
warnings.filterwarnings('ignore')

class SharingEmbedding:
    def __init__(self, category_feature, continuous_feature, ignore_feature=[], feature_dict={}, feature_size=0 , field_size=0, embedding_size=8, random_seed=2018):
        self.feature_dict = feature_dict
        self.feature_size = feature_size
        self.field_size = field_size
        self.ignore_feature = ignore_feature
        self.category_feature = category_feature
        self.continuous_feature = continuous_feature
        self.embedding_size = embedding_size
        self.random_seed= random_seed
        
    def to_sharing_embedding_vector(self, Xi, Xv, isPrintEmbeddingInfo=False): # category_feature与continuous_feature共享embedding

        tf.set_random_seed(self.random_seed)

        self.feature_index = tf.placeholder(tf.int32, shape=[None, self.field_size])
        self.feature_value = tf.placeholder(tf.float32, shape=[None, self.field_size])

        weights = {}

        weights['feature_embedding'] = tf.Variable(tf.random_normal([self.feature_size, self.embedding_size], mean=0.0, stddev=0.01))

        # Sparse Features -> Dense Embedding
        self.embedding = tf.nn.embedding_lookup(weights['feature_embedding'], ids=self.feature_index) # [None, field_size, embedding_size]
        feature_value = tf.reshape(self.feature_value, shape=[-1, self.field_size, 1])
        self.embedding = tf.multiply(self.embedding, feature_value)
        self.embedding = tf.reshape(self.embedding, shape=[-1, self.field_size * self.embedding_size]) # [None, field_size * embedding_size]

        if isPrintEmbeddingInfo:
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                feed_dict = {
                    self.feature_index: Xi,
                    self.feature_value: Xv
                }
                embedds = sess.run(self.embedding, feed_dict=feed_dict)
                print('value=', embedds)
                print('shape=', embedds.shape)

        return self.embedding

if __name__ == '__main__':
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    ignore_feature = ['id', 'target']
    category_feature = ['feat_cat_1', 'feat_cat_2']
    continuous_feature = ['feat_num_1', 'feat_num_2']

    embedding = SharingEmbedding(category_feature=category_feature, ignore_feature=ignore_feature, continuous_feature=continuous_feature)
    
    dataParse = DataParse(continuous_feature=continuous_feature, category_feature=category_feature)
    dataParse.FeatureDictionary(train, test)
    Xi, Xv, y = dataParse.parse(train)
    
    embedding.to_sharing_embedding_vector(Xi, Xv, isPrintEmbeddingInfo=True)
