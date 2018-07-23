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

    def FeatureDictionary(self, train, test):
        '''
        只对离散特征进行编号
        1.离散特征，one-hot之后每一列都是一个新的特征维度(计算编号时，不算0)。所以，原来的一维度对应的是很多维度，编号也是不同的。
        2.利用category_dict来存储索引信息，每列特征对应一个dict，这个dict存储不同取值对应的编号信息。方便后续进行embedding
        '''

        df = pd.concat([train, test], axis=0)
        cate_dict = {}
        total_cnt = 0

        for col in df.columns:
            if col in self.category_feature:
                unique_cnt = df[col].nunique()
                unique_vals = df[col].unique()
                cate_dict[col] = dict(zip(unique_vals, range(total_cnt, total_cnt + unique_cnt)))
                total_cnt = total_cnt + unique_cnt

        self.category_dict = cate_dict
        self.category_size = total_cnt
        print('category_dict=', cate_dict)
        print('=' * 20)
        print('category_size=', total_cnt)

    def parse(self, df):
        df_category_index = df[self.category_feature]
        for col in self.category_feature:
            df_category_index[col] = df_category_index[col].map(self.category_dict[col])

        category_index = df_category_index.values.tolist()
        del df_category_index
        gc.collect()

        self.category_field_size = len(category_index[0])
        print('category_index=', category_index)
        print('=' * 20)
        print('category_field_size=', self.category_field_size)
        return category_index

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
    embedding.FeatureDictionary(train, test)
    category_index = embedding.parse(train)
    embedding.to_embedding_vector(category_index, train, isPrintEmbeddingInfo=True)
