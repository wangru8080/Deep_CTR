import pandas as pd
import tensorflow as tf
import gc

class Embedding:
    def __init__(self, category_feature, continuous_feature, ignore_feature=[], feature_dict={}, feature_size=0 , field_size=0, embedding_size=8):
        self.feature_dict = feature_dict
        self.feature_size = feature_size
        self.field_size = field_size
        self.ignore_feature = ignore_feature
        self.category_feature = category_feature
        self.continuous_feature = continuous_feature
        self.embedding_size = embedding_size

    def FeatureDictionary(self, train, test):
        '''
        目的是给每一个特征维度都进行编号。
        1. 对于离散特征，one-hot之后每一列都是一个新的特征维度(计算编号时，不算0)。所以，原来的一维度对应的是很多维度，编号也是不同的。
        2. 对于连续特征，原来的一维特征依旧是一维特征。
        返回一个feat_dict，用于根据原特征名称和特征取值 快速查询出 对应的特征编号。
        train: 原始训练集
        test:  原始测试集
        continuous_feature: 所有数值型特征
        ignore_feature:  所有忽略的特征. 除了数值型和忽略的，剩下的全部认为是离散型
        feat_dict, feat_size
             1. feat_size: one-hot之后总的特征维度。
             2. feat_dict是一个{}， key是特征string的col_name, value可能是编号（int），可能也是一个字典。
             如果原特征是连续特征： value就是int，表示对应的特征编号；
             如果原特征是离散特征：value就是dict，里面是根据离散特征的 实际取值 查询 该维度的特征编号。 因为离散特征one-hot之后，一个取值就是一个维度，
             而一个维度就对应一个编号。
        '''
        df = pd.concat([train, test], axis=0)
        feat_dict = {}
        total_cnt = 0

        for col in df.columns:
            if col in self.ignore_feature:  # 忽略的特征不参与编号
                continue

            # 连续特征只有一个编号
            elif col in self.continuous_feature:
                feat_dict[col] = total_cnt
                total_cnt = total_cnt + 1

            # 离散特征，有多少个取值就有多少个编号
            elif col in self.category_feature:
                unique_vals = df[col].unique()
                unique_cnt = df[col].nunique()
                feat_dict[col] = dict(zip(unique_vals, range(total_cnt, total_cnt + unique_cnt)))
                total_cnt = total_cnt + unique_cnt

        self.feature_size = total_cnt
        self.feature_dict = feat_dict
        print('feat_dict=', feat_dict)
        print('='*20)
        print('feature_size=', total_cnt)

    def parse(self, df, has_label=True):
        '''
        构造FeatureDict，用于后面Embedding
        feat_dict: FeatureDictionary生成的。用于根据col和value查询出特征编号的字典
        df: 数据输入。可以是train也可以是test,不用拼接
        has_label:  数据中是否包含label
        return:  Xi, Xv, y
        '''

        # dfi是Feature index,大小和train相同，但是里面的值都是特征对应的编号。
        # dfv是Feature value, 可以是binary(0或1), 也可以是实值float，比如3.14

        dfi = df.copy()
        if has_label:
            y = df['target'].values.tolist()
            dfi.drop(['id', 'target'], axis=1, inplace=True)
        else:
            ids = dfi['id'].values.tolist()  # 预测样本的ids
            dfi.drop(['id'], axis=1, inplace=True)

        dfv = dfi.copy()
        for col in dfi.columns:
            if col in self.ignore_feature:
                dfi.drop([col], axis=1, inplace=True)
                dfv.drop([col], axis=1, inplace=True)

            elif col in self.continuous_feature: # 连续特征1个维度，对应1个编号，这个编号是一个定值
                dfi[col] = self.feature_dict[col]

            elif col in self.category_feature: # 离散特征。不同取值对应不同的特征维度，编号也是不同的。
                dfi[col] = dfi[col].map(self.feature_dict[col])
                dfv[col] = 1.0

        Xi = dfi.values.tolist()
        Xv = dfv.values.tolist()
        self.field_size = len(Xi[0])
        print('Xi=', Xi)
        print('='*20)
        print('Xv=', Xv)
        del dfi, dfv
        gc.collect()

        if has_label:
            print('='*20)
            print('y=', y)
            return Xi, Xv, y
        else:
            return Xi, Xv, ids

    def to_embedding_vector(self, Xi, Xv, isPrintEmbeddingInfo=False):

        feature_index = tf.placeholder(tf.int32, shape=[None, self.field_size])
        feature_value = tf.placeholder(tf.float32, shape=[None, self.field_size])

        weights = {}

        weights['feature_embedding'] = tf.Variable(tf.random_normal([self.feature_size, self.embedding_size], mean=0, stddev=0.1))

        # Sparse Features -> Dense Embedding
        embedding = tf.nn.embedding_lookup(weights['feature_embedding'], ids=feature_index) # [None, field_size, embedding_size]
        embedding = tf.reshape(embedding, shape=[-1, self.field_size * self.embedding_size]) # [None, field_size * embedding_size]

        if isPrintEmbeddingInfo:
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                feed_dict = {
                    feature_index: Xi,
                    feature_value: Xv
                }
                embedds = sess.run(embedding, feed_dict=feed_dict)
                print('shape=', embedds.shape)
                print('value=', embedds)

        return embedding




if __name__ == '__main__':
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    ignore_feature = ['id', 'target']
    category_feature = ['cat_1', 'cat_2']
    continuous_feature = ['con_1', 'con_2']

    embedding = Embedding(category_feature=category_feature, ignore_feature=ignore_feature, continuous_feature=continuous_feature)
    embedding.FeatureDictionary(train, test)
    Xi, Xv, y = embedding.parse(train)
    embedding.to_embedding_vector(Xi, Xv, isPrintEmbeddingInfo=True)
