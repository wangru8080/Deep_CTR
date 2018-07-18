import tensorflow as tf
import numpy as np

class DeepFM:
    def __init__(self, feature_size, field_size, embedding_size=8, deep_layers=[32, 32], dropout_fm=[1.0, 1.0],
                 dropout_deep=[0.5, 0.5, 0.5], deep_layers_activation=tf.nn.relu, epochs=10, batch_size=128,
                 learning_rate=0.001, optimizer_type='adam',random_seed=2018, loss_type='logloss', use_fm=True,
                 use_deep=True, metric_type='auc', l2_reg=0.0):
        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.deep_layers = deep_layers
        self.dropout_fm = dropout_fm
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.usefm = use_fm
        self.use_deep = use_deep
        self.metric_type = metric_type
        self.l2_reg = l2_reg

    def fit(self, train_index, train_value, label):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.feat_index = tf.placeholder(tf.int32, shape=[None, None], name='feat_index')
            self.feat_value = tf.placeholder(tf.float32, shape=[None, None], name='feat_value')
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name='label')

            self.dropout_keep_fm = tf.placeholder(tf.float32, shape=[None], name='dropout_keep_fm')
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name='dropout_keep_deep')

            weights = {}
            biases = {}

            with tf.name_scope('init'):
                weights['feature_embeddings'] = tf.Variable(tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01), name='feature_embeddings')
                biases['feature_bias'] = tf.Variable(tf.random_uniform([self.feature_size, 1], 0.0, 1.0), name='feature_bias')
                self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.feat_index)
                feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size, 1])
                self.embeddings = tf.multiply(self.embeddings, feat_value)

            with tf.name_scope('FM'):
                self.y_first_order = tf.nn.embedding_lookup(self.weights['feature_bias'], self.feat_index)
                self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order, feat_value), 2)
                self.y_first_order = tf.nn.dropout(self.y_first_order, self.dropout_keep_fm[0])

                self.summed_features_emb = tf.reduce_sum(self.embeddings, 1)
                self.summed_features_emb_square = tf.square(self.summed_features_emb)
                self.squared_features_emb = tf.square(self.embeddings)
                self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)
                self.y_second_order = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb)
                self.y_second_order = tf.nn.dropout(self.y_second_order, self.dropout_keep_fm[1])

            with tf.name_scope('Deep'):
                num_layer = len(self.deep_layers)
                input_size = self.field_size * self.embedding_size
                glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))
                weights['deep_layer_0'] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32)
                biases['deep_layer_bias_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])),
                                                dtype=np.float32)
                for i in range(1, num_layer):
                    glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
                    weights['deep_layer_%s' % i] = tf.Variable(
                        np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i - 1], self.deep_layers[i])),
                        dtype=np.float32)
                    biases['deep_layer_bias_%s' % i] = tf.Variable(
                        np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                        dtype=np.float32)

                self.y_deep = tf.reshape(self.embeddings, shape=[-1, self.field_size * self.embedding_size])
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])

                for i in range(len(self.deep_layers)):
                    self.y_deep = tf.add(tf.matmul(self.y_deep, weights['deep_layer_%s' % i]),
                                           biases['deep_layer_bias_%s' % i])
                    self.y_deep = self.deep_layers_activation(self.y_deep)
                    self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[i + 1])

            with tf.name_scope('DeepFM or FM or Deep'):
                if self.use_fm and self.use_deep:
                    input_size = self.field_size + self.embedding_size + self.deep_layers[-1]

                elif self.use_fm:
                    input_size = self.field_size + self.embedding_size

                elif self.use_deep:
                    input_size = self.deep_layers[-1]

                glorot = np.sqrt(2.0 / (input_size + 1))
                weights['concat_projection'] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
                    dtype=np.float32)
                biases['concat_bias'] = tf.Variable(tf.constant(0.01), dtype=np.float32)

                if self.use_fm and self.use_deep:
                    concat_input = tf.concat([self.y_first_order, self.y_second_order, self.y_deep], axis=1)
                elif self.use_fm:
                    concat_input = tf.concat([self.y_first_order, self.y_second_order], axis=1)
                elif self.use_deep:
                    concat_input = self.y_deep

                self.out = tf.add(tf.matmul(concat_input, self.weights['concat_projection']),
                                  self.weights['concat_bias'])

                # loss
                if self.loss_type == 'logloss':
                    self.out = tf.nn.sigmoid(self.out)
                    self.loss = tf.losses.log_loss(self.label, self.out)

                elif self.loss_type == 'mse':
                    self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))

                # l2 regularization on weights
                if self.l2_reg > 0:
                    self.loss += tf.contrib.layers.l2_regularizer(
                        self.l2_reg)(self.weights['concat_projection'])
                    if self.use_deep:
                        for i in range(len(self.deep_layers)):
                            self.loss += tf.contrib.layers.l2_regularizer(
                                self.l2_reg)(self.weights['deep_layer_%s' % i])

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
                                                                momentum=0.95).minimize(self.loss)

                elif self.optimizer_type == 'rmsprop':
                    self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

                # metric
                if self.metric_type == 'auc':
                    self.metric = tf.metrics.auc(self.label, self.out)

                elif self.metric_type == 'accuracy':
                    self.metric = tf.metrics.accuracy(self.label, self.out)

                init = tf.global_variables_initializer()
                self.sess = tf.Session()
                self.sess.run(init)

                init = tf.global_variables_initializer()
                self.sess = tf.Session()
                self.sess.run(init)

                # train
                for epoch in range(self.epochs):
                    for i in range(0, len(train_index), self.batch_size):
                        feed_dict = {
                            self.feat_index: train_index[i: i + self.batch_size],
                            self.feat_value: train_value[i: i + self.batch_size],
                            self.label: label[i: i + self.batch_size],
                            self.dropout_keep_fm: self.dropout_fm,
                            self.dropout_keep_deep: self.dropout_deep,
                        }
                        cost, opt = self.sess.run([self.loss, self.optimizer], feed_dict=feed_dict)

                    print('Epoch=%s, cost=%s' % (epoch + 1, cost))

    def predict(self):
        pass

if __name__ == '__main__':
    pass
