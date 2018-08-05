# !/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
import pandas as pd

tf.set_random_seed(2018)

df = pd.read_csv('data/multi_field.csv')
print(df)

```
df:
id
a|b|c
d|a|e|c
f|g|b|c|a
```

maxlen = 0
for data in df['name'].values:
    ds = data.split('|')
    if maxlen < len(ds):
        maxlen = len(ds)
print('maxlen=', maxlen)

embedding_size = 3

index = {}
count = 1
for data in df['name'].values:
    ds = data.split('|')
    for d in ds:
        if d not in index.keys():
            index[d] = count
            count = count + 1

print('index:', index)

multi_index = []
multi_length = []
for data in df['name'].values:
    ds = data.split('|')
    multi_length.append(len(ds))
    l = []
    for d in ds:
        l.append(index[d])
    multi_index.append(l)

print(multi_index)
print('multi_length:', multi_length)

# padding
for i in range(len(multi_index)):
    if len(multi_index[i]) < maxlen:
        for j in range(maxlen - len(multi_index[i])):
            multi_index[i].append(0)

print('after padding:',multi_index)

dynamic_index = tf.placeholder(tf.int32, shape=[None, maxlen])
dynamic_length = tf.placeholder(tf.int32, shape=[None])

weights = {}
biases = {}

embedding_size = 3
multi_size = len(index.keys())

weights['multi_embedding'] = tf.Variable(tf.random_normal([multi_size, embedding_size], mean=0.0, stddev=0.01))
embedding = tf.nn.embedding_lookup(weights['multi_embedding'], dynamic_index) # [None, maxlen, embedding_size]
mask = tf.sequence_mask(dynamic_length, maxlen)
mask_float = tf.to_float(mask) # [None, maxlen]
multi_mask = tf.expand_dims(mask_float, axis=-1) # [None, maxlen, 1]
multi_mask = tf.concat([multi_mask for i in range(embedding_size)], axis=-1) # [None, maxlen, embedding_size]

multi_embedding = tf.multiply(embedding, multi_mask) # [None, maxlen, embedding_size]

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    feed_dict = {
        dynamic_index: multi_index,
        dynamic_length: multi_length
    }
    print(sess.run(multi_embedding, feed_dict=feed_dict))
