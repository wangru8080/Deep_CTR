import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from keras.optimizers import Adam
from keras.layers import Input, Dense, Flatten, Embedding, Reshape, concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

def getDataSet():
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex',
         'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income_label']
    print('读取数据...')
    train = pd.read_csv('data/adult.data', names = columns, skipinitialspace = True) # skipinitialspace 忽略分隔符后的空白
    test = pd.read_csv('data/adult.test', names = columns, skipinitialspace = True)

    train['income_label'] = train['income_label'].apply(lambda x: '>50K' in x).astype(int)
    test['income_label'] = test['income_label'].apply(lambda x: '>50K' in x).astype(int)

    train['is_train'] = 1
    test['is_train'] = 0
    data = pd.concat([train, test], axis = 0)

    continuous_feature = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    category_feature = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']

    #labelencoder
    lbc = LabelEncoder()
    print('label encoder start...')
    for feature in category_feature:
        print("this is feature:", feature)
        try:
            data[feature] = lbc.fit_transform(data[feature].apply(int))
        except:
            data[feature] = lbc.fit_transform(data[feature].astype(str))

    train = data[data['is_train'] == 1].drop('is_train', axis = 1)
    test = data[data['is_train'] == 0].drop('is_train', axis = 1)

    return train, test

# cross feature
def cross_columns(cols): 
    crossed_columns = dict()
    col_name = ['_'.join(col) for col in cols]
    
    for name, cross in zip(col_name, cols):
        crossed_columns[name] = cross
    return crossed_columns

# helper to index categorical columns before embeddings -> labelencoder
def val2idx(df, cols):
    val_types = dict()
    for c in cols:
        val_types[c] = df[c].unique()
   
    val_to_idx = dict()
    for k in val_types.keys(): # 构建索引列表
        val_to_idx[k] = {val: i for i, val in enumerate(val_types[k])}

    for k, v in val_to_idx.items(): # df_val to index
        df[k] = df[k].apply(lambda x: v[x])
    
    unique_vals = dict()
    for c in cols: # 得到当前列非重复值个数
        unique_vals[c] = df[c].nunique()

    return df, unique_vals

def onehot(x):
    return np.array(OneHotEncoder().fit_transform(x).todense())

def embedding_input(name, input_dim, output_dim, regularizer):
    ipt = Input(shape = (1, ), dtype = 'int64', name = name)
    emdedding = Embedding(input_dim = input_dim, output_dim = output_dim, input_length=1, embeddings_regularizer = l2(regularizer))(ipt)
    return ipt, emdedding

def continuous_input(name):
    ipt = Input(shape = (1,), dtype = 'float32', name = name)
    reshape = Reshape((1, 1))(ipt)
    return ipt, reshape

def wide_part(train, test, continuous_feature, category_feature, cross_cols, target, model_type, method = 'logistic'):
    '''
    wide_part: linear model, require more feature engineering, generalize  cross-product feature
    params info:

    train, test: datasets
    wide_cols: used to fit the wide model
    cross_cols: used to generate cross feature
    target: label
    model_type: wide out or wide_deep input
    method: the fitting method. accepts regression, logistic and multiclass
    '''
    train['is_train'] = 1
    test['is_train'] = 0
    data = pd.concat([train, test], axis = 0)
   
    crossed_columns = cross_columns(cross_cols)
    wide_cols = continuous_feature + category_feature + list(crossed_columns.keys())

    for k, v in crossed_columns.items(): # cross feature
        data[k] = data[v].astype(str).apply(lambda x: '-'.join(x), axis = 1) # data[v1] + '-' + data[v2], v: v1, v2

    # category_feature one-hot
    dummy_cols = [col for col in category_feature + list(crossed_columns.keys())]
    data = pd.get_dummies(data, columns = [x for x in dummy_cols])
    
    # 归一化
    scaler = MinMaxScaler()
    for feature in continuous_feature:
        data[feature] = scaler.fit_transform(data[feature].values.reshape(-1, 1))

    train = data[data['is_train'] == 1].drop('is_train', axis = 1)
    test = data[data['is_train'] == 0].drop('is_train', axis = 1)
    
    print('start cut wide_part datasets')
    y_train = train[target].values.reshape(-1, 1)
    train.drop(target, axis = 1, inplace = True)
    y_val = test[target].values.reshape(-1, 1)
    test.drop(target, axis = 1, inplace = True)

    if method == 'multiclass':
        y_train = onehot(y_train)
        y_val = onehot(y_val)

    if model_type == 'wide':
        activation, loss, metrics = 'sigmoid', 'binary_crossentropy', 'accuracy'
        if metrics:
            metrics = [metrics]

        # 直接输出
        wide_ipt = Input(shape = (train.shape[1], ), dtype = 'float32', name = 'wide_ipt')
        out = Dense(y_train.shape[1], activation = activation)(wide_ipt)
        model = Model(inputs = wide_ipt, outputs = out)
        model.compile(Adam(0.01), loss = loss, metrics = metrics)
        model.fit(train, y_train, epochs = 10, batch_size = 64)
        result = model.evaluate(test, y_val)
        print('wide result:', result)
    else:
        return train, test, y_train, y_val

def deep_part(train, test, embedding_cols, continuous_cols, target, model_type, method = 'logistic'):
    '''
    deep_part: dnn model, generalize high-order non-linear feature interactions
    params info:

    train, test: datasets
    embedding_cols: category_feature -> embedding
    continuous_cols: continuous feature
    target: label
    model_type: deep out or wide_deep input
    method: the fitting method. accepts regression, logistic and multiclass
    '''
    train['is_train'] = 1
    test['is_train'] = 0
    data = pd.concat([train, test], axis = 0)

    data, unique_vals = val2idx(data, embedding_cols)
    train = data[data['is_train'] == 1].drop('is_train', axis=1)
    test = data[data['is_train'] == 0].drop('is_train', axis=1)

    deep_cols = embedding_cols + continuous_cols

    embeddings_tensors = []
    embedding_output_dim = 8
    reg = 1e-3
    for col in embedding_cols:
        layer_name = col + '_ipt'
        ipt, emdedding = embedding_input(layer_name, unique_vals[col], embedding_output_dim, reg)
        embeddings_tensors.append((ipt, emdedding))
        del (ipt, emdedding)
    
    continuous_tensors = []
    for col in continuous_cols:
        layer_name = col + '_ipt'
        ipt, reshape = continuous_input(layer_name)
        continuous_tensors.append((ipt, reshape))
        del (ipt, reshape)
    
    y_train = train[target].values.reshape(-1, 1)
    train.drop(target, axis = 1, inplace = True)
    train = [train[c] for c in deep_cols]
    y_val = test[target].values.reshape(-1, 1)
    test.drop(target, axis = 1, inplace = True)
    test = [test[c] for c in deep_cols]
    
    if method == 'multiclass':
        y_train = onehot(y_train)
        y_val = onehot(y_val)
    
    category_ipt_layer = [et[0] for et in embeddings_tensors]
    continuous_ipt_layer = [ct[0] for ct in continuous_tensors]
    ipt_layer = category_ipt_layer + continuous_ipt_layer
    category_embedding_layer = [et[1] for et in embeddings_tensors]
    continuous_embedding_layer = [ct[1] for ct in continuous_tensors]
    embedding_layer = category_embedding_layer + continuous_embedding_layer

    if model_type == 'deep':
        activation, loss, metrics = activation, loss, metrics = 'sigmoid', 'binary_crossentropy', 'accuracy'
        if metrics:
            metrics = [metrics]

        embedding = concatenate(embedding_layer)
        flatten = Flatten()(embedding)
        bn = BatchNormalization()(flatten)
        # 按照wide&deep论文来构造网络结构，存在的问题，数据量大小不一样，效果会不一样
        dense = Dense(100, activation='relu')(bn)
        dense = Dense(50, activation='relu')(dense)
        out = Dense(y_train.shape[1], activation = activation)(dense)
        model = Model(inputs = ipt_layer, outputs = out)
        model.compile(Adam(0.01), loss = loss, metrics = metrics)
        model.fit(train, y_train, epochs = 10, batch_size = 64) # 因为对每个特征进行embedding，所以传入fit的是一个list，所以需要对train进行list化
        result = model.evaluate(test, y_val)
        print('deep result:', result)
    else:
        return train, test, y_train, y_val, ipt_layer, embedding_layer

def wide_deep(train, test, continuous_feature, category_feature, cross_cols, embedding_cols, continuous_cols, target, model_type = 'wide_deep', method = 'logistic'):
    '''
    merge wide part and deep part
    '''
    wide_train, wide_test, wide_y_train, wide_y_val = wide_part(train, test, continuous_feature, category_feature, cross_cols, target, model_type)
    deep_train, deep_test, deep_y_train, deep_y_val, deep_ipt_layer, deep_embedding_layer = deep_part(train, test, embedding_cols, continuous_cols, target, model_type)

    activation, loss, metrics = activation, loss, metrics = 'sigmoid', 'binary_crossentropy', 'accuracy'
    if metrics:
        metrics = [metrics]

    wide_deep_train = [wide_train] + deep_train
    wide_deep_y_train = deep_y_train # or wide_y_train
    wide_deep_test = [wide_test] + deep_test
    wide_deep_y_val = deep_y_val # or wide_y_val
    
    # wide part
    wide_ipt = Input(shape = (wide_train.shape[1], ), dtype='float32', name = 'wide_part')

    # deep part
    embedding = concatenate(deep_embedding_layer)
    flatten = Flatten()(embedding)
    bn = BatchNormalization()(flatten)
    dense = Dense(100, activation = 'relu')(bn)
    deep_dense_ipt = Dense(50, activation = 'relu', name = 'deep')(dense)

    # wide&deep
    wide_deep_ipt = concatenate([wide_ipt, deep_dense_ipt])
    wide_deep_out = Dense(deep_y_train.shape[1], activation = activation, name = 'wide_deep')(wide_deep_ipt)
    model = Model(inputs = [wide_ipt] + deep_ipt_layer, outputs = wide_deep_out)
    model.compile(Adam(0.01), loss = loss, metrics = metrics)
    model.fit(wide_deep_train, wide_deep_y_train, epochs = 10, batch_size = 64)

    result = model.evaluate(wide_deep_test, wide_deep_y_val)
    print('wide_deep result:', result)

if __name__ == '__main__':
    train, test = getDataSet()
    target = 'income_label'
    continuous_feature = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    category_feature = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
    # wide
    cross_cols = (['education', 'occupation'], ['native_country', 'occupation'])
    # deep
    embedding_cols = category_feature
    continuous_cols = continuous_feature
    
    # wide_part(train, test, continuous_feature, category_feature, cross_cols, target, 'wide')
    # deep_part(train, test, embedding_cols, continuous_cols, target, 'deep')
    wide_deep(train, test, continuous_feature, category_feature, cross_cols, embedding_cols, continuous_cols, target)

    
