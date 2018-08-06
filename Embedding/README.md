### 参考论文
[A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](https://arxiv.org/pdf/1512.05287.pdf)

### TensorFlow中embedding_lookup的用法：
通俗来讲，首先创建一个参数矩阵，矩阵大小为(feature_size, embedding_size)，然后根据索引号，也就是feature_index，取出参数矩阵中下标为feature_index的元素组成一个一个矩阵返回。  

### 涉及到的两种Embedding方法
#### 1、category_feature与continuous_feature共享Embedding的做法
连续特征与离散特征一起编号，对于连续特征仅用一个编号来表示，而离散特征涉及到有多个取值，所以一个离散特征对应多个编号，而这多个编号组成一个field。  
如果离散特征中取值少，就直接编号。多的话可采用hash处理，这样避免了过多的加法运算。  
然后离散特征与连续特征一起进行embedding，所以最终需要将feature_value的值与Embedding矩阵对应元素的值相乘，不同于矩阵相乘（tf.multiply），得到最终的Embedding Vector  
如图所示：  
![](https://github.com/wangru8080/Deep_CTR/blob/master/picture/sharing_embedding.png)  

#### 2、Dense Vector
离散特征经过embedding后与连续特征进行concat

### 多值离散特征的Embedding方法
某一个特征的embedding  
1、计算当前特征的最长的长度maxlen, maxlen=5  
2、计算每个样本的长度，[3, 4, 5]  
3、计算每个值的索引号，对于当前样本长度不足maxlen的用0补充，[[1, 2, 3, 0, 0], [4, 1, 5, 6, 0], [5, 7, 8, 6, 9]]  
4、然后进行embedding_lookup，得到embedding向量  
5、为了过滤多加的值对最终结果的影响，需进行mean pooling操作  

最后将所有的Embedding Vector进行concat即可
