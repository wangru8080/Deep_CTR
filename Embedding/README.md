### 参考论文
[A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](https://arxiv.org/pdf/1512.05287.pdf)

### TensorFlow中embedding_lookup的用法：
通俗来讲，首先创建一个参数矩阵，矩阵大小为(feature_size, embedding_size)，然后根据索引号，也就是feature_index，取出参数矩阵中下标为feature_index的元素组成一个一个矩阵返回。  

### category_feature与continuous_feature共享Embedding的做法
对continuous_feature与category_feature一起编号，对于continuous_feature仅用一个编号来表示，而category_feature涉及到有多个取值，所以一个category_feature对应多个编号，而这多个编号组成一个field。  
如果category_feature中取值少，就直接编号。多的话可采用hash处理，这样避免了过多的加法运算。  
然后category_feature与continuous_feature一起进行embedding，所以最终需要将feature_value的值与Embedding矩阵对应元素的值相乘，不同于矩阵相乘（tf.multiply），得到最终的Embedding Vector
