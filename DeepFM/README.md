### DeepFM
![](https://github.com/wangru8080/Deep_CTR/blob/master/picture/DeepFM.png)  
FM部分与Deep部分共享Embedding Vector。  
在输入时，所有的离散特征与连续特征共同进行编码，并进行embedding得到Embedding Vector，使用Embedding Vector分别完成FM线性部分与二阶组合特征部分，以及作为MLP的输入，最后将FM与MLP的输出进行concat，通过sigmoid激活函数（二分类）得到输出。

### 效果
![](https://github.com/wangru8080/Deep_CTR/blob/master/picture/DeepFM_result.png)
