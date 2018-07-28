### PNN
![](https://github.com/wangru8080/Deep_CTR/blob/master/picture/pnn.png)  
每个特征经过embedding后得到相应自身的embedding vetor，互不影响。然后每个embeding vector进行两两相乘得到P，每个embedding vector进行线性变换或者直接输出得到Z，再将Z和P进行相加再加上bias_b，即l = relu(`!$l_{p} + l_{z} + b$`)作为MLP的输入，最终得到结果。

### 实验效果
![](https://github.com/wangru8080/Deep_CTR/blob/master/picture/pnn_result.png)
