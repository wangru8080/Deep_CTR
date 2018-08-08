### FNN
![](https://github.com/wangru8080/Deep_CTR/blob/master/picture/FNN.png)  
FNN模型以FM作为底层，FM模型本身可以得到低阶组合特征，取出FM的隐向量，将此构造出NN的输入层z:z=(W<sub>0</sub>, Z<sub>1</sub>, Z<sub>2</sub>,..., Z<sub>n</sub>)，而Z<sub>i</sub>=(W<sub>i</sub>, V<sub>i</sub><sup>1</sup>, V<sub>i</sub><sup>2</sup>,..., V<sub>i</sub><sup>k</sup>)，其中W<sub>i</sub>为FM中的一阶权重V<sub>i</sub>对应的隐向量。最后通过sigmoid激活函数得到输出

### 实验效果
![](https://github.com/wangru8080/Deep_CTR/blob/master/picture/FNN_result.png)
