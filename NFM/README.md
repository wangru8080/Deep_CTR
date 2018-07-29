### NFM
![](https://github.com/wangru8080/Deep_CTR/blob/master/picture/NFM.png)  
简单来说NFM模型实际上是FM加上NN，即当隐藏层数为0时，此模型实际上表示的就是FM模型。  
公式：`!$y(\mathbf{x}) = w_0+ \sum_{i=1}^n w_i x_i + \sum_{i=1}^n \sum_{j=i+1}^n \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j \label{eq:fm}\tag{2}$`  
其中的Bi-Interaction Layer实际上就是FM中二阶项

### 实验效果
![](https://github.com/wangru8080/Deep_CTR/blob/master/picture/NFM_result.png)
