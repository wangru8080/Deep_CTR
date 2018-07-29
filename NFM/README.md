### NFM
![](https://github.com/wangru8080/Deep_CTR/blob/master/picture/NFM.png)  
简单来说NFM模型实际上是FM加上NN，即当隐藏层数为0时，此模型实际上表示的就是FM模型。  
公式：<a href="https://www.codecogs.com/eqnedit.php?latex=y_{NFM}(\mathbf{x})&space;=&space;w_0&plus;&space;\sum_{i=1}^n&space;w_i&space;x_i&space;&plus;&space;f_{BI}(x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_{NFM}(\mathbf{x})&space;=&space;w_0&plus;&space;\sum_{i=1}^n&space;w_i&space;x_i&space;&plus;&space;f_{BI}(x)" title="y_{NFM}(\mathbf{x}) = w_0+ \sum_{i=1}^n w_i x_i + f_{BI}(x)" /></a>  
其中的Bi-Interaction Layer实际上就是FM中二阶项

### 实验效果
![](https://github.com/wangru8080/Deep_CTR/blob/master/picture/NFM_result.png)
