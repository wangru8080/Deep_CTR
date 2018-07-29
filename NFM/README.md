### NFM
![](https://github.com/wangru8080/Deep_CTR/blob/master/picture/NFM.png)  
简单来说NFM模型实际上是FM加上NN，即当隐藏层数为0时，此模型实际上表示的就是FM模型。  
公式：<a href="https://www.codecogs.com/eqnedit.php?latex=y_{NFM}(\mathbf{x})&space;=&space;w_0&plus;&space;\sum_{i=1}^n&space;w_i&space;x_i&space;&plus;&space;f_{BI}(x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_{NFM}(\mathbf{x})&space;=&space;w_0&plus;&space;\sum_{i=1}^n&space;w_i&space;x_i&space;&plus;&space;f_{BI}(x)" title="y_{NFM}(\mathbf{x}) = w_0+ \sum_{i=1}^n w_i x_i + f_{BI}(x)" /></a>  
<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_cm&space;f_{BI}(x)=\frac{1}{2}&space;\sum_{f=1}^k&space;\left(\left(&space;\sum_{i=1}^n&space;v_{i,&space;f}&space;x_i&space;\right)^2-\sum_{i=1}^n&space;v_{i,&space;f}^2&space;x_{i}^2&space;\right)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\fn_cm&space;f_{BI}(x)=\frac{1}{2}&space;\sum_{f=1}^k&space;\left(\left(&space;\sum_{i=1}^n&space;v_{i,&space;f}&space;x_i&space;\right)^2-\sum_{i=1}^n&space;v_{i,&space;f}^2&space;x_{i}^2&space;\right)" title="f_{BI}(x)=\frac{1}{2} \sum_{f=1}^k \left(\left( \sum_{i=1}^n v_{i, f} x_i \right)^2-\sum_{i=1}^n v_{i, f}^2 x_{i}^2 \right)" /></a>  
其中的Bi-Interaction Layer实际上就是FM中二阶项

### 实验效果
![](https://github.com/wangru8080/Deep_CTR/blob/master/picture/NFM_result.png)
