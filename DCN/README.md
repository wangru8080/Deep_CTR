### DCN
![](https://github.com/wangru8080/Deep_CTR/blob/master/picture/DCN.png)  
首先离散特征进行embedding，然后与连续特征进行concat，得到embedding vector。  
embedding vector作为cross network与deep network的输入，这里详细讲述cross network  
cross network主要是捕获组合特征，论文给出公式：<a href="https://www.codecogs.com/eqnedit.php?latex=x_{l&plus;1}&space;=&space;x_{0}x_{l}^Tw_{l}&plus;b_{l}&plus;x_{l}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_{l&plus;1}&space;=&space;x_{0}x_{l}^Tw_{l}&plus;b_{l}&plus;x_{l}" title="x_{l+1} = x_{0}x_{l}^Tw_{l}+b_{l}+x_{l}" /></a>，随着cross layer层数的增加，组合特征的度也会进一步增加。

### 实验效果
![](https://github.com/wangru8080/Deep_CTR/blob/master/picture/DCN_result.png)
