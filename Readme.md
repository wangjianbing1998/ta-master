进入Task_1/和Task_2/  目录下


- 这里面主要是投资学的indicator的计算过程（给定formula，code化）
- 当然也包括一些数据预处理的code，主要是为了熟练自己的操作技术
# Data
- 主要的symbol有
  - 标普500 （time unit=1 min）
  - 纳斯达克指数(time unit = 1 min)
  - A股的tushare (time unit = 5 min)
  
  
# Task_1/
- 主要进入Task_1/  目录下
- indicator有102个主流的
- 还有一些数据处理的函数


# Task_2/
- 主要是outline的data的preprocessing的constribution
- 主要的数据是使用的SPX500和NASDAQ

# Code Strcture
- finplot_usage/, 主要使用finplot对stock data进行visualization
- data/, 包含一些raw data
- Interactive_Plot/, 主要是一些finplot 的 tutorials
- ta/, 主要是一些indicators 的 implementation，分为了以下几种，每一个indicator都有comments，里面有对应的formula：
  - momentum
  - volatility
  - volume
  - my
  还有一个wrapper，用来封装这些indicators


# Main Usage
 ```  python main.py ```
 可以通过command ```python main.py -h ``` 查看使用方法
 

# Installation
``` pip install tushare pandas numpy yahoo talib ```


# Updating ... 



