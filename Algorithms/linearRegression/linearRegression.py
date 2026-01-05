import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


# 生成數據集
x = np.linspace(-3,3,30)    #返回-3到3區間内,間隔均匀的30個數值
y = 2*x+ 1

#繪製數據集
plt.scatter(x,y)
plt.show()


x = [[i] for i in x]
y = [[i] for i in y]
x_ = [[1],[2]]              #測試用
'''
列表推导式, 将一维数组x转换为二维嵌套列表 (形状为 [30, 1])
原x是一维结构: [x0, x1, x2, ..., x29] 
转换后x是二维结构: [[x0], [x1], [x2], ..., [x29]]S
核心原因: scikit-learn的线性模型要求特征数据 (自变量) 必须是二维数组 (矩阵形式) ,即使只有 1 个特征 (单变量线性回归),也不能传入一维数组。


定义测试特征数据x_, 是一个二维嵌套列表,包含两个测试样本 1和2,用于后续预测模型的输出结果。
关键注意点：测试数据的格式必须与训练数据(x)的格式保持一致,均为二维结构,否则模型会报错。
'''



model = linear_model.LinearRegression() 
model.fit(x,y)  
model.predict(x_)
'''
LinearRegression()是scikit-learn中普通最小二乘法线性回归的实现类，初始化时默认不添加正则化，适合普通的线性拟合场景。
fit()方法的核心作用：根据输入的训练特征x和训练标签y，通过最小二乘法求解最优的回归系数（斜率）和截距，完成模型的参数拟合，拟合后的参数会存储在model实例中。
'''







"""
\r\n修改注释行不通

git commit -m "feat: 增加二号文件, 和1号注释, 修改注释回车 
            -增加1号注释
            -增加二号注释"


"""