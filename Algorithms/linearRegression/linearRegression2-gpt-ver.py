import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# 生成數據集
x = np.linspace(-3,3,30)    #返回-3到3區間内，間隔均匀的30個數值
y = x+np.random.rand(30)    #让y值在y=2x+1直线上下浮动
#y = 2*x+ 1                 #让样本点完全在y=2x+1直线上

# 繪製原始數據集(可选，保留原代码功能)
plt.scatter(x, y, label='Original Data', color='blue')

# 数据格式转换(适配scikit-learn模型要求)
x = [[i] for i in x]
y = [[i] for i in y]
x_ = [[1],[2]]              # 測試用

# 初始化并训练线性回归模型
model = linear_model.LinearRegression()
model.fit(x, y)

# 【补全1:执行预测并保存/打印预测结果】
y_pred = model.predict(x_)  # 保存预测结果到变量y_pred
print("测试数据 x_ = [[1],[2]] 对应的预测结果:")
print(y_pred)

# 【补全2:提取并打印模型关键参数(斜率&截距)】
# 提取斜率(回归系数):coef_ 属性，返回二维数组，对应每个特征的回归系数
slope = model.coef_[0][0]       #coef_是法向量w，intercept_是截距b
''' model.coef_[0][0] 提取到的是单变量线性回归模型中的「斜率(slope)」，对应一元线性回归方程 y = wx + b 中的权重系数 w(也称为回归系数)
'''

# 提取截距:intercept_ 属性，返回一维数组，对应线性模型的截距项
intercept = model.intercept_[0]
print("\n线性回归模型参数:")
print(f"斜率(回归系数): {slope:.4f}")
print(f"截距: {intercept:.4f}")
print(f"拟合的线性方程:y = {slope:.4f}x + {intercept:.4f}")

# 【补全3:生成完整的拟合直线数据(用于绘图)】
# 为了绘制平滑的拟合直线，使用原始一维x数据生成对应的预测值
x_plot = np.linspace(-3, 3, 30)  # 与原始数据区间一致的x轴数据
x_plot_2d = [[i] for i in x_plot]  # 转换为二维格式(适配模型要求)
y_fit = model.predict(x_plot_2d)  # 生成拟合直线的y值

# 【补全4:绘制拟合直线并完善可视化图表】
plt.plot(x_plot, y_fit, label='Fitted Line', color='red', linewidth=2)
plt.xlabel('X Value')  # x轴标签
plt.ylabel('Y Value')  # y轴标签
plt.title('Linear Regression: Original Data vs Fitted Line')  # 图表标题
plt.legend()  # 显示图例
plt.grid(alpha=0.3)  # 显示网格(增强可读性)
plt.show()

# 【补全5:可选:评估模型拟合效果(因无噪声，拟合度为1.0)】
from sklearn.metrics import r2_score  # 导入R²评分函数
y_true = np.array(y).flatten()  # 转换为一维数组(适配r2_score要求)
y_fit_flatten = np.array(y_fit).flatten()  # 转换为一维数组
r2 = r2_score(y_true, y_fit_flatten)
print(f"\n模型拟合优度(R²):{r2:.4f}")