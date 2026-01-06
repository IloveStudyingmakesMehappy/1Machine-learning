import numpy as np
import matplotlib.pyplot as plt

# ===================== 步骤1:生成带轻微噪声的数据集(更贴近真实场景,便于验证) =====================
# 生成自变量x(一维数组)
x = np.linspace(-3, 3, 30)  # [-3, 3]区间内30个均匀分布的数值
# 生成因变量y:基于线性关系y=2x+1,添加轻微正态分布噪声(避免完美拟合,更易验证模型有效性)
np.random.seed(42)  # 设置随机种子,保证结果可复现
y = 2 * x + 1 + np.random.normal(0, 0.5, size=len(x))  # 噪声均值0,标准差0.5

# ===================== 步骤2:最小二乘法手动求解线性回归参数(w:斜率,b:截距) =====================
n = len(x)  # 样本数量
x_mean = np.mean(x)  # x的均值
y_mean = np.mean(y)  # y的均值

# 计算分子:Σ(xi - x_mean)(yi - y_mean)
numerator = np.sum((x - x_mean) * (y - y_mean))
# 计算分母:Σ(xi - x_mean)²
denominator = np.sum((x - x_mean) ** 2)

# 求解核心参数
w = numerator / denominator  # 斜率(回归系数)
b = y_mean - w * x_mean      # 截距 y的均值-斜率*x的均值

print("=== 线性回归模型参数 ===")
print(f"斜率 w = {w:.4f}")
print(f"截距 b = {b:.4f}")
print(f"拟合线性方程:y = {w:.4f}x + {b:.4f}")

# ===================== 步骤3:模型预测(生成拟合值与测试数据预测) =====================
# 1. 对训练数据x生成拟合值(用于绘制拟合直线)
y_fit = w * x + b

# 2. 定义测试数据并进行预测(数据验证)
x_test = np.array([1, 2])  # 测试自变量(与之前场景一致) 
'''
np.array([1, 2]): 创建一个包含数值 1 和 2 的一维 NumPy 数组, 并将其命名为 x_test,
这个数组的形状为 (2,)（表示包含 2 个元素的一维数组）
'''
y_test_pred = w * x_test + b  # 基于求解的参数计算预测值

print("\n=== 测试数据预测结果 ===")
for xi, y_pred in zip(x_test, y_test_pred):
    print(f"x = {xi} 对应的预测值 y = {y_pred:.4f}")

# ===================== 步骤4:数据验证(计算R²拟合优度,评估模型效果) =====================
# R²公式:R² = 1 - [Σ(yi - y_fit)² / Σ(yi - y_mean)²]
ss_res = np.sum((y - y_fit) ** 2)  # 残差平方和(模型无法解释的变异)
ss_tot = np.sum((y - y_mean) ** 2)  # 总平方和(y的总变异)
r2_score = 1 - (ss_res / ss_tot)

print("\n=== 模型拟合效果验证 ===")
print(f"R² 拟合优度:{r2_score:.4f}")
print(f"说明:R²越接近1,模型拟合效果越好(本次因噪声较小,R²接近1)")

# ===================== 步骤5:matplotlib 可视化绘图(对比原始数据与拟合结果) =====================
plt.figure(figsize=(10, 6))  # 设置图表大小

# 1. 绘制原始数据散点图
plt.scatter(x, y, color='steelblue', label='Original Data (with noise)', alpha=0.7, s=50)

# 2. 绘制拟合直线
plt.plot(x, y_fit, color='crimson', linewidth=2, label='Fitted Line (y=wx+b)')

# 3. 绘制测试数据预测点(突出显示,便于验证)
plt.scatter(x_test, y_test_pred, color='gold', marker='*', s=200, label='Test Data Predictions', zorder=5)

# 4. 完善图表标注(提升可读性)
plt.xlabel('X Value', fontsize=12)
plt.ylabel('Y Value', fontsize=12)
plt.title('Linear Regression with NumPy (Least Squares Method)', fontsize=14, pad=20)
plt.legend(fontsize=10)
plt.grid(alpha=0.3, linestyle='--', color='gray')

# 5. 显示图表
plt.show()