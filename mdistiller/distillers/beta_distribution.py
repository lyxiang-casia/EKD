import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# 定义Beta分布的参数
alpha_values = [0.5, 2, 2, 5]
beta_values = [0.5, 2, 5, 1]

# 创建x轴上的点
x = np.linspace(0, 1, 100)

# 创建一个图形
plt.figure(figsize=(10, 6))

# 绘制每个参数组合的Beta分布
for alpha, beta_param in zip(alpha_values, beta_values):
    y = beta.pdf(x, alpha, beta_param)
    plt.plot(x, y, label=f'α={alpha}, β={beta_param}')

# 添加标题和标签
plt.title('Beta Distribution')
plt.xlabel('x')
plt.ylabel('Probability Density Function')
plt.legend()

# 显示图形
plt.grid(True)
print("Hello World")
plt.show()
