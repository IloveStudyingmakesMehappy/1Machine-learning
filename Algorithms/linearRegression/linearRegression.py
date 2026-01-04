import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3,3,30)
y = 2*x+ 1

plt.scatter(x,y)
#plt.show()

print(x)
print(y)

x = [[i] for i in x]
y = [[i] for i in y]

print(x)
print(y)

#1111111111
#\r\n修改注释行不通


"""
git commit -m "feat: 增加二号文件, 和1号注释, 修改注释回车 
            -增加1号注释
            -增加二号注释"


"""