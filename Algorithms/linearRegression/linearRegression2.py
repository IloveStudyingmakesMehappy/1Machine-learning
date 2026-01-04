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

#22222222222222222