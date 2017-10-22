import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import shlex
import sys



k = map(int, sys.argv[1])[0]
f = open('res.txt', 'r')
g = [0]*k
for i in range(0, k):
    g[i] = shlex.split(f.readline())
    g[i] = map(float, g[i])
    print g[i], '\n'
f.close()

for i in range(0, k):
    x = [g[i][2*j] for j in range(0, len(g[i])/2)]
    y = [g[i][2*j + 1] for j in range(0, len(g[i])/2)]
    
    plt.plot(x, y, color = clr.hsv_to_rgb((1./k*i, 1., 1.)), marker = 'o', linestyle="None");


#plt.plot(x1, y1, 'ro')
#plt.plot(x2, y2, 'go')
plt.show()

