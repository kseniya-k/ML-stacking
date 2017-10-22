import random
import matplotlib.pyplot as plt
import shlex

n = 3000

f = open('in.txt', 'w')
f.write(str(n))
f.write('\n')

for i in range(0, 2*n):
    f.write(str(random.randint(0, 100)))
    f.write(' ')
    
f.write('\n')
f.close()

f = open('in.txt', 'r')
f.readline()
p = shlex.split(f.readline())
f.close()

p = map(float, p)

x = [p[2*i] for i in range(0, len(p)/2)]
y = [p[2*i + 1] for i in range(0, len(p)/2)]

plt.plot(x, y, 'ro')
plt.show() 