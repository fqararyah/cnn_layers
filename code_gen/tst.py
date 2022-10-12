a = 11

b = a << 4 | 7

print(b & 15)
print((b & (15 << 4)) >> 4)

import numpy as np

a = np.random.randint(low=0, high=8, size=(4,4,2,2))

print(a)
print('***************')
if a.shape[0] % 3 != 0:
    a = np.append(a, np.zeros(( int(3 - a.shape[0] % 3), a.shape[1], a.shape[2], a.shape[3])), 0).astype(np.int8)
print(a)
print('***************')
b = np.split(a, int(a.shape[0] / 3), 0)
b = [np.transpose(b[i], (1,2,3,0)) for i in range(len(b))]
c = np.concatenate(b, 0)
c = c.reshape(c.size) 
print(c)
print('***************')