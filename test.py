import numpy as np
a = np.random.uniform(-1,1,[1,2])
insert = np.zeros([1,2])
a = np.append(a,insert,1)
print(a)
