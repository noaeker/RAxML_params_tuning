import numpy as np
a = np.arange(9).reshape(3, 3)
il1 = np.triu_indices(3,1)
print(a[il1])
