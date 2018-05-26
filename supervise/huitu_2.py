import numpy as np
import matplotlib.pyplot as plt

cp = np.fromfile("curri_pre_s.bin", dtype=np.int32)
cnp = np.fromfile("curri_nopre_s.bin", dtype=np.int32)
ncp = np.fromfile("nocurri_pre_s.bin", dtype=np.int32)
ncnp = np.fromfile("nocurri_nopre_s.bin", dtype=np.int32)

print(cp)
print(cnp)
print(ncp)
print(ncnp)
#plt.plot(rnn,'b')
#plt.plot(adam,'r')
#