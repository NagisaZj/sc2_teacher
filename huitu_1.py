import numpy as np
import matplotlib.pyplot as plt

nopre_5 = np.fromfile("1e-5_nopre.bin", dtype=np.float32)
nopre_4 = np.fromfile("1e-4_nopre.bin", dtype=np.float32)
nopre_3 = np.fromfile("1e-3_nopre.bin", dtype=np.float32)
pre_5 = np.fromfile("1e-5_pre.bin", dtype=np.float32)
pre_4 = np.fromfile("1e-4_pre.bin", dtype=np.float32)
pre_3 = np.fromfile("1e-3_pre.bin", dtype=np.float32)
n5 = range(len(nopre_5))
n4 = range(len(nopre_4))
n3 = range(len(nopre_3))
p5 = range(len(pre_5))
p4 = range(len(pre_4))
p3 = range(len(pre_3))
print(n5)
plt.plot(n5,nopre_5,label='1e-5')
plt.plot(n4,nopre_4,label='1e-4')
plt.plot(n3,nopre_3,label='1e-3')
plt.legend()
#plt.show()
foo_fig = plt.gcf() # 'get current figure'
foo_fig.savefig('nopre_difflr.eps', format='eps', dpi=500)
plt.show()
#plt.plot(rnn,'b')
#plt.plot(adam,'r')
#