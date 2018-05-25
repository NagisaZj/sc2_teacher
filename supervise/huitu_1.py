import numpy as np
import matplotlib.pyplot as plt

e0_1 = np.fromfile("sup_0.1.bin", dtype=np.float32)
e1_0 = np.fromfile("sup_1.0.bin", dtype=np.float32)
e5_0 = np.fromfile("sup_5.0.bin", dtype=np.float32)

q0_1 = range(len(e0_1))
q1_0 = range(len(e1_0))
q5_0 = range(len(e5_0))



plt.plot(q0_1,e0_1,label='sup_0.1')
plt.plot(q1_0,e1_0,label='sup_1.0')
plt.plot(q5_0,e5_0,label='sup_5.0')
plt.legend()
#plt.show()
#foo_fig = plt.gcf() # 'get current figure'
#foo_fig.savefig('nopre_difflr.eps', format='eps', dpi=500)
plt.show()
#plt.plot(rnn,'b')
#plt.plot(adam,'r')
#