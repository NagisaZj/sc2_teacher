import numpy
def action(info):
   if info[1]:
      #print(obs[:,:,4].nonzero())
      return 2,numpy.random.randint(0,64),numpy.random.randint(0,64)
   else:
      return 0,0,0