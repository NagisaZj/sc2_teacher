import numpy
def action(obs,info):
   if info[1]:
      #print(obs[:,:,4].nonzero())
      neutral_y, neutral_x = obs[:,:,4].nonzero()
      player_y, player_x = obs[:,:,3].nonzero()

      player = [int(player_x.mean()), int(player_y.mean())]
      closest, min_dist = None, None
      for p in zip(neutral_x, neutral_y):
          dist = numpy.linalg.norm(numpy.array(player) - numpy.array(p))
          if not min_dist or dist < min_dist:
              closest, min_dist = p, dist
      if closest != None:
          return 2,closest[0],closest[1]
      else:
          return 0,0,0
   else:
      return 0,0,0
