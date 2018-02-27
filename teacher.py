def action(obs):
      neutral_y, neutral_x = obs[4].nonzero()
      player_y, player_x = obs[3].nonzero()
      if not neutral_y.any() or not player_y.any():
          return [1,0,0]
      player = [int(player_x.mean()), int(player_y.mean())]
      closest, min_dist = None, None
      for p in zip(neutral_x, neutral_y):
          dist = numpy.linalg.norm(numpy.array(player) - numpy.array(p))
          if not min_dist or dist < min_dist:
              closest, min_dist = p, dist
      return [2,closest[0],closest[1]]
