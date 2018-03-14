from pysc2.lib import actions
from pysc2.env import sc2_env
from pysc2.lib import features
from absl import flags
import sys
import pdb
import numpy as np

FLAGS=flags.FLAGS

# Important hparams
flags.DEFINE_string("game", "CollectMineralShards_2", "game map name")

# Irrelevant hparams
flags.DEFINE_bool("render", False, "Whether to render with pygame.")
flags.DEFINE_bool("verbose", True, "increase output verbosity")

# Environment hparams
flags.DEFINE_integer("screen_resolution", 64,
                     "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64,
                     "Resolution for minimap feature layers.")
flags.DEFINE_integer("max_agent_steps", 2500, "Total agent steps.")
flags.DEFINE_integer("game_steps_per_episode", 0, "Game steps per episode.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_bool("save_replay", False, "Whether to save a replay at the end.")
flags.DEFINE_float("action_cost", 1e-3, "What does it take to make an action")
flags.DEFINE_float("time_cost", 0, "each time step cost certain value")
flags.DEFINE_float("reward_scaling", 1.0, "adjust reward to proper scale")
flags.DEFINE_float("invalid_cost", 0, "cost for invalid action")

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_SELECTED = features.SCREEN_FEATURES.selected.index

_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]

class scenv:
    def __init__(self,name):
        self.env = sc2_env.SC2Env(
            map_name=name,
            step_mul=FLAGS.step_mul,
            game_steps_per_episode=FLAGS.game_steps_per_episode,
            screen_size_px=(
                FLAGS.screen_resolution,
                FLAGS.screen_resolution),
            minimap_size_px=(
                FLAGS.minimap_resolution,
                FLAGS.minimap_resolution),
            visualize=FLAGS.render)
        self.history = None
        self.ts = None

    def action_shape(self):
        # no_op + select_all, move map size
        return 2, (FLAGS.screen_resolution, FLAGS.screen_resolution, 1)

    def state_shape(self):
        # image + last_image + selection_map: 2 + 2 + 1
        return (FLAGS.screen_resolution, FLAGS.screen_resolution, 5)

    def render(self):
        return self.env.render()

    def reset(self):
        self.ts = self.env.reset()
        #self.history = None
        return self.trans_ts()

    def trans_ts(self):
        obs = self.ts[0]
        feature = obs.observation["screen"][_PLAYER_RELATIVE]
        selected = obs.observation["screen"][_SELECTED].astype(np.float32)
        available = obs.observation["available_actions"]
        a1 = 1 if _SELECT_ARMY in available else 0
        a2 = 1 if _MOVE_SCREEN in available else 0
        f1 = ((feature == 1).astype(np.float32))
        f2 = ((feature == 3).astype(np.float32))

        if self.history is None:
            state = np.stack((selected, f1,f2, f1, f2), axis=2)
        else:
            state = np.stack(
                (selected,
                 self.history[0],self.history[1],
                 f1, f2),
                axis=2)
        reward = obs.reward
        done = obs.last()
        self.info = np.array([a1,a2])

        self.history = (f1, f2)
        return state, reward, done, self.info

    def close(self):
        return self.env.close()

    def step(self,action):
        if (not self.info[1]) and action >1 :
            state, reward, done, info = self.trans_ts()
            reward = -FLAGS.action_cost - FLAGS.time_cost - FLAGS.invalid_cost
            reward *= FLAGS.reward_scaling

            return state, reward, done, info

        action_cost = FLAGS.action_cost
        if action ==0:
            act_fun = actions.FunctionCall(_NO_OP,[])
            action_cost = 0
        elif action ==1 :
            act_fun = actions.FunctionCall(_SELECT_ARMY,[_SELECT_ALL])
        else:
            index = action - 2
            neutral_x = index % FLAGS.screen_resolution
            neutral_y = index // FLAGS.screen_resolution
            target =  [neutral_y, neutral_x]
            act_fun = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED,target])

        self.ts = self.env.step([act_fun])

        state, reward, done, info = self.trans_ts()

        reward -= action_cost
        reward -= FLAGS.time_cost

        reward *= FLAGS.reward_scaling

        return state, reward, done, info

def wrap(name):
    FLAGS(sys.argv)
    sc_wrapper = scenv(name)
    return sc_wrapper

if __name__=="__main__":
    FLAGS(sys.argv)
    sc_wrapper = scenv("CollectMineralShards_20")
    ts = sc_wrapper.reset()
    step_sum = 0
    try:
        while True:
            if _MOVE_SCREEN in sc_wrapper.ts[0].observation["available_actions"]:
                ts = sc_wrapper.step(1000*(step_sum % 2))
                state = ts[0]
                print(state.shape, state.min(), state.max(), state.mean())
            else:
                print(step_sum, "Select all")
                ts = sc_wrapper.step(0)
            step_sum = step_sum + 1
    except KeyboardInterrupt:
        sc_wrapper.close()




