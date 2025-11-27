import random
from collections import Counter

import textworld
import textworld.gym
import gymnasium as gym


class AlfworldResetEnv(gym.Env):
    """
    Gym-make-style TextWorld env with reset(game_file=...).
    Construct it exactly where you'd normally do register_games + make.
    """

    def __init__(
        self,
        game_files,
        request_infos,
        max_episode_steps,
        wrappers=None,
        batch_size=1,
        asynchronous=True,
        auto_reset=False,
    ):
        self.game_files = list(game_files)
        self.request_infos = request_infos
        self.max_episode_steps = max_episode_steps
        self.wrappers = wrappers or []
        self.batch_size = batch_size
        self.asynchronous = asynchronous
        self.auto_reset = auto_reset

        self._env = self._make(self.game_files)

    def _make(self, files):
        env_id = textworld.gym.register_games(
            files,
            self.request_infos,
            batch_size=self.batch_size,
            asynchronous=self.asynchronous,
            auto_reset=self.auto_reset,
            max_episode_steps=self.max_episode_steps,
            wrappers=self.wrappers,
        )
        return textworld.gym.make(env_id)

    def reset(self, game_file=None, seed=None):
        if game_file is not None:
            self.close()
            self._env = self._make([game_file])
        if seed is not None and hasattr(self._env, "seed"):
            self._env.seed(seed)
        obs, infos = self._env.reset()
        return obs, infos

    def step(self, actions):
        # Manually hacking because alfworld doesn't return 'look' when you include description in the env_infos
        obs, scores, dones, infos = self._env.step(actions)
        return obs, scores, dones, infos

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self):
        if hasattr(self._env, "close"):
            self._env.close()

    def __getattr__(self, name):
        # forward anything else (action_space, observation_space, etc.)
        return getattr(self._env, name)



# Alfworld Demanglers
class Demangler(object):

    def __init__(self, obj_dict=None, game_infos=None, shuffle=False):
        if obj_dict is None:
            self.obj_count = Counter()
        else:
            self.obj_count = obj_dict

        self.obj_names = {}
        if game_infos:
            ids = sorted([info.id for info in game_infos.values()])
            if shuffle:
                random.shuffle(ids)

            # count the number of instances
            for id in ids:
                splits = id.split("_bar_", 1)
                if len(splits) > 1:
                    name, rest = splits
                    if "basin" in id:
                        name += "basin"
                    self.obj_count[name] += 1

            # make list of num ids for each object (shuffle the ids if required)
            obj_num_ids = {}
            for obj, count in self.obj_count.most_common():
                num_ids = list(range(count+1)[1:])  # start from index 1
                obj_num_ids[obj] = num_ids

            # assign unique num id for each object based on precomputed list
            for id in ids:
                text = id
                text = text.replace("_bar_", "|")
                text = text.replace("_minus_", "-")
                text = text.replace("_plus_", "+")
                text = text.replace("_dot_", ".")
                text = text.replace("_comma_", ",")

                splits = text.split("|", 1)
                if len(splits) == 1:
                    self.obj_names[id] = {'name': text, 'id': 0}
                else:
                    name, rest = splits
                    if "basin" in id:
                        name += "basin"
                    self.obj_names[id] = {'name': name, 'id': obj_num_ids[name].pop()}

    def demangle_alfred_name(self, text):
        assert(text in self.obj_names)
        name, id = self.obj_names[text].values()
        id = str(id) if id > 0 else ""
        res = "{} {}".format(name, id)
        return res


class AlfredDemangler(textworld.core.Wrapper):

    def __init__(self, *args, shuffle=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.shuffle = shuffle

    def load(self, *args, **kwargs):
        super().load(*args, **kwargs)

        demangler = Demangler(game_infos=self._entity_infos, shuffle=self.shuffle)
        for info in self._entity_infos.values():
            info.name = demangler.demangle_alfred_name(info.id)

