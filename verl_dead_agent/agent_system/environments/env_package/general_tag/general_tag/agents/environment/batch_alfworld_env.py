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
