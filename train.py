from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os
from envs.env import LEOEnv
import argparse


def make_env(seed: int = 0):
    def _init():
        base_env = LEOEnv(json_path="data/parsed_data.json")
        try:
            base_env.reset(seed=seed)
        except TypeError:
            try:
                base_env.seed(seed)
            except Exception:
                pass
        return Monitor(base_env)
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=50000)
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--model-path', type=str, default='model/ppo_leoenv')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    env_fns = [make_env(seed=i) for i in range(args.num_envs)]
    venv = DummyVecEnv(env_fns)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10., norm_obs_keys=[
        "energy", "sunlight", "comm", "location", "progress", "size", "workload",
    ])

    # choose policy
    policy_choice = "MlpPolicy"
    try:
        from gymnasium import spaces as gspaces
        if isinstance(venv.observation_space, gspaces.Dict):
            policy_choice = "MultiInputPolicy"
    except Exception:
        if venv.observation_space.__class__.__name__.endswith("Dict"):
            policy_choice = "MultiInputPolicy"

    model = PPO(policy=policy_choice, env=venv, verbose=1)
    model.learn(total_timesteps=args.timesteps)
    model.save(args.model_path)
    venv.save(args.model_path + '.vecnormalize')


if __name__ == '__main__':
    main()
