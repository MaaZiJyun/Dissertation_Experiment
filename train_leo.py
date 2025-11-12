from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.env import LEOEnv
from train_leo import GymToSB3Wrapper  # if you kept that wrapper in train_leo.py

JSON = "data/parsed_data.json"

def make_env():
    env = LEOEnv(JSON)
    return Monitor(GymToSB3Wrapper(env))  # Monitor helps SB3 record episode info

ven = DummyVecEnv([make_env])
model = PPO("MlpPolicy", ven, verbose=1)
model.learn(total_timesteps=2000)   # short run for smoke test
model.save("ppo_leo_smoke")