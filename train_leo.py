from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.env import MultiSatInferenceEnv


from stable_baselines3 import PPO

env = MultiSatInferenceEnv(n_sats=10, n_layers=6)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200_000)

