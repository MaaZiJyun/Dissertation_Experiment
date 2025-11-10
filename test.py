import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env_name = "LunarLander-v3"
# use render_mode="human" so render() is supported by gymnasium
env = gym.make(env_name)

env = DummyVecEnv([lambda : env])

model = DQN(
    "MlpPolicy", 
    env=env,
    verbose=1
)

model.learn(total_timesteps=1e5)