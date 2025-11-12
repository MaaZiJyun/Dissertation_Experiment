from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import os

# 环境实例化
from envs.env import LEOEnv
env = LEOEnv(json_path="data/parsed_data.json")

# 检查环境符合 Gym 接口规范
check_env(env, warn=True)

model_path = "model/ppo_leoenv"
if os.path.exists(model_path + ".zip"):
    print("Loading existing model...")
    model = PPO.load(model_path, env=env)
else:
    print("Training new model...")
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, batch_size=64, n_steps=2048)
    model.learn(total_timesteps=100000)
    model.save(model_path)

# 测试
obs, _ = env.reset()
for _ in range(10000):
    action, _ = model.predict(obs)
    obs, reward, done, _, info = env.step(action)
    env.render()
    if done:
        break