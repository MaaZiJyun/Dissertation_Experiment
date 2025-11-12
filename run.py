from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import os
from envs.env import LEOEnv

# 环境实例化
env = LEOEnv(json_path="data/parsed_data.json")

# 检查环境符合 Gym 接口规范
check_env(env, warn=True)

model_path = "model/ppo_leoenv"

# 如果已有模型则加载，否则创建并训练新模型
if os.path.exists(model_path + ".zip"):
    
    print("Loading existing model...")
    model = PPO.load(model_path, env=env)
    
else:
    
    print("Training new model...")
    
    # 创建新的 PPO 模型
    model = PPO(
        policy = "MlpPolicy", 
        env = env, 
        verbose = 1, 
        learning_rate = 3e-4, 
        batch_size = 64, 
        n_steps = 2048
    )
    
    # 训练模型
    model.learn(total_timesteps=100000)
    
    # 保存模型
    model.save(model_path)

# 测试
obs, _ = env.reset()

for _ in range(10000):
    action, _ = model.predict(obs)
    obs, reward, done, _, info = env.step(action)
    env.render()
    if done:
        break