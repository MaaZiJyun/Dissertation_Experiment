from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import os
from envs.env import LEOEnv
import logging
import json
from datetime import datetime

# 环境实例化
env = LEOEnv(json_path="data/parsed_data.json")

# 检查环境符合 Gym 接口规范
check_env(env, warn=True)

model_path = "model/ppo_leoenv"

# setup logging to file
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logger = logging.getLogger("leo.run")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(log_file)
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

# 如果已有模型则加载，否则创建并训练新模型
if os.path.exists(model_path + ".zip"):
    
    print("Loading existing model...")
    model = PPO.load(model_path, env=env)
    
else:
    
    print("Training new model...")
    
    # 创建新的 PPO 模型
    # choose policy type based on observation space
    policy_choice = "MlpPolicy"
    try:
        from gymnasium import spaces as gspaces
        if isinstance(env.observation_space, gspaces.Dict):
            policy_choice = "MultiInputPolicy"
    except Exception:
        # fallback: if we can't import gymnasium, inspect type name
        if env.observation_space.__class__.__name__.endswith("Dict"):
            policy_choice = "MultiInputPolicy"

    model = PPO(
        policy = policy_choice,
        env = env,
        verbose = 1,
        learning_rate = 3e-4,
        batch_size = 64,
        n_steps = 2048
    )
    
    # 训练模型
    model.learn(total_timesteps=50000)
    
    # 保存模型
    model.save(model_path)

# 测试
obs, _ = env.reset()

for _ in range(10000):
    action, _ = model.predict(obs)
    obs, reward, done, _, info = env.step(action)
    # write info to log file (JSON serializable if possible)
    try:
        logger.info(json.dumps(info, default=lambda o: str(o)))
    except Exception:
        try:
            logger.info(str(info))
        except Exception:
            pass

    env.render()
    if done:
        break