from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import os
from envs.env import LEOEnv
import logging
import json
from datetime import datetime

from envs.param import STEP_PER_SECOND, STEP_PER_SLOT, T_SLOT
import sys

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
    print("No trained model found. Please run 'python3 train.py' to train a model or provide 'model/ppo_leoenv.zip'. Exiting.")
    sys.exit(1)

# 测试
obs, _ = env.reset()

T = 2 * STEP_PER_SLOT

for _ in range(T):
    action, _ = model.predict(obs)
    obs, reward, done, _, info = env.step(action)
    # print(info.__str__())
    
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