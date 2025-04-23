import os
import sys
import datetime
import tensorflow as tf
from agent import DQNAgent
from snake_env import SnakeEnv

# ✅ Mixed Precision & XLA aktivieren
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
#tf.config.optimizer.set_jit(True)

# 💻 GPU-Konfiguration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("🚀 GPU wird verwendet.")
    except RuntimeError as e:
        print("⚠️ Fehler bei GPU-Initialisierung:", e)
else:
    print("🧠 Kein GPU-Gerät gefunden – CPU wird verwendet.")

# 🛠️ Logger für Konsole + Datei mit Timestamp
class Logger:
    def __init__(self, logfile_path):
        self.terminal = sys.__stdout__
        self.log = open(logfile_path, "a", encoding="utf-8")

    def write(self, message):
        if message.strip():
            timestamped = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {message}"
            if not message.endswith('\n'):
                timestamped += '\n'
            self.terminal.write(timestamped)
            self.terminal.flush()
            self.log.write(timestamped)
            self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger("log/train.log")

# 📁 Modellverzeichnis
SAVE_DIR = "models/V5"
os.makedirs(SAVE_DIR, exist_ok=True)

# 🐍 Umgebung & Agent
env = SnakeEnv(render=False)
state_size = 12
action_size = 3
agent = DQNAgent(state_size, action_size, model_name="snake_dqn")

# ⚙️ Trainingseinstellungen
episodes = 1000
batch_size = 64
save_interval = 25
max_no_food_steps = 200

for e in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0
    total_reward = 0
    step = 0
    steps_since_last_food = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done, score = env.step(action)

        if reward == -10:
            done = True
        else:
            if reward == 10:
                steps_since_last_food = 0
            else:
                steps_since_last_food += 1
                reward -= 0.1
            if steps_since_last_food > max_no_food_steps:
                reward = -10
                done = True

        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        step += 1

    agent.replay(batch_size)

    print(f"Episode {e:04}/{episodes} | Score: {score:3} | Reward: {total_reward:.1f} | Epsilon: {agent.epsilon:.8f} | Steps: {step:3}")

    if e % save_interval == 0:
        model_path = os.path.join(SAVE_DIR, f"snake_model_v{e}.h5")
        agent.save(model_path)
        print(f"✅ Modell gespeichert: {model_path}")

env.close()
