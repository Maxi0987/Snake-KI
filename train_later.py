import os
import sys
import datetime
import re
import tensorflow as tf
from agent import DQNAgent
from snake_env import SnakeEnv

# 💻 GPU Setup
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

# 📋 Logger für Konsole + Datei
class Logger:
    def __init__(self, logfile_path):
        self.terminal = sys.__stdout__
        self.log = open(logfile_path, "a", encoding="utf-8")

    def write(self, message):
        if message.strip():
            timestamped = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {message.rstrip()}"
            self.terminal.write(timestamped + "\n")
            self.terminal.flush()
            self.log.write(timestamped + "\n")
            self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 🪵 Logging aktivieren
log_path = "log/train.log"
sys.stdout = Logger(log_path)

# 📁 Modellverzeichnis
SAVE_DIR = "models/V4"
os.makedirs(SAVE_DIR, exist_ok=True)

# 🔍 Letzte gespeicherte Episode finden
def find_latest_model_info():
    models = [f for f in os.listdir(SAVE_DIR) if f.startswith("snake_model_v") and f.endswith(".h5")]
    latest_ep = 0
    latest_file = None
    for f in models:
        match = re.search(r"v(\d+)", f)
        if match:
            num = int(match.group(1))
            if num > latest_ep:
                latest_ep = num
                latest_file = f
    return latest_ep, os.path.join(SAVE_DIR, latest_file) if latest_file else None

# 🔍 Letztes Epsilon aus Log-Datei extrahieren
def get_last_epsilon(log_file, latest_episode):
    if not os.path.exists(log_file):
        return 1.0  # Default

    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    epsilon_pattern = re.compile(rf"Episode {latest_episode:04}.*?Epsilon: ([0-9.]+)")
    for line in reversed(lines):
        match = epsilon_pattern.search(line)
        if match:
            return float(match.group(1))
    return 1.0

# 📈 Trainingseinstellungen
episodes = 5000
batch_size = 64
save_interval = 25
max_no_food_steps = 120

# 🧠 Agent & Env
env = SnakeEnv(render=False)
state_size = 12
action_size = 3
agent = DQNAgent(state_size, action_size, model_name="snake_dqn")

# ⏪ Wiederaufnahme vorbereiten
latest_episode, latest_model_path = find_latest_model_info()
start_episode = latest_episode + 1

if latest_model_path:
    agent.load(latest_model_path)
    print(f"📦 Modell geladen: {latest_model_path}")
else:
    print("🆕 Neues Training startet – kein Modell gefunden")

agent.epsilon = get_last_epsilon(log_path, latest_episode)
print(f"🔁 Starte bei Episode {start_episode} mit Epsilon: {agent.epsilon:.8f}")

# 🧠 Haupt-Trainingsloop
try:
    for e in range(start_episode, episodes + 1):
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

    # ✅ Erfolg markieren
    with open("train_done.flag", "w") as f:
        f.write("Training abgeschlossen.")

except Exception as err:
    print(f"❌ Training wurde durch einen Fehler unterbrochen: {err}")

finally:
    env.close()
    print("🏁 Training abgeschlossen oder sauber beendet.")
