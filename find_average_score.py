import os
import sys
import datetime
from agent import DQNAgent
from snake_env import SnakeEnv

# 🛠️ Logger-Klasse für Terminal + Datei mit Timestamp
class Logger:
    def __init__(self, logfile_path):
        self.terminal = sys.__stdout__
        self.log = open(logfile_path, "a", encoding="utf-8")

    def write(self, message):
        if message.strip():  # keine leeren Zeilen loggen
            timestamped = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {message}"
            self.terminal.write(timestamped + "\n")
            self.log.write(timestamped + "\n")

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 📁 Log aktivieren
logfile = Logger("log/evaluation.log")
sys.stdout = logfile

# 📁 Modelleinstellungen
MODEL_DIR = "models/V4"
EPISODES_PER_MODEL = 3
MAX_STEPS_WITHOUT_FOOD = 200

state_size = 12
action_size = 3

best_score = -1
best_model = None
results = []

# 🔍 Durchsuche Modellordner
for filename in sorted(os.listdir(MODEL_DIR)):
    if filename.endswith(".h5"):
        model_path = os.path.join(MODEL_DIR, filename)
        print(f"🔍 Teste Modell: {filename}")
        
        env = SnakeEnv(render=False)
        agent = DQNAgent(state_size, action_size)
        agent.load(model_path)
        agent.epsilon = 0.0  # Nur best move

        total_score = 0

        for _ in range(EPISODES_PER_MODEL):
            state = env.reset()
            done = False
            score = 0
            steps_without_food = 0
            previous_score = 0

            while not done:
                action = agent.act(state)
                next_state, reward, done, score = env.step(action)
                state = next_state

                if score > previous_score:
                    steps_without_food = 0
                    previous_score = score
                else:
                    steps_without_food += 1

                if steps_without_food > MAX_STEPS_WITHOUT_FOOD:
                    done = True

            total_score += score

        avg_score = total_score / EPISODES_PER_MODEL
        results.append((filename, avg_score))

        if avg_score > best_score:
            best_score = avg_score
            best_model = filename

        print(f"📊 {filename} → ⌀ Score: {avg_score:.2f}")

# 🏆 Ergebnis
print("\n🏆 Bestes Modell:")
print(f"{best_model} mit durchschnittlichem Score von {best_score:.2f}")

# 🔚 Logger zurücksetzen
logfile.log.close()
sys.stdout = sys.__stdout__
