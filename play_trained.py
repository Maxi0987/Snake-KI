import time
import pygame
from agent import DQNAgent
from snake_env import SnakeEnv

MODEL_PATH = "models/V4/snake_model_v3800.h5"  # ⬅️ Pfad zum gespeicherten Modell

# Initialisiere Umgebung mit Anzeige
env = SnakeEnv(render=True)
state_size = 12
action_size = 3
agent = DQNAgent(state_size, action_size)

# Modell laden
agent.load(MODEL_PATH)
agent.epsilon = 0.0  # Keine zufälligen Züge, nur "best move"

print(f"📦 Modell geladen aus: {MODEL_PATH}")
print("▶️ Starte Testlauf...")

episodes = 5

for ep in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0
    esc_pressed = False

    while not done:
        # ⌨️ ESC-Abbruch prüfen
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                esc_pressed = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                done = True
                esc_pressed = True

        if not esc_pressed:
            action = agent.act(state)
            next_state, reward, done, score = env.step(action)
            state = next_state
            time.sleep(0.01)  # für bessere visuelle Darstellung

    if esc_pressed:
        print(f"⏭️ Episode {ep} abgebrochen mit ESC - Score: {score}")
    else:
        print(f"🏁 Episode {ep} beendet – Score: {score}")

env.close()