import numpy as np
import random
from collections import deque
import tensorflow as tf

# ✅ GPU-Support aktivieren (optional, aber empfohlen)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("🚀 GPU wird verwendet.")
    except RuntimeError as e:
        print("⚠️ Fehler bei der GPU-Initialisierung:", e)
else:
    print("💡 Keine GPU gefunden, Training läuft auf CPU.")

class DQNAgent:
    def __init__(self, state_size, action_size, model_name="snake_model"):
        self.state_size = state_size
        self.action_size = action_size
        self.model_name = model_name

        # Hyperparameter
        self.gamma = 0.95            # Discount-Faktor (Bellman)
        self.epsilon = 1.0           # Startwert für Exploration
        self.epsilon_min = 0.01      # Minimale Exploration
        self.epsilon_decay = 0.996   # Wie schnell Exploration sinkt
        self.learning_rate = 0.001

        self.memory = deque(maxlen=2000)  # Replay Buffer
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Zufällige Aktion
        q_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(q_values[0])  # Beste Aktion wählen

    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        states = []
        targets = []

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]), verbose=0)[0])
            target_f = self.model.predict(np.array([state]), verbose=0)[0]
            target_f[action] = target
            states.append(state)
            targets.append(target_f)

        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename=None):
        if filename is None:
            filename = f"{self.model_name}.h5"
        self.model.save(filename)

    def load(self, filename):
    # Modellarchitektur neu aufbauen
        self.model = self._build_model()

    # Gewichte separat laden
        self.model.load_weights(filename)
