import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import mixed_precision

# ✅ Mixed Precision aktivieren
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# ✅ GPU konfigurieren
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU wird verwendet.")
    except RuntimeError as e:
        print("⚠️ Fehler bei GPU-Initialisierung:", e)
else:
    print("🧠 Keine GPU gefunden – Training läuft auf CPU.")

class DQNAgent:
    def __init__(self, state_size, action_size, model_name="snake_model"):
        self.state_size = state_size
        self.action_size = action_size
        self.model_name = model_name

        # Hyperparameter
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.998
        self.learning_rate = 0.001

        self.memory = deque(maxlen=2000)
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear', dtype='float32')  # Output bleibt float32
        ])
        base_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        optimizer = mixed_precision.LossScaleOptimizer(base_optimizer)
        model.compile(loss='mse', optimizer=optimizer)
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states = []
        targets = []

        for state, action, reward, next_state, done in minibatch:
            # Fix: expand dims for correct input shape
            next_qs = self.model(tf.convert_to_tensor(np.expand_dims(next_state, axis=0), dtype=tf.float32))
            target = reward
            if not done:
                target += self.gamma * tf.reduce_max(next_qs[0])

            current_qs = self.model(tf.convert_to_tensor(np.expand_dims(state, axis=0), dtype=tf.float32))[0]
            current_qs = current_qs.numpy()
            current_qs[action] = target

            states.append(state)
            targets.append(current_qs)

        self.model.fit(tf.convert_to_tensor(states, dtype=tf.float32),
                       tf.convert_to_tensor(targets, dtype=tf.float32),
                       epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename=None):
        if filename is None:
            filename = f"{self.model_name}.h5"
        self.model.save(filename)

    def load(self, filename):
        self.model = self._build_model()
        self.model.load_weights(filename)
