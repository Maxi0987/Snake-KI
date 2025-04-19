import os
from agent import DQNAgent
from snake_env import SnakeEnv

SAVE_DIR = "models/V4"
os.makedirs(SAVE_DIR, exist_ok=True)

env = SnakeEnv(render=False)
state_size = 12
action_size = 3
agent = DQNAgent(state_size, action_size, model_name="snake_dqn")

episodes = 1000
batch_size = 64
save_interval = 25

for e in range(1, episodes + 1):

    state = env.reset()
    done = False
    score = 0
    total_reward = 0
    step = 0
    steps_since_last_food = 0
    max_no_food_steps = 200

    while not done:
        action = agent.act(state)
        next_state, reward, done, score = env.step(action)

        if reward == -10:
            # Kollision – Spiel endet sofort
            done = True
        else:
            if reward == 10:
                steps_since_last_food = 0
            else:
                steps_since_last_food += 1
                reward -= 0.1  # kleine Strafe fürs Rumstehen
            if steps_since_last_food > max_no_food_steps:
                reward = -10
                done = True


        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        step += 1

    agent.replay(batch_size)

    print(f"Episode {e:03}/{episodes} | Score: {score:3} | Reward: {total_reward:.1f} | Epsilon: {agent.epsilon:.8f} | Steps: {step:3}")

    if e % save_interval == 0:
        model_path = os.path.join(SAVE_DIR, f"snake_model_v{e}.h5")
        agent.save(model_path)
        print(f"✅ Modell gespeichert: {model_path}")

env.close()
