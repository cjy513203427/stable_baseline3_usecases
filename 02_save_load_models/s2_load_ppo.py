import gymnasium
from stable_baselines3 import PPO

models_dir = "models/PPO"

env = gymnasium.make('LunarLander-v2', render_mode='human')
env.reset()

model_path = f"{models_dir}/10000.zip"
model = PPO.load(model_path, env=env)

episodes = 5

for ep in range(episodes):
    obs = env.reset()
    # unpack nd array
    obs_ndarray, _ = obs
    done = False
    while not done:
        action, _states = model.predict(obs_ndarray)
        step_results = env.step(action)
        obs, rewards, done, info = step_results[:4]
        env.render()
        print(rewards)
