import gymnasium

env = gymnasium.make('LunarLander-v2', render_mode='human')  # Specify render_mode as "human"
env.reset()

for step in range(200):
    env.render()
    # take random action
    env.step(env.action_space.sample())

env.close()
