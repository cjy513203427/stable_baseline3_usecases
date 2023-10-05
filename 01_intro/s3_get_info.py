import gymnasium

env = gymnasium.make('LunarLander-v2', render_mode='human')  # Specify render_mode as "human"
env.reset()

for step in range(200):
    env.render()
    # take random action
    # step return 5 params, so we unpack it.
    step_results = env.step(env.action_space.sample())  # Store the returned tuple
    obs, reward, done, info = step_results[:4]  # Unpack the first four values
    print(reward, done)

env.close()
