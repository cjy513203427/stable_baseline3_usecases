import gymnasium

# Create the environment
env = gymnasium.make('LunarLander-v2')  # continuous: LunarLanderContinuous-v2

# required before you can step the environment
env.reset()

# sample action:
print("sample action:", env.action_space.sample())

# observation space shape:
print("observation space shape:", env.observation_space.shape)

# sample observation:
"""
    env: The environment
    s (list): The state. Attributes:
        s[0] is the horizontal coordinate
        s[1] is the vertical coordinate
        s[2] is the horizontal speed
        s[3] is the vertical speed
        s[4] is the angle
        s[5] is the angular speed
        s[6] 1 if first leg has contact, else 0
        s[7] 1 if second leg has contact, else 0
"""
print("sample observation:", env.observation_space.sample())

env.close()
