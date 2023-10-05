import gymnasium
from stable_baselines3 import A2C

env = gymnasium.make('LunarLander-v2', render_mode='human')  # continuous: LunarLanderContinuous-v2
env.reset()

model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)
episodes = 10

# use model to play games
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

"""
Return value explanation
------------------------------------
| rollout/              |          |
|    ep_len_mean        | 97.2     |
|    ep_rew_mean        | -14.7    |
| time/                 |          |
|    fps                | 48       |
|    iterations         | 20000    |
|    time_elapsed       | 2044     |
|    total_timesteps    | 100000   |
| train/                |          |
|    entropy_loss       | -0.923   |
|    explained_variance | 0.963    |
|    learning_rate      | 0.0007   |
|    n_updates          | 19999    |
|    policy_loss        | -0.245   |
|    value_loss         | 0.127    |
------------------------------------

rollout/ep_len_mean (Episode Length Mean): The average number of steps (time steps) per episode during training.

rollout/ep_rew_mean (Episode Reward Mean): The average reward per episode during training. In the context of 
LunarLander, this often reflects the quality of landings; positive values indicate successful landings, 
while negative values indicate poor landings.

time/fps (Frames Per Second): The number of steps (or interactions with the environment) the model completes per second.

time/iterations (Iterations): The number of training iterations, each comprising the collection of experience data 
and model parameter updates.

time/time_elapsed (Time Elapsed): The time elapsed during training, measured in seconds.

time/total_timesteps (Total Timesteps): The cumulative total number of steps taken during training.

train/entropy_loss (Entropy Loss): The entropy loss during training, used to encourage a more uniform action 
probability distribution, thereby enhancing exploration.

train/explained_variance (Explained Variance): A measure of how well the model explains the variance in the observed 
values. Values closer to 1 indicate better model performance.

train/learning_rate (Learning Rate): The learning rate used during training.

train/n_updates (Number of Updates): The number of parameter updates performed during training.

train/policy_loss (Policy Loss): The policy loss during training, reflecting the model's performance in action selection.

train/value_loss (Value Function Loss): The value function loss during training, indicating how well the model estimates state values.

"""