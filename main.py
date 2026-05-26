import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Create CartPole environment and wrap it in DummyVecEnv for compatibility
env = DummyVecEnv([lambda: gym.make('CartPole-v1')])

# Training parameters
total_timesteps = 10000

# DQN Training
start_time_dqn = time.time()
dqn_model = DQN('MlpPolicy', env, verbose=0)
dqn_model.learn(total_timesteps=total_timesteps)
dqn_training_time = time.time() - start_time_dqn

# PPO Training
start_time_ppo = time.time()
ppo_model = PPO('MlpPolicy', env, verbose=0)
ppo_model.learn(total_timesteps=total_timesteps)
ppo_training_time = time.time() - start_time_ppo

# Evaluation function compatible with VecEnv
def evaluate_model(model, env, num_episodes=100):
    episode_rewards = []
    for episode in range(num_episodes):
        done = False
        obs = env.reset()
        total_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            done = done.any()  # To handle VecEnv
        episode_rewards.append(total_reward)
    return episode_rewards

# Get returns for both DQN and PPO
dqn_rewards = evaluate_model(dqn_model, env)
ppo_rewards = evaluate_model(ppo_model, env)

# Calculate average return for each episode
dqn_avg_returns = [np.mean(dqn_rewards[:i+1]) for i in range(len(dqn_rewards))]
ppo_avg_returns = [np.mean(ppo_rewards[:i+1]) for i in range(len(ppo_rewards))]


# Calculate metrics
def calculate_metrics(rewards):
    avg_return = np.mean(rewards)
    cum_return = np.sum(rewards)
    std_dev_return = np.std(rewards)
    return avg_return, cum_return, std_dev_return

dqn_avg_return, dqn_cum_return, dqn_std_dev_return = calculate_metrics(dqn_rewards)
ppo_avg_return, ppo_cum_return, ppo_std_dev_return = calculate_metrics(ppo_rewards)

# Printing metrics
print("DQN - Avg Return:", dqn_avg_return, "Cumulative Return:", dqn_cum_return, "Std Dev:", dqn_std_dev_return, "Training Time:", dqn_training_time)
print("PPO - Avg Return:", ppo_avg_return, "Cumulative Return:", ppo_cum_return, "Std Dev:", ppo_std_dev_return, "Training Time:", ppo_training_time)

# Create a DataFrame to store the results
results_df = pd.DataFrame({
    'Episode': range(1, 101),
    'DQN Average Return': dqn_avg_returns,
    'PPO Average Return': ppo_avg_returns
})

# Print the DataFrame
print(results_df)

# Save the DataFrame to a CSV file
results_df.to_csv('average_returns_dqn_ppo.csv', index=False)

# Graphs
episodes = range(1, 101)

# Plotting average return
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(episodes, dqn_rewards, label="DQN")
plt.plot(episodes, ppo_rewards, label="PPO")
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Average Return')
plt.legend()

# Plotting cumulative return
plt.subplot(2, 2, 2)
plt.bar(['DQN', 'PPO'], [dqn_cum_return, ppo_cum_return])
plt.ylabel('Cumulative Return')
plt.title('Cumulative Return')

# Plotting standard deviation of return
plt.subplot(2, 2, 3)
plt.bar(['DQN', 'PPO'], [dqn_std_dev_return, ppo_std_dev_return])
plt.ylabel('Standard Deviation of Return')
plt.title('Standard Deviation of Return')

# Plotting convergence time
plt.subplot(2, 2, 4)
plt.bar(['DQN', 'PPO'], [dqn_training_time, ppo_training_time])
plt.ylabel('Convergence Time (seconds)')
plt.title('Convergence Time')

plt.tight_layout()
plt.show()

