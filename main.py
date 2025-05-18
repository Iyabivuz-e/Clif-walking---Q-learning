# import gym
# import numpy as np

# # Create the CliffWalking environment
# env = gym.make("CliffWalking-v0")

# # Get number of states and actions
# n_states = env.observation_space.n
# n_actions = env.action_space.n

# # Create Q-table filled with zeros
# q_table = np.zeros((n_states, n_actions))

# # Hyperparameters
# alpha = 0.1      # Learning rate
# gamma = 0.9      # Discount factor
# epsilon = 0.1    # Exploration rate
# episodes = 5000  # Total training episodes

# # Training loop
# for episode in range(episodes):
#     state = env.reset()[0]  # Get initial state
#     done = False            # To track end of episode

#     while not done:
#         # Choose action: explore or exploit
#         if np.random.uniform(0, 1) < epsilon:
#             action = env.action_space.sample()  # Explore
#         else:
#             action = np.argmax(q_table[state])  # Exploit best action

#         # Take the action and observe the result
#         next_state, reward, done, _, _ = env.step(action)

#         # Q-learning update rule
#         best_next_action = np.max(q_table[next_state])
#         q_table[state, action] = q_table[state, action] + alpha * \
#             (reward + gamma * best_next_action - q_table[state, action])

#         # Move to next state
#         state = next_state

# # Test the learned policy
# state = env.reset()[0]
# done = False

# print("\nLearned path:")

# while not done:
#     env.render()  # Show the grid
#     action = np.argmax(q_table[state])  # Choose best action
#     state, _, done, _, _ = env.step(action)

# env.close()


# import matplotlib.pyplot as plt
# import numpy as np
# import gym
# import time
# from IPython.display import clear_output

# # Create the Cliff Walking environment
# env = gym.make("CliffWalking-v0")

# # Environment details
# print("Number of states:", env.observation_space.n)
# print("Number of actions:", env.action_space.n)


# def visualize_cliff_walking(agent_position=None):
#     """
#     Visualize the Cliff Walking grid with agent's current position.
    
#     Parameters:
#     - agent_position: Integer representing agent's current position (0-47)
    
#     The grid layout:
#     S: Start
#     G: Goal
#     *: Cliff
#     0: Safe path
#     A: Agent's current position (if provided)
#     """
#     rows, cols = 4, 12
#     grid = [["0"] * cols for _ in range(rows)]

#     # Define special positions
#     start = (3, 0)
#     goal = (3, 11)
#     cliff = [(3, i) for i in range(1, 11)]

#     # Mark special positions
#     grid[start[0]][start[1]] = 'S'
#     grid[goal[0]][goal[1]] = 'G'
#     for r, c in cliff:
#         grid[r][c] = '*'

#     # Mark agent if provided
#     if agent_position is not None:
#         r, c = divmod(agent_position, cols)
#         if grid[r][c] not in ('S', 'G', '*'):
#             grid[r][c] = 'A'

#     # Print the grid
#     for row in grid:
#         print(" ".join(row))
#     print()


# def plot_learning_curve(rewards_history, algorithm_name, window_size=100):
#     """
#     Plot the learning curve showing rewards over episodes.
    
#     Parameters:
#     - rewards_history: List of rewards obtained per episode
#     - algorithm_name: Name of the algorithm for title
#     - window_size: Size of moving average window
#     """
#     # Calculate moving average
#     moving_avg = np.convolve(rewards_history, np.ones(
#         window_size)/window_size, mode='valid')

#     plt.figure(figsize=(10, 5))
#     plt.plot(rewards_history, alpha=0.3, label='Raw rewards')
#     plt.plot(range(window_size-1, len(rewards_history)),
#              moving_avg, label=f'{window_size}-episode moving avg')
#     plt.title(f'{algorithm_name} Learning Curve')
#     plt.xlabel('Episode')
#     plt.ylabel('Total Reward')
#     plt.legend()
#     plt.grid()
#     plt.show()

# # Modified Q-learning function to track rewards


# def q_learning(episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
#     Q = np.zeros((env.observation_space.n, env.action_space.n))
#     rewards_history = []

#     for episode in range(episodes):
#         state = env.reset()
#         done = False
#         total_reward = 0

#         while not done:
#             if np.random.uniform(0, 1) < epsilon:
#                 action = env.action_space.sample()
#             else:
#                 action = np.argmax(Q[state])

#             next_state, reward, done, info = env.step(action)
#             total_reward += reward

#             best_next_action = np.argmax(Q[next_state])
#             Q[state, action] += alpha * \
#                 (reward + gamma * Q[next_state,
#                  best_next_action] - Q[state, action])

#             state = next_state

#         rewards_history.append(total_reward)

#         if episode % 100 == 0:
#             print(f"Episode: {episode}, Reward: {total_reward}")
#             visualize_cliff_walking(state)

#     plot_learning_curve(rewards_history, "Q-learning")
#     return Q



# def sarsa(episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
#     """
#     SARSA algorithm implementation for Cliff Walking.
    
#     Parameters same as Q-learning.
    
#     SARSA is an on-policy algorithm that learns the Q-values while following 
#     an epsilon-greedy policy. It updates Q-values based on the actual action taken.
#     The name comes from the sequence State-Action-Reward-State-Action.
#     """

#     Q = np.zeros((env.observation_space.n, env.action_space.n))

#     for episode in range(episodes):
#         state = env.reset()

#         # Choose initial action using epsilon-greedy
#         if np.random.uniform(0, 1) < epsilon:
#             action = env.action_space.sample()
#         else:
#             action = np.argmax(Q[state])

#         done = False
#         total_reward = 0

#         while not done:
#             # Take action and observe outcome
#             next_state, reward, done, info = env.step(action)
#             total_reward += reward

#             # Choose next action using epsilon-greedy
#             if np.random.uniform(0, 1) < epsilon:
#                 next_action = env.action_space.sample()
#             else:
#                 next_action = np.argmax(Q[next_state])

#             # SARSA update rule
#             Q[state, action] = Q[state, action] + alpha * (
#                 reward + gamma * Q[next_state, next_action] - Q[state, action]
#             )

#             state = next_state
#             action = next_action

#         if episode % 100 == 0:
#             print(f"Episode: {episode}, Total Reward: {total_reward}")

#     print("Training completed!")
#     plot_learning_curve(rewards_history, "SARSA")
#     return Q


# def monte_carlo(episodes=500, gamma=0.9, epsilon=0.1):
#     """
#     Monte Carlo algorithm implementation for Cliff Walking.
    
#     Parameters:
#     - episodes: Number of training episodes
#     - gamma: Discount factor
#     - epsilon: Exploration rate
    
#     Monte Carlo methods learn from complete episodes. They wait until the end
#     of the episode to update Q-values based on the actual returns.
#     """

#     Q = np.zeros((env.observation_space.n, env.action_space.n))
#     # To keep track of how many times we've visited each state-action pair
#     N = np.zeros((env.observation_space.n, env.action_space.n))

#     for episode in range(episodes):
#         state = env.reset()
#         done = False
#         episode_history = []
#         total_reward = 0

#         # Generate an episode
#         while not done:
#             # Epsilon-greedy action selection
#             if np.random.uniform(0, 1) < epsilon:
#                 action = env.action_space.sample()
#             else:
#                 action = np.argmax(Q[state])

#             next_state, reward, done, info = env.step(action)
#             total_reward += reward

#             # Store the state, action, reward
#             episode_history.append((state, action, reward))
#             state = next_state

#         # Now process the episode backwards to calculate returns
#         G = 0  # Return
#         visited_state_actions = set()

#         for t in reversed(range(len(episode_history))):
#             state, action, reward = episode_history[t]
#             G = gamma * G + reward

#             # Only update the first visit to each state-action pair (First-visit MC)
#             if (state, action) not in visited_state_actions:
#                 visited_state_actions.add((state, action))
#                 N[state, action] += 1
#                 Q[state, action] += (G - Q[state, action]) / N[state, action]

#         if episode % 100 == 0:
#             print(f"Episode: {episode}, Total Reward: {total_reward}")

#     print("Training completed!")
#     return Q


# def evaluate_agent(Q, render=False):
#     """
#     Evaluate the trained agent by running it through the environment.
    
#     Parameters:
#     - Q: The learned Q-table
#     - render: Whether to visualize the agent's movement
#     """
#     state = env.reset()
#     done = False
#     total_reward = 0

#     while not done:
#         if render:
#             clear_output(wait=True)
#             # Convert state number to grid position (row, col)
#             row = 0 if state < 12 else 1
#             col = state % 12
#             visualize_cliff_walking((row, col))
#             time.sleep(0.3)

#         # Always choose the best action according to Q-table
#         action = np.argmax(Q[state])
#         state, reward, done, info = env.step(action)
#         total_reward += reward

#     print(f"Total reward during evaluation: {total_reward}")
#     env.close()


# # Train with Q-learning
# print("Training with Q-learning...")
# Q_qlearning = q_learning(episodes=1000)
# print("\nEvaluating Q-learning agent...")
# evaluate_agent(Q_qlearning, render=True)

# # Train with SARSA
# print("\nTraining with SARSA...")
# Q_sarsa = sarsa(episodes=1000)
# print("\nEvaluating SARSA agent...")
# evaluate_agent(Q_sarsa, render=True)

# # Train with Monte Carlo
# print("\nTraining with Monte Carlo...")
# Q_mc = monte_carlo(episodes=1000)
# print("\nEvaluating Monte Carlo agent...")
# evaluate_agent(Q_mc, render=True)


# Creating the corrected and improved version of the user's Cliff Walking project code as a .py file

# corrected_code = 
import matplotlib.pyplot as plt
import numpy as np
import gym
import time
from IPython.display import clear_output

# Set random seed for reproducibility
# np.random.seed(42)

# Create the Cliff Walking environment
env = gym.make("CliffWalking-v0")
env.reset()

# Value initialization

num_states = env.observation_space.n ## Number of states in the environment
num_actions = env.action_space.n ## Number of possible actions in the environment
print("Number of states:", num_states)
print("Number of actions:", num_actions)


# Q-learning Parameters:
# alpha = 0.1: Learning rate for the Q-learning algorithm.
# gamma = 0.9: Discount factor for future rewards in the Q-learning algorithm.
# epsilon = 0.1: Exploration-exploitation trade-off parameter for the epsilon-greedy policy.

def q_learning(episodes=100, alpha=0.1, gamma=0.9, epsilon=0.1):
    Q = np.zeros((num_states, num_actions)) ## Initializes a Q-table with zeros for each state-action pair
    rewards_history = []
    for episode in range(episodes): ## Evaluation loop
        state = env.reset() ## Resets the environment
        done = False
        total_reward = 0
        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            next_state, reward, done, _, = env.step(action)
            total_reward += reward
            best_next_action = np.argmax(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * Q[next_state, best_next_action] - Q[state, action])
            state = next_state
        rewards_history.append(total_reward)
        if episode % 100 == 0:
            print(f"Episode: {episode}, Reward: {total_reward}")
            visualize_cliff_walking(state)
    plot_learning_curve(rewards_history, "Q-learning")
    return Q

# def sarsa(episodes=100, alpha=0.1, gamma=0.9, epsilon=0.1):
#     Q = np.zeros((env.observation_space.n, env.action_space.n))
#     rewards_history = []
#     for episode in range(episodes):
#         state = env.reset()
#         action = env.action_space.sample() if np.random.uniform(0, 1) < epsilon else np.argmax(Q[state])
#         done = False
#         total_reward = 0
#         while not done:
#             next_state, reward, done, _, = env.step(action)
#             total_reward += reward
#             next_action = env.action_space.sample() if np.random.uniform(0, 1) < epsilon else np.argmax(Q[next_state])
#             Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
#             state, action = next_state, next_action
#         rewards_history.append(total_reward)
#         if episode % 100 == 0:
#             print(f"Episode: {episode}, Total Reward: {total_reward}")
#     print("Training completed!")
#     plot_learning_curve(rewards_history, "SARSA")
#     return Q

# def monte_carlo(episodes=500, gamma=0.9, epsilon=0.1):
#     Q = np.zeros((env.observation_space.n, env.action_space.n))
#     N = np.zeros((env.observation_space.n, env.action_space.n))
#     rewards_history = []
#     for episode in range(episodes):
#         state = env.reset()
#         done = False
#         episode_history = []
#         total_reward = 0
#         while not done:
#             action = env.action_space.sample() if np.random.uniform(0, 1) < epsilon else np.argmax(Q[state])
#             next_state, reward, done, _, = env.step(action)
#             total_reward += reward
#             episode_history.append((state, action, reward))
#             state = next_state
#         G = 0
#         visited_state_actions = set()
#         for t in reversed(range(len(episode_history))):
#             state, action, reward = episode_history[t]
#             G = gamma * G + reward
#             if (state, action) not in visited_state_actions:
#                 visited_state_actions.add((state, action))
#                 N[state, action] += 1
#                 Q[state, action] += (G - Q[state, action]) / N[state, action]
#         rewards_history.append(total_reward)
#         if episode % 100 == 0:
#             print(f"Episode: {episode}, Total Reward: {total_reward}")
#     print("Training completed!")
#     plot_learning_curve(rewards_history, "Monte Carlo")
#     return Q

# def evaluate_agent(Q, render=False):
#     state =env.reset()
#     done = False
#     total_reward = 0
#     while not done:
#         if render:
#             clear_output(wait=True)
#             visualize_cliff_walking(state)
#             time.sleep(0.3)
#         action = np.argmax(Q[state])
#         state, reward, done, _, = env.step(action)
#         total_reward += reward
#     print(f"Total reward during evaluation: {total_reward}")
#     env.close()
    
# def visualize_cliff_walking(agent_position=None):
#     rows, cols = 4, 12
#     grid = [["0"] * cols for _ in range(rows)]
#     start = (3, 0)
#     goal = (3, 11)
#     cliff = [(3, i) for i in range(1, 11)]
#     grid[start[0]][start[1]] = 'S'
#     grid[goal[0]][goal[1]] = 'G'
#     for r, c in cliff:
#         grid[r][c] = '*'
#     if agent_position is not None:
#         r, c = divmod(agent_position, cols)
#         if grid[r][c] not in ('S', 'G', '*'):
#             grid[r][c] = 'A'
#     for row in grid:
#         print(" ".join(row))
#     print()


# def plot_learning_curve(rewards_history, algorithm_name, window_size=100):
#     moving_avg = np.convolve(rewards_history, np.ones(
#         window_size)/window_size, mode='valid')
#     plt.figure(figsize=(10, 5))
#     plt.plot(rewards_history, alpha=0.3, label='Raw rewards')
#     plt.plot(range(window_size-1, len(rewards_history)),
#              moving_avg, label=f'{window_size}-episode moving avg')
#     plt.title(f'{algorithm_name} Learning Curve')
#     plt.xlabel('Episode')
#     plt.ylabel('Total Reward')
#     plt.legend()
#     plt.grid()
#     plt.show()

# print("Training with Q-learning...")
# Q_qlearning = q_learning(episodes=15000)
# print("\\nEvaluating Q-learning agent...")
# evaluate_agent(Q_qlearning, render=True)

# print("\\nTraining with SARSA...")
# Q_sarsa = sarsa(episodes=1000)
# print("\\nEvaluating SARSA agent...")
# evaluate_agent(Q_sarsa, render=True)

# print("\\nTraining with Monte Carlo...")
# Q_mc = monte_carlo(episodes=1000)
# print("\\nEvaluating Monte Carlo agent...")
# evaluate_agent(Q_mc, render=True)


"""

# Save to file
file_path = "/mnt/data/cliff_walking_rl_fixed.py"
with open(file_path, "w") as f:
    f.write(corrected_code)

file_path
"""