import matplotlib.pyplot as plt
import numpy as np
import gym
import time
from IPython.display import clear_output

# Set random seed for reproducibility
# np.random.seed(42)

# We are creating the Cliff Walking environment
env = gym.make("CliffWalking-v0")

# Value initialization
num_states = env.observation_space.n  # Number of states in the environment
num_actions = env.action_space.n  # Number of possible actions in the environment
print("Number of states:", num_states)
print("Number of actions:", num_actions)

# Here we then define the action meanings for better visualization
action_meanings = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

# We then create a helper function to convert state (single integer) to row, col coordinates


def state_to_coords(state):
    # considering the CliffWalking grid which is 4x12
    nrow, ncol = 4, 12
    row = state // ncol
    col = state % ncol
    return row, col


# *************We create a function to visualize the cliff walking environment*********
def print_cliffwalking_grid(agent_position, path_history=None, q_values_for_current_state=None):
    nrow, ncol = 4, 12
    grid = np.full((nrow, ncol), '0', dtype=str)

    # Mark the cliff (Danger zone)
    grid[3, 1:11] = '*'

    # Mark start (S), and the goal (G)
    grid[3, 0] = 'S'
    grid[3, 11] = 'G'

    # Mark path history if provided (enhancement, good for understanding)
    if path_history:
        for pos_state in path_history:
            row, col = state_to_coords(pos_state)
            # Avoid overwriting S, G, or current A with path marker
            if grid[row, col] not in ['S', 'G']:
                grid[row, col] = '·'

    # Mark agent position which is an agent_position and is integer state
    agent_row, agent_col = state_to_coords(agent_position)
    grid[agent_row, agent_col] = 'A'

    # Displaying the grid
    print("\nEnvironment Grid:")
    for r in range(nrow):
        print(' '.join(grid[r]))

    # Displaying Q-values for current state
    if q_values_for_current_state is not None:
        print("\nQ-values for current state (" +
              str(state_to_coords(agent_position)) + "):")
        for action_idx in range(num_actions):
            print(
                f"{action_meanings[action_idx]}: {q_values_for_current_state[action_idx]:.4f}")
    print("\n")


# ***********We then create a function to plot the learning curve ***************
def plot_learning_curve(rewards_history_list, title):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history_list)
    plt.title(f"{title} Learning Curve")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.grid(True)

    window_size = min(50, len(rewards_history_list) //
                      10 if len(rewards_history_list) > 0 else 0)
    if window_size > 0:
        rolling_mean = np.convolve(rewards_history_list, np.ones(
            window_size)/window_size, mode='valid')
        plt.plot(np.arange(window_size-1, len(rewards_history_list)), rolling_mean, 'r-',  # Use np.arange for x-axis
                 linewidth=2, label=f'Rolling Average ({window_size} episodes)')
        plt.legend()
    plt.show()


# ****** We then create a function to log training information - better for understanding what is going on **********
def log_training_info(episode, current_reward, current_state, chosen_action, next_obs, q_table_for_state, current_epsilon):
    print(f"\nEpisode {episode + 1}:")
    print(f"State: {current_state} (Coords: {state_to_coords(current_state)})")
    print(f"Action: {chosen_action} ({action_meanings[chosen_action]})")
    print(f"Next State: {next_obs} (Coords: {state_to_coords(next_obs)})")
    print(f"Reward: {current_reward}")
    print(f"Epsilon: {current_epsilon:.4f}")
    if current_state < q_table_for_state.shape[0]:
        print("\nQ-values for current state:")
        for a_idx in range(num_actions):
            print(
                f"{action_meanings[a_idx]}: {q_table_for_state[current_state][a_idx]:.4f}")


# ******We then create a Q-learning algorithm for our project *********
def q_learning(episodes=5000, alpha=0.1, gamma=0.9, epsilon=0.1):
    Q = np.zeros((num_states, num_actions))
    rewards_history = []
    optimal_paths_found = []  # This one is used to track shortest paths found
    episode_lengths = []

    for episode_idx in range(episodes):
        state, info = env.reset()
        done = False
        total_episode_reward = 0
        current_path = [state]
        num_steps = 0

        # Here we log the periodic detailed training - for seeing what is going on and the path the agent is taking
        log_detailed_this_episode = (episode_idx + 1) % 100 == 0

        while not done:
            num_steps += 1

            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # This is the agent exploring the environment
            else:
                # This is the agent exploiting the environment
                action = np.argmax(Q[state])

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_episode_reward += reward
            current_path.append(next_state)

            # Q-value of best action in next state
            best_next_action_q_value = np.max(Q[next_state])
            Q[state, action] = Q[state, action] + alpha * \
                (reward + gamma * best_next_action_q_value - Q[state, action])

            if log_detailed_this_episode and num_steps <= 5:  # Log first few steps of milestone episodes
                print(
                    f"--- Training: Episode {episode_idx + 1}, Step {num_steps} ---")
                log_training_info(episode_idx, reward, state,
                                  action, next_state, Q, epsilon)
                print_cliffwalking_grid(
                    next_state, current_path, Q[next_state])
                if 'ipykernel' in __import__('sys').modules:
                    time.sleep(0.1)  # Shorter sleep for training log

            state = next_state

        rewards_history.append(total_episode_reward)
        episode_lengths.append(num_steps)

        # We then track the optimal paths - shortest ones found so far
        if not optimal_paths_found or len(current_path) < len(optimal_paths_found[0]):
            optimal_paths_found = [current_path]
        elif len(current_path) == len(optimal_paths_found[0]) and current_path not in optimal_paths_found:
            optimal_paths_found.append(current_path)

        if (episode_idx + 1) % 100 == 0:
            print(
                f"\nEpisode: {episode_idx+1}/{episodes}, Steps: {num_steps}, "
                f"Reward: {total_episode_reward:.2f}, Epsilon: {epsilon:.4f}")
            if optimal_paths_found:
                print(
                    f"Found {len(optimal_paths_found)} optimal paths of length {len(optimal_paths_found[0])}")
            # Print final state of milestone training episodes
            print_cliffwalking_grid(state, current_path, Q[state])

    plot_learning_curve(rewards_history, "Q-learning Rewards")
    # Plotting episode lengths
    plot_learning_curve(episode_lengths, "Episode Lengths Over Time")

    print(
        f"\nTraining complete. Found {len(optimal_paths_found)} optimal paths.")
    return Q, optimal_paths_found

# *************We then evaluate the trained agent***************


def evaluate_policy(q_table, optimal_paths_from_training, n_eval_episodes=10):
    print("\n--- Evaluating learned policy ---")

    if optimal_paths_from_training:
        print(
            f"Found {len(optimal_paths_from_training)} optimal paths during training (showing up to 3):")
        for i, path_val in enumerate(optimal_paths_from_training[:3]):
            print(f"\nOptimal Path #{i+1} (Length: {len(path_val)}):")
            state_sequence_str = [str(state_to_coords(s)) for s in path_val]
            print(" -> ".join(state_sequence_str))
    else:
        print("No optimal paths were recorded during training.")

    for episode_num in range(n_eval_episodes):
        state, info = env.reset()
        done = False
        total_eval_reward = 0
        eval_steps = 0
        eval_path = [state]

        print(
            f"\n--- Evaluation Episode {episode_num + 1}/{n_eval_episodes} ---")

        # Show initial state with Q-values
        print_cliffwalking_grid(state, None, q_table[state])
        if 'ipykernel' in __import__('sys').modules:
            time.sleep(1)

        while not done:
            eval_steps += 1
            # Greedy action to decide which direction to take next
            action_to_take = np.argmax(q_table[state])

            next_state, reward, terminated, truncated, info = env.step(
                action_to_take)
            done = terminated or truncated

            total_eval_reward += reward
            eval_path.append(next_state)
            state = next_state

            if 'ipykernel' in __import__('sys').modules:
                clear_output(wait=True)
            else:
                print("\033c", end="")  # ANSI clear screen

            print(f"Evaluation Episode {episode_num + 1}, Step {eval_steps}")
            print(
                f"Action: {action_meanings[action_to_take]}, Reward: {reward}")
            # We then print the current state of the grid using print_cliffwalking_grid.
            print_cliffwalking_grid(state, eval_path, q_table[state])
            if 'ipykernel' in __import__('sys').modules:
                time.sleep(0.5)

        print(f"Evaluation Episode {episode_num + 1} finished.")
        print(
            f"Total Reward: {total_eval_reward}, Path Length: {len(eval_path)}")
        is_optimal_this_run = False
        if optimal_paths_from_training:
            for opt_path in optimal_paths_from_training:
                if eval_path == opt_path:
                    is_optimal_this_run = True
                    break
        print(
            f"Path matches a training optimal path: {'Yes' if is_optimal_this_run else 'No'}")

    env.close()


# *************Run the training and evaluation**********
if __name__ == "__main__":
    print("Starting Q-learning agent training...")

    learned_Q_table, found_optimal_paths = q_learning(episodes=5000,
                                                      alpha=0.1,
                                                      gamma=0.9,
                                                      epsilon=0.1)

    evaluate_policy(learned_Q_table, found_optimal_paths, n_eval_episodes=5)
