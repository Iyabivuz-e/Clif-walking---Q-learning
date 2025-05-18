import numpy as np  # for matrix operations
import gym          # to use OpenAI Gym environments

# Create the CliffWalking environment
env = gym.make("CliffWalking-v0")

# Get number of states and actions
num_states = env.observation_space.n
num_actions = env.action_space.n


def q_learning(env, num_episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
    Q = np.zeros((num_states, num_actions))  # initialize Q-table to zero

    for episode in range(num_episodes):
        state = env.reset()  # reset the environment at start of episode
        done = False

        while not done:
            # ε-greedy policy: sometimes explore, sometimes exploit
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # explore: random action
            else:
                action = np.argmax(Q[state])  # exploit: best known action

            next_state, reward, done, _, = env.step(
                action)  # take action and observe result

            # Q-learning update: off-policy method
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + gamma * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta  # update rule

            state = next_state  # move to next state

    return Q


def sarsa(env, num_episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
    Q = np.zeros((num_states, num_actions))

    for episode in range(num_episodes):
        state = env.reset()
        # Choose initial action using ε-greedy
        action = np.random.choice(num_actions) if np.random.rand(
        ) < epsilon else np.argmax(Q[state])
        done = False

        while not done:
            next_state, reward, done, _, = env.step(action)

            # Choose next action using ε-greedy again (on-policy)
            next_action = np.random.choice(num_actions) if np.random.rand(
            ) < epsilon else np.argmax(Q[next_state])

            # SARSA update: on-policy method
            td_target = reward + gamma * Q[next_state][next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            state, action = next_state, next_action  # update both state and action

    return Q


def monte_carlo(env, num_episodes=1000, gamma=0.9, epsilon=0.1):
    Q = np.zeros((num_states, num_actions))
    returns_sum = np.zeros((num_states, num_actions))
    returns_count = np.zeros((num_states, num_actions))

    for episode in range(num_episodes):
        state = env.reset()
        episode_data = []

        done = False
        while not done:
            action = np.random.choice(num_actions) if np.random.rand(
            ) < epsilon else np.argmax(Q[state])
            next_state, reward, done, _, = env.step(action)
            episode_data.append((state, action, reward))
            state = next_state

        # Calculate returns and update Q
        G = 0
        visited = set()

        for t in reversed(range(len(episode_data))):
            s, a, r = episode_data[t]
            G = gamma * G + r

            if (s, a) not in visited:  # first-visit check
                visited.add((s, a))
                returns_sum[s][a] += G
                returns_count[s][a] += 1
                Q[s][a] = returns_sum[s][a] / \
                    returns_count[s][a]  # average return

    return Q


def print_cliffwalking_grid(agent_position):
    rows, cols = 4, 12
    grid = [["0"] * cols for _ in range(rows)]

    # Coordinates
    start = (3, 0)
    goal = (3, 11)
    cliff = [(3, i) for i in range(1, 11)]

    grid[start[0]][start[1]] = 'S'
    grid[goal[0]][goal[1]] = 'G'
    for r, c in cliff:
        grid[r][c] = '*'

    # Mark agent
    r, c = divmod(agent_position, cols)
    if grid[r][c] not in ('S', 'G', '*'):
        grid[r][c] = 'A'

    for row in grid:
        print(" ".join(row))
    print()


def evaluate_policy(env, Q):
    state = env.reset()
    done = False

    while not done:
        print_cliffwalking_grid(state)
        action = np.argmax(Q[state])  # always pick best action
        state, _, done, _, = env.step(action)


q_table = q_learning(env)
evaluate_policy(env, q_table)
