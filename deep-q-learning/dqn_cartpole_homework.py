import gymnasium as gym
import math
import random
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#Initialize the Environment and make an Initial Random Step
env = gym.make('CartPole-v1')
observation, info = env.reset(seed=42)
action = env.action_space.sample() 
observation, reward, terminated, truncated, info = env.step(action)

#Constants
BATCH_SIZE = 128    #Numbers of Samples fed into the nerual network during trainig at once
GAMMA = 0.99        #The decaying factor in the bellman function. Still remember the accumulated *discounted* return?
TAU = 0.005         #Update rate of the duplicate network
LR = 1e-4           #Learning rate of your Q - network
# Epsilon-greedy exploration parameters:
# EPS_START: Initially, the agent explores 90% of the time to discover the environment
# EPS_END: Over time, exploration decays to just 5%, so the agent mostly exploits learned knowledge
# EPS_DECAY: Controls how fast epsilon decays — after ~1000 steps, epsilon is near its minimum
# The decay formula is: eps = EPS_END + (EPS_START - EPS_END) * e^(-steps_done / EPS_DECAY)
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        # A 3-layer fully connected (linear) neural network:
        # Input layer: takes the state observation vector (size 4 for CartPole: cart position,
        #   cart velocity, pole angle, pole angular velocity) and maps it to 128 hidden units
        self.layer1 = nn.Linear(n_observations, 128)
        # Hidden layer: 128 -> 128 units, adds more capacity for learning complex Q-value patterns
        self.layer2 = nn.Linear(128, 128)
        # Output layer: 128 -> n_actions (2 for CartPole: push left or push right)
        # Each output neuron represents the estimated Q-value for that action
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        # Pass input through layer1, then apply ReLU activation (sets negative values to 0)
        x = F.relu(self.layer1(x))
        # Pass through layer2 with ReLU activation
        x = F.relu(self.layer2(x))
        # Output layer has NO activation — Q-values can be any real number (positive or negative)
        return self.layer3(x)
    
#Creating to instances of the Q-network.
#Policy net is trained online directly by loss function
#Target network updates slower and provides a more stable target
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

def select_action(state):
    global steps_done
    # Generate a random number between 0 and 1
    sample = random.random()
    # Calculate the current epsilon threshold using exponential decay:
    # Early in training, eps_threshold is high (~0.9) so the agent explores often
    # Later, eps_threshold shrinks toward EPS_END (~0.05) so the agent mostly exploits
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        # EXPLOIT: Pick the action with the highest Q-value from the policy network
        # torch.no_grad() disables gradient computation since we're only doing inference here
        # .max(1) finds the max Q-value across actions, .indices gives the action index
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        # EXPLORE: Pick a random action from the action space to discover new strategies
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []
steps_done = 0


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                        batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    # Compute Q_target using the Bellman equation: Q_target = reward + gamma * max_a'(Q(s', a'))
    # Initialize all next state values to 0 (terminal states will stay 0)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        # For non-terminal states, use the target network to estimate max Q-value of the next state
        # target_net provides stable Q-value estimates (it updates slowly via soft update with TAU)
        # .max(1).values returns the maximum Q-value across all actions for each next state
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Bellman equation: expected Q = immediate reward + discounted future Q-value
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Smooth L1 Loss (Huber Loss): combines the best of L1 and L2 loss
    # - Behaves like L2 (MSE) for small errors, giving smooth gradients near zero
    # - Behaves like L1 (MAE) for large errors, making it robust to outliers
    # This is preferred over plain MSE because replay memory can produce large Q-value errors
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# The main training loop: run episodes of the environment, collect experience, and optimize the model
# We run more episodes if we have a GPU available since training will be faster, allowing the agent to learn better policies
# If running on CPU, we limit the number of episodes to keep training time reasonable while still allowing the agent to learn a decent policy
# Note: The number of episodes is a hyperparameter that can be tuned for better performance. More episodes generally lead to better learning but take more time.
if torch.cuda.is_available():
    num_episodes = 500
else:
    num_episodes = 300

for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    total_reward = 0
    num_steps = 0
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)
        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            break
    if i_episode % 10 == 0:
        print(f"Training episode {i_episode}/{num_episodes} | Last episode duration: {episode_durations[-1]} steps")

print("Training complete!\n")
env.close()

TestEnv = gym.make("CartPole-v1", render_mode="human")
observation, info = TestEnv.reset(seed=42)

end_count = 0
for step in range(1000):
    state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    # During testing, use a purely greedy policy (no exploration):
    # Always pick the action with the highest Q-value from the trained policy network
    # This gives the best performance since the agent no longer needs to explore
    with torch.no_grad():
        action = policy_net(state).max(1).indices.view(1, 1)
    observation, reward, terminated, truncated, _ = TestEnv.step(action.item())

    if terminated or truncated:
        end_count += 1
        print(f"Reset #{end_count} at step {step + 1}")
        observation, info = TestEnv.reset()

TestEnv.close()
print(f"\nTotal resets in 1000 steps: {end_count}")