# gymnasium: OpenAI Gym successor — provides the CartPole-v1 simulation environment,
#            including the state space, action space, step logic, and reward signal.
import gymnasium as gym

# torch: Core PyTorch library for tensor operations and autograd (automatic differentiation),
#        which enables gradient-based optimization of neural network parameters.
import torch

# torch.nn: Contains building blocks for neural networks such as Linear layers,
#           activation functions, and the base Module class all networks inherit from.
import torch.nn as nn

# torch.optim: Provides optimization algorithms. We use Adam, which adapts the learning
#              rate per parameter and is well-suited for policy gradient methods.
import torch.optim as optim

# torch.nn.functional: Stateless functions like relu (activation), softmax (probability
#                      output), and mse_loss (mean-squared error for critic training).
import torch.nn.functional as F

# Categorical: A probability distribution over discrete actions. Used to sample actions
#              stochastically during training and to compute log-probabilities and entropy.
from torch.distributions import Categorical

# count: An infinite integer counter from itertools. Used to step through each timestep
#        inside an episode without a fixed upper bound — the loop breaks on 'done'.
from itertools import count

# ---------------------------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------------------------
# Create the CartPole-v1 environment. The goal is to balance a pole on a cart
# by pushing the cart left (0) or right (1). An episode ends when the pole
# falls beyond 12 degrees or the cart moves too far from center, or after
# 500 steps (episode solved = 500 consecutive steps without falling).
env = gym.make('CartPole-v1')

# Reset the environment to get the initial state. seed=42 ensures reproducibility
# of the starting conditions across different runs.
state, info = env.reset(seed=42)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
# GAMMA: Discount factor for future rewards (0 < γ ≤ 1). A value of 0.99 means
#        the agent values a reward 100 steps in the future at ~37% of its face value,
#        encouraging long-horizon planning while keeping returns bounded.
GAMMA = 0.99

# LR: Learning rate for both the actor and critic Adam optimizers. Controls how
#     large a parameter update step is taken after each gradient computation.
LR = 1e-3

# ENTROPY_COEF: Weight on the entropy bonus added to the actor loss. Higher entropy
#               means a more uniform action distribution, encouraging exploration.
#               This prevents the policy from collapsing to a single deterministic action too early.
ENTROPY_COEF = 0.005

# Derive the number of discrete actions (2: push left, push right) and the
# dimensionality of the state vector (4: cart position, cart velocity,
# pole angle, pole angular velocity) directly from the environment.
n_actions = env.action_space.n
n_observations = len(state)

# Select GPU if available for faster tensor operations, otherwise fall back to CPU.
# CartPole is simple enough that CPU training is still fast.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Actor Network
# ---------------------------------------------------------------------------
# The Actor is the "policy network" — it maps an observed state to a probability
# distribution over actions. It answers: "Given what I see, how likely should
# each action be?" During training, actions are sampled stochastically from this
# distribution; during testing, the highest-probability action is chosen (greedy).
class Actor(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(Actor, self).__init__()
        # First hidden layer: projects the 4-dimensional state vector up to 128 features.
        self.fc1 = nn.Linear(n_observations, 128)
        # Second hidden layer: learns higher-level combinations of the 128 features.
        self.fc2 = nn.Linear(128, 128)
        # Output layer: produces one raw logit per action (2 logits for CartPole).
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, x):
        # ReLU activations introduce non-linearity, allowing the network to learn
        # complex, non-linear relationships between state features and action values.
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Softmax converts raw logits into a valid probability distribution (sums to 1)
        # over the action space so they can be sampled or used to compute log-probs.
        return F.softmax(self.fc3(x), dim=-1)


# ---------------------------------------------------------------------------
# Critic Network
# ---------------------------------------------------------------------------
# The Critic is the "value network" — it maps an observed state to a scalar
# estimate of how good that state is (the expected cumulative discounted reward
# from that state onward under the current policy). It answers: "How valuable
# is the situation I am currently in?" This baseline reduces variance in the
# policy gradient update, making training more stable.
class Critic(nn.Module):
    def __init__(self, n_observations):
        super(Critic, self).__init__()
        # Same two-layer hidden architecture as the actor for symmetry,
        # but the output is a single scalar value V(s) rather than action probabilities.
        self.fc1 = nn.Linear(n_observations, 128)
        self.fc2 = nn.Linear(128, 128)
        # Single output neuron: outputs V(s), the estimated state value (no activation —
        # the value can be any real number, not constrained to [0,1]).
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ---------------------------------------------------------------------------
# Instantiate Networks and Optimizers
# ---------------------------------------------------------------------------
# Move both networks to the selected device (CPU or GPU).
actor = Actor(n_observations, n_actions).to(device)
critic = Critic(n_observations).to(device)

# Separate Adam optimizers for the actor and critic allow each network to have
# its own gradient update, preventing the critic's loss from interfering with
# the actor's policy gradient and vice versa.
actor_optimizer = optim.Adam(actor.parameters(), lr=LR)
critic_optimizer = optim.Adam(critic.parameters(), lr=LR)


# ---------------------------------------------------------------------------
# Action Selection
# ---------------------------------------------------------------------------
def select_action(state):
    # Pass the current state through the actor to get action probabilities, e.g. [0.3, 0.7].
    probs = actor(state)

    # Wrap probabilities in a Categorical distribution to enable stochastic sampling.
    # Stochastic action selection is essential during training to explore the state space.
    dist = Categorical(probs)

    # Sample one action from the distribution. Higher-probability actions are more
    # likely to be chosen, but lower-probability ones can still be selected (exploration).
    action = dist.sample()

    # log_prob is ln(π(a|s)): the log-probability of the chosen action under the current policy.
    # Used in the policy gradient loss: ∇J = E[∇log π(a|s) * A(s,a)].
    log_prob = dist.log_prob(action)

    # Entropy H(π) = -Σ π(a) log π(a): measures how spread out the action distribution is.
    # Adding entropy to the loss penalizes a collapsed (nearly deterministic) policy,
    # keeping exploration alive throughout training.
    entropy = dist.entropy()

    return action, log_prob, entropy


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
# We use the Monte Carlo Actor-Critic strategy:
#   1. Run a full episode to collect a trajectory (states, actions, rewards).
#   2. Compute Monte Carlo returns G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
#   3. Update the critic to minimize MSE(V(s), G_t).
#   4. Update the actor via policy gradient weighted by the advantage A_t = G_t - V(s_t).
# Collecting full episodes before updating (instead of updating every step) gives
# more accurate return estimates and leads to stabler learning on CartPole.
num_episodes = 1000

for i_episode in range(num_episodes):
    # Reset the environment at the start of each episode and convert the initial
    # state numpy array to a PyTorch tensor with a batch dimension: shape [1, 4].
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    # Trajectory buffers: store per-timestep data for the entire episode
    # before computing any losses or performing any parameter updates.
    log_probs = []   # log π(a_t | s_t) for each timestep
    values = []      # V(s_t) critic estimates for each timestep
    rewards = []     # r_t environment rewards for each timestep
    entropies = []   # H(π(·|s_t)) entropy values for each timestep

    for t in count():
        # Select an action stochastically and record the log-prob and entropy.
        action, log_prob, entropy = select_action(state)

        # Query the critic for the value of the current state.
        # squeeze(-1) removes the last dimension: [1, 1] → [1] (scalar per batch element).
        value = critic(state).squeeze(-1)

        # Execute the chosen action in the environment.
        # Returns: next_state, reward (+1 for every step the pole stays up),
        #          terminated (pole fell / cart out of bounds), truncated (500-step limit hit).
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        # Accumulate trajectory data for use in the loss computation after the episode ends.
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        entropies.append(entropy)

        # End the episode if terminal or truncated; otherwise advance to the next state.
        if done:
            break

        # Convert the next state to a tensor for the next iteration's forward pass.
        state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

    # -----------------------------------------------------------------------
    # Compute Monte Carlo Returns
    # -----------------------------------------------------------------------
    # G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... is computed by iterating
    # backwards through the reward sequence so each G_t reuses G_{t+1}.
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + GAMMA * G          # discounted cumulative reward from step t onward
        returns.insert(0, G)       # prepend to maintain chronological order

    returns = torch.tensor(returns, dtype=torch.float32, device=device)

    # Normalize returns across the episode to zero mean and unit variance.
    # This reduces the variance of the gradient estimates, preventing large
    # parameter updates from either very long or very short episodes.
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # Concatenate the per-timestep tensors into single batch tensors of shape [T].
    log_probs = torch.cat(log_probs)      # log-probabilities of selected actions
    values = torch.cat(values)            # critic value estimates V(s_t)
    entropies = torch.cat(entropies)      # per-step entropy values

    # -----------------------------------------------------------------------
    # Compute Advantage
    # -----------------------------------------------------------------------
    # Advantage A_t = G_t - V(s_t): measures whether the actual return was
    # better (A > 0) or worse (A < 0) than the critic's baseline prediction.
    # Using the advantage instead of raw returns reduces variance in the
    # policy gradient without introducing bias.
    # .detach() stops gradients from flowing through the advantage into the critic,
    # keeping the actor and critic updates independent.
    advantages = returns - values.detach()

    # -----------------------------------------------------------------------
    # Critic Loss (Value Function Loss)
    # -----------------------------------------------------------------------
    # Minimize the mean squared error between the critic's predictions V(s_t)
    # and the true Monte Carlo returns G_t.  This trains the critic to be a
    # better baseline, which in turn improves advantage estimates for the actor.
    critic_loss = F.mse_loss(values, returns)

    # -----------------------------------------------------------------------
    # Actor Loss (Policy Gradient Loss)
    # -----------------------------------------------------------------------
    # Policy gradient theorem: maximize E[log π(a|s) * A(s,a)].
    # We negate it because PyTorch optimizers minimize by default.
    # The entropy term −H(π) is subtracted (i.e. entropy is maximized) as a
    # regularizer to prevent the policy from becoming deterministic too quickly.
    actor_loss = -(log_probs * advantages).mean() - ENTROPY_COEF * entropies.mean()

    # -----------------------------------------------------------------------
    # Backpropagation and Parameter Updates
    # -----------------------------------------------------------------------
    # Zero out any accumulated gradients from the previous episode before
    # computing fresh gradients, preventing stale gradient accumulation.
    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()

    # Compute gradients via backpropagation through each loss graph.
    actor_loss.backward()
    critic_loss.backward()

    # Gradient clipping caps the norm of the gradient vector at 0.5.
    # This prevents "exploding gradients" — abnormally large updates that
    # can destabilize or permanently damage the learned policy.
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)

    # Apply the clipped gradients to update the network parameters.
    actor_optimizer.step()
    critic_optimizer.step()

    # Log progress every 50 episodes so convergence can be monitored.
    episode_reward = sum(rewards)
    if i_episode % 50 == 0:
        print(f"Episode {i_episode:4d} | Reward: {episode_reward:.1f} | "
              f"Steps: {len(rewards):4d} | "
              f"Actor Loss: {actor_loss.item():.4f} | Critic Loss: {critic_loss.item():.4f}")

env.close()
print("\nTraining complete.")


# ---------------------------------------------------------------------------
# Testing (Greedy Evaluation)
# ---------------------------------------------------------------------------
# After training, evaluate the learned policy deterministically over 1,000
# environment steps.  We count how many times the episode resets (pole fell
# or cart went out of bounds).  The rubric requires fewer than 50 resets —
# a well-trained agent should stay balanced for hundreds of steps at a time.
print("\n--- Testing ---")
resets = 0

# Try human rendering; fall back to headless if no display is available.
try:
    TestEnv = gym.make("CartPole-v1", render_mode="human")
    state, _ = TestEnv.reset()
    render_ok = True
except Exception:
    TestEnv = gym.make("CartPole-v1")
    state, _ = TestEnv.reset()
    render_ok = False

# Switch the actor to evaluation mode: disables dropout/batchnorm randomness
# (not used here, but best practice before any inference pass).
actor.eval()
for step in range(1000):
    # Convert the current state numpy array to a tensor with batch dimension [1, 4].
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    # Greedy action selection: pick the action with the highest predicted probability.
    # torch.no_grad() disables gradient tracking during inference — saves memory
    # and computation since we are not performing any parameter updates here.
    with torch.no_grad():
        probs = actor(state_tensor)
    action = probs.argmax(dim=-1)

    # Step the environment with the greedy action.
    state, _, terminated, truncated, _ = TestEnv.step(action.item())

    # Count resets: each reset means the pole fell or the cart went out of bounds.
    if terminated or truncated:
        resets += 1
        state, _ = TestEnv.reset()

TestEnv.close()
print(f"Total resets in 1000 steps: {resets}")
if resets < 50:
    print("SUCCESS: fewer than 50 resets achieved!")
else:
    print("NOTE: more than 50 resets — consider more training.")