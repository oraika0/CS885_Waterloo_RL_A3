import gym
import numpy as np
import utils_rnn.envs, utils_rnn.seed, utils_rnn.buffers, utils_rnn.torch
import torch
import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Deep Recurrent Q Learning
# Slide 17
# cs.uwaterloo.ca/~ppoupart/teaching/cs885-winter22/slides/cs885-module4.pdf

# Constants
SEEDS = [1, 2, 3, 4, 5]
# SEEDS = [1]
t = utils_rnn.torch.TorchHelper()
DEVICE = t.device
OBS_N = 2               # State space size
ACT_N = 2               # Action space size
STARTING_EPSILON = 1.0  # Starting epsilon
STEPS_MAX = 10000       # Gradually reduce epsilon over these many steps
EPSILON_END = 0.1       # At the end, keep epsilon at this value
MINIBATCH_SIZE = 15     # How many examples to sample per train step
GAMMA = 0.95           # Discount factor in episodic reward objective
LEARNING_RATE = 5e-4    # Learning rate for Adam optimizer
# TRAIN_AFTER_EPISODES = 10   # Just collect episodes for these many episodes
TRAIN_AFTER_EPISODES = 100   # Just collect episodes for these many episodes
TRAIN_EPOCHS = 6        # Train for these many epochs every time
BUFSIZE = 600         # Replay buffer size
EPISODES = 2000         # Total number of episodes to learn over
TEST_EPISODES = 10     # Test episodes
# HIDDEN = 128            # Hidden nodes
TARGET_NETWORK_UPDATE_FREQ = 10 # Target network update frequency

# Global variables
EPSILON = STARTING_EPSILON
Q = None

# Deep recurrent Q network
class DRQN(torch.nn.Module):
    
    def __init__(self, obs_n=2, act_n=2, hidden_size=512, num_layers=1):
        super().__init__()
        ## TODO: Create layers of DRQN
        self.obs_n = obs_n
        self.act_n = act_n
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = torch.nn.LSTM(
            input_size=obs_n,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,   # input 形狀會是 [batch, seq, feat]
        )

        # 把最後一個 time step 的 hidden → Q 值
        self.fc = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, act_n)
        )
    
    def forward(self, x, hidden):
        ## TODO: Forward pass
         # 如果是單一步，就補成 seq_len = 1
        if x.dim() == 2:
            # [B, obs_n] -> [B, 1, obs_n]
            x = x.unsqueeze(1)

        # 丟進 LSTM
        # out: [B, seq_len, hidden_size]
        # hidden: (h_n, c_n)，各 [num_layers, B, hidden_size]
        out, hidden = self.lstm(x, hidden)

        # 只拿序列最後一個時間步的輸出做 Q(s,·)
        last_out = out[:, -1, :]          # [B, hidden_size]

        q_values = self.fc(last_out)      # [B, act_n]

        return q_values, hidden

# Create environment
# Create replay buffer
# Create network for Q(s, a)
# Create target network
# Create optimizer
def create_everything(seed):
    utils_rnn.seed.seed(seed)
    env = utils_rnn.envs.TimeLimit(utils_rnn.envs.PartiallyObservableCartPole(), 200)
    env.seed(seed)
    test_env = utils_rnn.envs.TimeLimit(utils_rnn.envs.PartiallyObservableCartPole(), 200)
    test_env.seed(seed)
    # buf = utils_rnn.buffers.ReplayBuffer(BUFSIZE, recurrent=True)
    buf = utils_rnn.buffers.ReplayBuffer(BUFSIZE)
    Q = DRQN().to(DEVICE)
    Qt = DRQN().to(DEVICE)
    Qt.load_state_dict(Q.state_dict())
    OPT = torch.optim.Adam(Q.parameters(), lr = LEARNING_RATE)
    return env, test_env, buf, Q, Qt, OPT

# Create epsilon-greedy policy
# TODO: Adjust this policy to handle hidden states?
def policy(env, obs, cell_mem=None):

    global EPSILON, EPSILON_END, STEPS_MAX, Q
    obs = t.f(obs).view(-1, OBS_N)  # Convert to torch tensor

    with torch.no_grad():
        # hidden=None -> LSTM 使用全 0 初始 hidden / cell
        q_values, cell_mem = Q(obs, hidden=cell_mem)   # q_values: [1, ACT_N]
            
    # With probability EPSILON, choose a random action
    # Rest of the time, choose argmax_a Q(s, a) 
    if np.random.rand() < EPSILON:
        action = np.random.randint(ACT_N)
    else:
        ## TODO: Implement greedy policy
        action = int(torch.argmax(q_values, dim=1).item())
    
    # Epsilon update rule: Keep reducing a small amount over
    # STEPS_MAX number of steps, and at the end, fix to EPSILON_END
    EPSILON = max(EPSILON_END, EPSILON - (1.0 / STEPS_MAX))
    
    return action, cell_mem

def greedy_policy(env, obs, cell_mem=None):
    global Q
    obs = t.f(obs).view(-1, OBS_N)
    with torch.no_grad():
        q_values, cell_mem = Q(obs, hidden=cell_mem)
        action = int(torch.argmax(q_values, dim=1).item())
    return action, cell_mem

# Update networks
def update_networks(epi, buf, Q, Qt, OPT):

    if len(buf.buf) < MINIBATCH_SIZE:
        return 0.0

    losses = []
    episodes = buf.sample(MINIBATCH_SIZE)

    for (states, actions, rewards) in episodes:
        T = len(rewards)
        if T == 0:
            continue

        states_t = torch.tensor(states, dtype=torch.float32, device=DEVICE)  # [T+1, 2]
        actions_t = torch.tensor(actions, dtype=torch.long, device=DEVICE)   # [T]
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=DEVICE) # [T]

        # [1, T+1, 2]
        seq_full = states_t.unsqueeze(0)

        # ============= Q network =============
        out_q, _ = Q.lstm(seq_full, None)  # [1, T+1, H]

        h_t = out_q[:, :-1, :]             # [1, T, H]

        B, TT, Hh = h_t.shape
        q_all = Q.fc(h_t.reshape(B*TT, Hh)).view(TT, ACT_N)  # [T, ACT_N]

        q_sa = q_all.gather(1, actions_t.view(-1,1)).squeeze(1)  # [T]

        # ============= Target network =============
        with torch.no_grad():
            out_t, _ = Qt.lstm(seq_full, None)     # [1, T+1, H]
            h_tp1 = out_t[:, 1:, :]                # [1, T, H]
            
            B2, TT2, H2 = h_tp1.shape
            q_next = Qt.fc(h_tp1.reshape(B2*TT2, H2)).view(TT2, ACT_N)

            max_q_next = q_next.max(dim=1)[0]

            targets = rewards_t + GAMMA * max_q_next
            targets[-1] = rewards_t[-1]   # terminal

        loss_epi = torch.nn.functional.mse_loss(q_sa, targets)
        losses.append(loss_epi)

    if len(losses) == 0:
        return 0.0

    OPT.zero_grad()
    total_loss = torch.stack(losses).mean()
    total_loss.backward()
    OPT.step()

    if epi % TARGET_NETWORK_UPDATE_FREQ == 0:
        Qt.load_state_dict(Q.state_dict())

    return total_loss.item()


# Play episodes
# Training function
def train(seed):

    global EPSILON, Q
    print("Seed=%d" % seed)

    # Create environment, buffer, Q, Q target, optimizer
    env, test_env, buf, Q, Qt, OPT = create_everything(seed)

    # epsilon greedy exploration
    EPSILON = STARTING_EPSILON

    testRs = [] 
    last25testRs = []
    print("Training:")
    pbar = tqdm.trange(EPISODES)
    totalLoss = 0.
    for epi in pbar:

        # Play an episode and log episodic reward
        S, A, R = utils_rnn.envs.play_episode_rb(env, policy, buf)
        
        # Train after collecting sufficient experience
        if epi >= TRAIN_AFTER_EPISODES:

            # Train for TRAIN_EPOCHS
            for tri in range(TRAIN_EPOCHS): 
                totalLoss = update_networks(epi, buf, Q, Qt, OPT)

        # Evaluate for TEST_EPISODES number of episodes
        Rews = []
        for epj in range(TEST_EPISODES):
            S, A, R = utils_rnn.envs.play_episode(test_env, policy, render = False)
            Rews += [sum(R)]
        testRs += [sum(Rews)/TEST_EPISODES]

        # Update progress bar
        last25testRs += [sum(testRs[-25:])/len(testRs[-25:])]
        pbar.set_description(f"R25({last25testRs[-1]}, loss:{totalLoss})")

    pbar.close()
    print("Training finished!")
    env.close()

    return last25testRs

# Plot mean curve and (mean-std, mean+std) curve with some transparency
# Clip the curves to be between 0, 200
def plot_arrays(vars, color, label):
    mean = np.mean(vars, axis=0)
    std = np.std(vars, axis=0)
    plt.plot(range(len(mean)), mean, color=color, label=label)
    plt.fill_between(range(len(mean)), np.maximum(mean-std, 0), np.minimum(mean+std,200), color=color, alpha=0.3)

if __name__ == "__main__":

    # Train for different seeds
    curves = []
    for seed in SEEDS:
        curves += [train(seed)]

    # Plot the curve for the given seeds
    plot_arrays(curves, 'b', 'drqn')
    plt.legend(loc='best')
    # plt.show()
    plt.savefig("DRQN_CartPole.png")