from turtle import update
import gym
import numpy as np
import utils.envs, utils.seed, utils.buffers, utils.torch
import torch
import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# C51
# Based on Slide 11
# cs.uwaterloo.ca/~ppoupart/teaching/cs885-winter22/slides/cs885-module5.pdf

# Constants
# SEEDS = [1]
SEEDS = [1, 2, 3, 4, 5]
t = utils.torch.TorchHelper()
DEVICE = t.device
OBS_N = 4               # State space size
ACT_N = 2               # Action space size
STARTING_EPSILON = 1.0  # Starting epsilon
STEPS_MAX = 10000       # Gradually reduce epsilon over these many steps
EPSILON_END = 0.1       # At the end, keep epsilon at this value
MINIBATCH_SIZE = 64     # How many examples to sample per train step
# GAMMA = 0.9            # Discount factor in episodic reward objective
GAMMA = 0.99           # Discount factor in episodic reward objective
LEARNING_RATE = 5e-4    # Learning rate for Adam optimizer
TRAIN_AFTER_EPISODES = 10   # Just collect episodes for these many episodes
TRAIN_EPOCHS = 25        # Train for these many epochs every time
BUFSIZE = 10000         # Replay buffer size
EPISODES = 500          # Total number of episodes to learn over
TEST_EPISODES = 2      # Test episodes
HIDDEN = 512            # Hidden nodes
TARGET_NETWORK_UPDATE_FREQ = 10 # Target network update frequency

# Suggested constants
ATOMS = 51              # Number of atoms for distributional network
ZRANGE = [0, 200]       # Range for Z projection

# Global variables
EPSILON = STARTING_EPSILON
Z = None

# Create environment
# Create replay buffer
# Create distributional networks
# Create optimizer
def create_everything(seed):
    utils.seed.seed(seed)
    env = utils.envs.TimeLimit(utils.envs.NoisyCartPole(), 500)
    env.seed(seed)
    test_env = utils.envs.TimeLimit(utils.envs.NoisyCartPole(), 500)
    test_env.seed(seed)
    buf = utils.buffers.ReplayBuffer(BUFSIZE)
    Z = torch.nn.Sequential(
        torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, ACT_N*ATOMS)
    ).to(DEVICE)
    Zt = torch.nn.Sequential(
        torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, ACT_N*ATOMS)
    ).to(DEVICE)
    Zt.load_state_dict(Z.state_dict()) 
    OPT = torch.optim.Adam(Z.parameters(), lr = LEARNING_RATE)
    return env, test_env, buf, Z, Zt, OPT

# Create epsilon-greedy policy
def policy(env, obs):

    global EPSILON, EPSILON_END, STEPS_MAX, Z
    obs = t.f(obs).view(-1, OBS_N)  # Convert to torch tensor
    
    # With probability EPSILON, choose a random action
    # Rest of the time, choose argmax_a Q(s, a) 
    if np.random.rand() < EPSILON:
        action = np.random.randint(ACT_N)
    else:
        ## TODO: use Z to compute greedy action
        with torch.no_grad():
            logits = Z(obs)
            logits = logits.view(-1, ACT_N, ATOMS)
            
             # 固定的 atom 支撐值 z_i
            z_atoms = torch.linspace(ZRANGE[0], ZRANGE[1], ATOMS).to(DEVICE)  # (ATOMS,)

            # 每個 action 的機率分佈 p(z | s, a)
            probs = torch.softmax(logits, dim=2)      # softmax over atoms

            # Q(s,a) = Σ_z z * p(z | s,a)
            q_values = torch.sum(probs * z_atoms.view(1, 1, -1), dim=2)  # (1, ACT_N)

            action = torch.argmax(q_values, dim=1).item()
    
    # Epsilon update rule: Keep reducing a small amount over
    # STEPS_MAX number of steps, and at the end, fix to EPSILON_END
    EPSILON = max(EPSILON_END, EPSILON - (1.0 / STEPS_MAX))
    
    return action

# # Update networks
# def update_networks(epi, buf, Z, Zt, OPT):
    
#     loss = 0.
#     ## TODO: Implement this function
#     miniBatch = buf.sample(MINIBATCH_SIZE, t)
#     S, A, R, S2, D = miniBatch
#     zqvalues = Zt(S).view(-1, ACT_N, ATOMS)
#     zqvalues_next = Zt(S2).view(-1, ACT_N, ATOMS)
#     qvalues_next = torch.sum(zqvalues_next, dim=2) / ATOMS
#     a_max = torch.argmax(qvalues_next, dim=1)
    
   
#     Tz = R.view(-1,1) + GAMMA * (1 - D.view(-1,1)) * zqvalues_next
#     Tz = Tz.clamp(ZRANGE[0], ZRANGE[1])
#     Tz_index = Tz - ZRANGE[0] / (ZRANGE[1] - ZRANGE[0]) * (ATOMS - 1)
#     l = torch.floor(Tz_index).long() #shape: [batch, atoms]
#     u = torch.ceil(Tz_index).long()
#     Tp = torch.zeros(MINIBATCH_SIZE, ATOMS).to(DEVICE) # target Z(s',a') projection shape: [batch, atoms]
#     for i in range(MINIBATCH_SIZE):
#         for j in range(ATOMS):
#             if l[i][j] == u[i][j]:
#                 Tp[i][l[i][j]] += zqvalues_next[i][a_max[i]][j]
#             else:
#                 Tp[i][l[i][j]] += zqvalues_next[i][a_max[i]][j] * (u[i][j] - Tz_index[i][j])
#                 Tp[i][u[i][j]] += zqvalues_next[i][a_max[i]][j] * (Tz_index[i][j] - l[i][j])
    
#     Z_log_probs = torch.log_softmax(zqvalues, dim=1)             # (B, ACT_N, ATOMS)
#     Z_log_probs = Z_log_probs.gather(1, A.view(-1,1,1).expand(-1,-1,ATOMS)).squeeze(1)  # (B, ATOMS)
#     losses = -torch.sum(Tp * Z_log_probs, dim=1)         # cross-entropy
#     losses = losses.mean()
    
#     OPT.zero_grad()
#     losses.backward()
#     OPT.step()

    
#     # Update target network
#     if epi%TARGET_NETWORK_UPDATE_FREQ==0:
#         Zt.load_state_dict(Z.state_dict())

#     return losses.item()

def update_networks(epi, buf, Z, Zt, OPT):

    # 1. 取一個 minibatch
    S, A, R, S2, D = buf.sample(MINIBATCH_SIZE, t)
    A = A.long()
    batch_size = S.size(0)

    v_min, v_max = ZRANGE
    z_atoms = torch.linspace(v_min, v_max, ATOMS).to(DEVICE)    # (N,)
    delta_z = (v_max - v_min) / (ATOMS - 1)

    # 2. 用 target 網路 Z 算下一狀態 Q(s',a)，選出 a*（Double DQN）
    with torch.no_grad():
        next_logits_online = Z(S2).view(batch_size, ACT_N, ATOMS)       # (B,A,N)
        next_probs_online = torch.softmax(next_logits_online, dim=2)    # (B,A,N)
        q_next = torch.sum(next_probs_online * z_atoms.view(1,1,-1),
                           dim=2)                                       # (B,A)
        a_max = torch.argmax(q_next, dim=1)                             # (B,)

        # 3. 用 target 網路 Zt 取出 a* 對應的分佈 p(z | s', a*)
        next_logits_target = Zt(S2).view(batch_size, ACT_N, ATOMS)      # (B,A,N)
        next_probs_target = torch.softmax(next_logits_target, dim=2)    # (B,A,N)
        p_next = next_probs_target[torch.arange(batch_size), a_max, :]  # (B,N)

        # 4. 算 Bellman 更新的 Tz = r + γ(1 - d) z_i，然後投影回 z_atoms
        Tz = R.unsqueeze(1) + GAMMA * (1 - D.unsqueeze(1)) * z_atoms.view(1, -1)
        Tz = Tz.clamp(v_min, v_max)  # (B,N)

        # 投影後的目標分佈 m
        m = torch.zeros(batch_size, ATOMS, device=DEVICE)

        for b in range(batch_size):
            for j in range(ATOMS):
                bj = (Tz[b, j] - v_min) / delta_z
                l = int(torch.floor(bj).item())
                u = int(torch.ceil(bj).item())

                # clamp
                if l < 0: l = 0
                if u < 0: u = 0
                if l > ATOMS-1: l = ATOMS-1
                if u > ATOMS-1: u = ATOMS-1

                pj = p_next[b, j].item()
                m[b, l] += pj * (u - bj)
                m[b, u] += pj * (bj - l)

    # 5. 現在網路 Z 對 (S, A) 輸出的分佈，算 cross-entropy loss
    logits = Z(S).view(batch_size, ACT_N, ATOMS)          # (B,A,N)
    logits_a = logits[torch.arange(batch_size), A, :]     # (B,N)
    log_probs = torch.log_softmax(logits_a, dim=1)        # (B,N)

    # cross-entropy: - Σ m * log q
    loss = -(m * log_probs).sum(dim=1).mean()

    # 6. 反向傳遞 + optimizer 更新
    OPT.zero_grad()
    loss.backward()
    OPT.step()

    # 7. 更新 target network
    if epi % TARGET_NETWORK_UPDATE_FREQ == 0:
        Zt.load_state_dict(Z.state_dict())

    return loss.item()


# Play episodes
# Training function
def train(seed):

    global EPSILON, Z
    print("Seed=%d" % seed)

    # Create environment, buffer, Z, Z target, optimizer
    env, test_env, buf, Z, Zt, OPT = create_everything(seed)

    # epsilon greedy exploration
    EPSILON = STARTING_EPSILON
    loss = 0.
    testRs = [] 
    last25testRs = []
    print("Training:")
    pbar = tqdm.trange(EPISODES)
    for epi in pbar:

        # Play an episode and log episodic reward
        S, A, R = utils.envs.play_episode_rb(env, policy, buf)
        
        # Train after collecting sufficient experience
        if epi >= TRAIN_AFTER_EPISODES:

            # Train for TRAIN_EPOCHS
            for tri in range(TRAIN_EPOCHS): 
                loss = update_networks(epi, buf, Z, Zt, OPT)

        # Evaluate for TEST_EPISODES number of episodes
        Rews = []
        for epj in range(TEST_EPISODES):
            S, A, R = utils.envs.play_episode(test_env, policy, render = False)
            Rews += [sum(R)]
        testRs += [sum(Rews)/TEST_EPISODES]

        # Update progress bar
        last25testRs += [sum(testRs[-25:])/len(testRs[-25:])]
        pbar.set_description(f"R25({last25testRs[-1]})、 loss:{loss:2.4f}")

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
    plt.fill_between(range(len(mean)), np.maximum(mean-std, 0), np.minimum(mean+std,500), color=color, alpha=0.3)

if __name__ == "__main__":

    # Train for different seeds
    curves = []
    for seed in SEEDS:
        curves += [train(seed)]

    # Plot the curve for the given seeds
    plot_arrays(curves, 'b', 'c51')
    plt.legend(loc='best')
    # plt.show()
    plt.savefig(f"C51_{ZRANGE[1]}.png")