import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import matplotlib.pyplot as plt
from rm_environment import ClusterEnv  # Importing the custom environment

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.weight_sigma.size(1)))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)

class RainbowDQN(nn.Module):
    def __init__(self, state_size, action_size, atoms=51, Vmin=-10, Vmax=10):
        super(RainbowDQN, self).__init__()
        self.atoms = atoms
        self.Vmin = Vmin
        self.Vmax = Vmax

        self.feature = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            NoisyLinear(128, 128),
            nn.ReLU(),
            NoisyLinear(128, atoms)
        )

        self.advantage_stream = nn.Sequential(
            NoisyLinear(128, 128),
            nn.ReLU(),
            NoisyLinear(128, action_size * atoms)
        )

        self.support = torch.linspace(self.Vmin, self.Vmax, self.atoms)

    def forward(self, state):
        dist = self.dist(state)
        dist = dist * self.support
        q = dist.sum(dim=2)
        return q

    def dist(self, state):
        feature = self.feature(state)
        value = self.value_stream(feature)
        advantage = self.advantage_stream(feature)

        value = value.view(-1, 1, self.atoms)
        advantage = advantage.view(-1, action_size, self.atoms)

        q_atoms = value + advantage - advantage.mean(1, keepdim=True)
        dist = F.softmax(q_atoms, dim=-1)
        return dist

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

def read_jobs_from_csv(file_path):
    df = pd.read_csv(file_path)
    df.columns = ['arrival_time', 'j_id', 'j_type', 'cpu', 'mem', 'ex', 'duration']  # Rename columns appropriately
    jobs = []
    state_size = 6  # Define the correct state size
    for _, row in df.iterrows():
        state = np.array([row['arrival_time'], row['j_id'], row['j_type'], row['cpu'], row['mem'], row['ex']])
        if len(state) != state_size:
            print(f"Skipping job with mismatched state size: {len(state)}")
            continue
        action = row['duration']
        reward = row['duration']
        next_state = state  # Assuming next_state as the same state for simplicity; adjust according to your logic
        done = False  # Adjust this based on your logic
        jobs.append((state, action, reward, next_state, done))
    return jobs

class Agent:
    def __init__(self, state_size, action_size, seed, buffer_size, batch_size, gamma, tau, lr, update_every, atoms=51, Vmin=-10, Vmax=10, jobs_file=None):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.update_every = update_every
        self.atoms = atoms
        self.Vmin = Vmin
        self.Vmax = Vmax

        self.qnetwork_local = RainbowDQN(state_size, action_size, atoms, Vmin, Vmax).to(device)
        self.qnetwork_target = RainbowDQN(state_size, action_size, atoms, Vmin, Vmax).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        self.memory = ReplayBuffer(buffer_size, batch_size)
        self.t_step = 0

        if jobs_file:
            jobs = read_jobs_from_csv(jobs_file)
            for job in jobs:
                self.memory.add(*job)

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.memory.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        dist = self.qnetwork_local.dist(next_states).detach() * self.qnetwork_local.support
        dist = dist.sum(dim=2)
        best_actions = dist.argmax(dim=1)
        next_dist = self.qnetwork_target.dist(next_states)
        next_dist = next_dist[range(next_dist.size(0)), best_actions]

        Tz = rewards + (1 - dones) * gamma * self.qnetwork_target.support
        Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)
        b = (Tz - self.Vmin) / self.delta_z
        l, u = b.floor().long(), b.ceil().long()

        offset = torch.linspace(0, (batch_size - 1) * atoms, batch_size).long().unsqueeze(1).expand(batch_size, atoms)
        m = states.new_zeros(batch_size, atoms)
        m.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
        m.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        dist = self.qnetwork_local.dist(states)
        log_p = torch.log(dist[range(batch_size), actions])
        loss = -(m * log_p).sum(1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

def dqn(n_episodes, max_t, eps_start, eps_end, eps_decay, agent):
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    env = ClusterEnv()  # Initialize the custom environment

    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)

        print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}", end="")
        if i_episode % 100 == 0:
            print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}")
        if np.mean(scores_window) >= 200.0:
            print(f"\nEnvironment solved in {i_episode - 100} episodes!\tAverage Score: {np.mean(scores_window):.2f}")
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores

if __name__ == "__main__":
    state_size = 22  # Ensure this matches the environment's observation spec
    action_size = 10  # Ensure this matches the environment's action spec
    seed = 0
    buffer_size = int(1e5)
    batch_size = 64
    gamma = 0.99
    tau = 1e-3
    lr = 5e-4
    update_every = 4
    jobs_file = 'D:\\sumit\\RM_DeepRL-master\\input\\jobs.csv'  # Provide the correct path to the jobs CSV file

    agent = Agent(state_size, action_size, seed, buffer_size, batch_size, gamma, tau, lr, update_every, jobs_file=jobs_file)

    n_episodes = 2000
    max_t = 1000
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.995

    scores = dqn(n_episodes, max_t, eps_start, eps_end, eps_decay, agent)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
