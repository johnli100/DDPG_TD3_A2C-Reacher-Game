import numpy as np
import torch
from torch import optim, nn
from collections import deque, namedtuple

LR = 0.00001

Transitions = namedtuple('Transitions',('states','actions','rewards','dones','log_probs','entropy','Vs'))

class A2CAgent():

    def __init__(self, state_dim=129, action_dim=20, n_step=4, entropy_weight=0., value_loss_weight=1.):
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_step = n_step
        self.entropy_weight = entropy_weight
        self.value_loss_weight = value_loss_weight

        self.network = Network(self.state_dim,self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=LR)
        self.memory = deque(maxlen = n_step + 1)

    def step(self, states):
        states = torch.tensor(states, dtype=torch.float, device=self.device)
        return self.network(states)

    def cache(self, states, actions, rewards, dones, log_probs, entropy, Vs):
        states = torch.tensor(data=states, dtype=torch.float, device=self.device).squeeze(0)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.long, device=self.device).unsqueeze(1)

        self.memory.append(Transitions(states, actions.squeeze(0), rewards, dones, log_probs, entropy, Vs))

    def reset_cache(self):
        self.memory.clear()

    def learn(self, Gs, Ads):
        Gs = torch.tensor(Gs,dtype=torch.float,device=self.device)
        Ads = torch.tensor(Ads,dtype=torch.float,device=self.device)
        policy_loss = -(self.memory[0].log_probs * Ads).mean()
        entropy = self.memory[0].entropy.mean()
        value_loss = nn.functional.mse_loss(self.memory[0].Vs, Gs) * 0.5
        loss = policy_loss - entropy * self.entropy_weight + value_loss * self.value_loss_weight

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 5.)
        self.optimizer.step()


class Network(nn.Module):
    def __init__(self, state_dim, action_dim, critic_hidden_dims=[400, 300], actor_hidden_dims=[400, 300]):
        super().__init__()
        actor_hidden_dims = [state_dim] + actor_hidden_dims
        self.layers_actor = nn.ModuleList([nn.Linear(actor_hidden_dims[i],actor_hidden_dims[i+1]) \
                                      for i in range(len(actor_hidden_dims)-1)])
        self.layer_output_actor = nn.Linear(actor_hidden_dims[-1], action_dim)

        critic_hidden_dims = [state_dim] + critic_hidden_dims
        self.layers_critic = nn.ModuleList([nn.Linear(critic_hidden_dims[i],critic_hidden_dims[i+1]) \
                                           for i in range(len(critic_hidden_dims)-1)])
        self.layer_output_critic = nn.Linear(critic_hidden_dims[-1], 1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, states):
        x = states
        for layer in self.layers_actor:
            x = layer(x)
            x = self.relu(x)
        phi = self.tanh(self.layer_output_actor(x))

        dist = torch.distributions.Normal(phi, nn.functional.softplus(self.std))
        actions = torch.clamp(dist.sample(), -1., 1.)
        log_probs = dist.log_prob(actions).sum(-1).unsqueeze(1)
        entropy = dist.entropy().sum(-1).unsqueeze(1)

        x = states
        for layer in self.layers_critic:
            x = layer(x)
            x = self.relu(x)
        Vs = self.layer_output_critic(x)

        return {'actions':actions, 'log_probs':log_probs, 'entropy':entropy, 'Vs':Vs}