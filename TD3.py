import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from collections import defaultdict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

# filename template for transitions data
# FILENAME_TEMPLATE = "./results/TD3_{}_seed{}_batch{}_trans.pkl"
FILENAME_TEMPLATE = "./results/TD3_{}_seed{}_trans.pkl"

class Actor(nn.Module):
        def __init__(self, state_dim, action_dim, max_action):
                super(Actor, self).__init__()

                self.l1 = nn.Linear(state_dim, 256)
                self.l2 = nn.Linear(256, 256)
                self.l3 = nn.Linear(256, action_dim)
                
                self.max_action = max_action
                

        def forward(self, state):
                a = F.relu(self.l1(state))
                a = F.relu(self.l2(a))
                return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
        def __init__(self, state_dim, action_dim):
                super(Critic, self).__init__()

                # Q1 architecture
                self.l1 = nn.Linear(state_dim + action_dim, 256)
                self.l2 = nn.Linear(256, 256)
                self.l3 = nn.Linear(256, 1)

                # Q2 architecture
                self.l4 = nn.Linear(state_dim + action_dim, 256)
                self.l5 = nn.Linear(256, 256)
                self.l6 = nn.Linear(256, 1)


        def forward(self, state, action):
                sa = torch.cat([state, action], 1)

                q1 = F.relu(self.l1(sa))
                q1 = F.relu(self.l2(q1))
                q1 = self.l3(q1)

                q2 = F.relu(self.l4(sa))
                q2 = F.relu(self.l5(q2))
                q2 = self.l6(q2)
                return q1, q2


        def Q1(self, state, action):
                sa = torch.cat([state, action], 1)

                q1 = F.relu(self.l1(sa))
                q1 = F.relu(self.l2(q1))
                q1 = self.l3(q1)
                return q1


# need this for pickle
def defaultdict_int_fn():
    return defaultdict(int)

class TD3(object):
        def __init__(
                self,
                state_dim,
                action_dim,
                max_action,
                discount=0.99,
                tau=0.005,
                policy_noise=0.2,
                noise_clip=0.5,
                policy_freq=2,
                seed=0,
                env_name="HalfCheetah-v3",
        ):

                self.actor = Actor(state_dim, action_dim, max_action).to(device)
                self.actor_target = copy.deepcopy(self.actor)
                self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

                self.critic = Critic(state_dim, action_dim).to(device)
                self.critic_target = copy.deepcopy(self.critic)
                self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

                self.max_action = max_action
                self.discount = discount
                self.tau = tau
                self.policy_noise = policy_noise
                self.noise_clip = noise_clip
                self.policy_freq = policy_freq

                self.total_it = 0

                # extra state storage
                self.seed = seed
                self.env_name = env_name
                # maps states to counts
                self.transitions = defaultdict(int)
                self.transitions_batch_id = 0
                self.num_transitions_per_batch = int(1e6)
                # state -> id
                self.state_to_id = {}
                # state_id -> { next_state_id -> freq }
                self.cond_freq = defaultdict(defaultdict_int_fn)
                self.free_id = 0


        def select_action(self, state):
                state = torch.FloatTensor(state.reshape(1, -1)).to(device)
                return self.actor(state).cpu().data.numpy().flatten()

        def save_transitions_v0(self, force=False):
            # import pdb; pdb.set_trace()
            if len(self.transitions) >= self.num_transitions_per_batch or force:
                # save transitions
                save_dict = { "transitions" : self.transitions }
                transitions_filename = FILENAME_TEMPLATE.format(self.env_name,
                        self.seed, self.transitions_batch_id)
                pickle.dump( save_dict, open(transitions_filename, "wb") )
                # reset transitions
                self.transitions = defaultdict(int)
                self.transitions_batch_id += 1

        def store_transitions_v0(self, state, action, next_state, reward, not_done):
            # loop through each state in batch
            for s in state:
                # https://stackoverflow.com/questions/53376786/convert-byte-array-back-to-numpy-array
                self.transitions[ s.detach().cpu().numpy().tobytes() ] += 1
            self.save_transitions()

        def save_transitions(self, force=False):
            # import pdb; pdb.set_trace()
            save_dict = {
                "cond_freq" : self.cond_freq,
                "state_to_id" : self.state_to_id,
            }
            save_filename = FILENAME_TEMPLATE.format(self.env_name,
                self.seed )
            pickle.dump( save_dict, open(save_filename, "wb") )

        def update_state_to_id( self, state ):
            if state not in self.state_to_id:
                self.state_to_id[ state ] = self.free_id
                self.free_id += 1

        def store_transitions(self, state, action, next_state, reward, not_done): 
            # update cond_freq map
            for s, ns in zip(state, next_state):
                s = s.detach().cpu().numpy().tobytes()
                ns = ns.detach().cpu().numpy().tobytes()
                self.update_state_to_id( s )
                self.update_state_to_id( ns )
                s_id, ns_id = self.state_to_id[s], self.state_to_id[ns]
                self.cond_freq[s_id][ns_id] += 1

        def train(self, replay_buffer, batch_size=256):
                self.total_it += 1

                # Sample replay buffer 
                state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
                self.store_transitions( state, action, next_state, reward, not_done )
                

                with torch.no_grad():
                        # Select action according to policy and add clipped noise
                        noise = (
                                torch.randn_like(action) * self.policy_noise
                        ).clamp(-self.noise_clip, self.noise_clip)
                        
                        next_action = (
                                self.actor_target(next_state) + noise
                        ).clamp(-self.max_action, self.max_action)

                        # Compute the target Q value
                        target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                        target_Q = torch.min(target_Q1, target_Q2)
                        target_Q = reward + not_done * self.discount * target_Q

                # Get current Q estimates
                current_Q1, current_Q2 = self.critic(state, action)

                # Compute critic loss
                critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

                # Optimize the critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                # Delayed policy updates
                if self.total_it % self.policy_freq == 0:

                        # Compute actor losse
                        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                        
                        # Optimize the actor 
                        self.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor_optimizer.step()

                        # Update the frozen target models
                        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


        def save(self, filename):
                torch.save(self.critic.state_dict(), filename + "_critic")
                torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
                
                torch.save(self.actor.state_dict(), filename + "_actor")
                torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


        def load(self, filename):
                self.critic.load_state_dict(torch.load(filename + "_critic"))
                self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
                self.critic_target = copy.deepcopy(self.critic)

                self.actor.load_state_dict(torch.load(filename + "_actor"))
                self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
                self.actor_target = copy.deepcopy(self.actor)
                
