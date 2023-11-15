import torch
import random
import numpy as np

from torch.autograd import Variable
from collections import namedtuple, deque


"""
This code has been extended from:
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


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


class DeepQAgent:
    """ The Deep Q Network. """

    def __init__(self, policy_network, criterion, optimizer, target_network=None, memory=None):
        self.policy_network = policy_network
        self.criterion = criterion
        self.optimizer = optimizer
        self.target_network = target_network  # Double Deep Q-Learning to stabilize performance
        self.memory = memory  # experience replay by remembering previous experiences

    def update(self, states, updated_q_values):
        """Update the weights of the policy network given a batch of training samples. """
        predicted_q_values = self.policy_network(torch.Tensor(states))
        loss = self.criterion(predicted_q_values, Variable(torch.Tensor(updated_q_values)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, states):
        """ Compute Q values for all actions using the DQL. """
        with torch.no_grad():
            return self.policy_network(torch.Tensor(states))

    def select_action(self, states, config):
        sample = random.random()
        if sample > config['current_eps']:
            with torch.no_grad():
                return self.policy_network(torch.Tensor(states)).argmax()
        else:
            n_actions = self.policy_network[-1].out_features
            return torch.tensor(random.randrange(n_actions), device=config['device'], dtype=torch.long)

    def replay(self, config):
        if self.memory is None or len(self.memory) < config['batch_size']:
            return None
        else:
            transitions = self.memory.sample(config['batch_size'])
            """
            Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
            detailed explanation). This converts batch-array of Transitions
            to Transition of batch-arrays.
            """
            batch = Transition(*zip(*transitions))

            """
            Compute a mask of non-final states and concatenate the batch elements
            (a final state would've been the one after which simulation ended).
            """
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not True, batch.done)), device=config['device'], dtype=torch.bool)
            if not isinstance(batch.next_state, torch.Tensor):
                next_states = torch.Tensor(np.array(batch.next_state))
            is_nonterminal_state = torch.tensor(~np.array(batch.done)).long().unsqueeze(-1)
            non_final_next_states = next_states[torch.where(is_nonterminal_state)[0]]
            states = torch.tensor(np.array(batch.state))
            actions = torch.tensor(np.array(batch.action))
            rewards = torch.tensor(np.array([reward.item() for reward in batch.reward]))

            """
            Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            columns of the actions taken. These are the actions which would've been taken
            for each batch state according to the policy network.
            """
            state_action_values = self.policy_network(states).gather(1, actions.unsqueeze(dim=-1))

            """
            Compute V(s_{t+1}) for all next states. Expected values of actions for non_final_next_states are 
            computed based on the "older" target_net; selecting their best reward with max(1)[0]. This is merged 
            based on the mask, such that we'll have either the expected state value or 0 in case the state was final.
            """
            next_state_values = torch.zeros(config['batch_size'], device=config['device'])
            next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(dim=1)[0].detach()
            """
            Compute the expected Q values.
            """
            expected_state_action_values = (next_state_values * config['gamma']) + rewards

            loss = self.criterion(state_action_values.float(), expected_state_action_values.unsqueeze(1).float())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
