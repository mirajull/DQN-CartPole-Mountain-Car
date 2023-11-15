import gym
import torch
import numpy as np

from ray import tune
from itertools import count
from functools import partial
from ray.tune import CLIReporter
from config import configuration_settings  # go to config.py to make your edits
from ray.tune.schedulers import ASHAScheduler
from q_learning import ReplayMemory, DeepQAgent as DQA
from utils import env_seed, debugger_is_active, inside_tune, evaluate_on_environment


"""
Objectives:
    1. Try to achieve an average of 495+ reward for the last 100 episodes in Cart Pole.
    2. Discover what hyperparameters work well for solving the Mountain Car. The goal is achieving an average
        of -110 reward for the last 100 episodes.
    
    Please visit config.py to review a selection of possible hyperparameters to edit/tune.
"""


"""
This code has been extended from:
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
https://towardsdatascience.com/deep-q-learning-for-the-cartpole-44d761085c2f
"""


def neural_network(in_features, out_features, hidden_dim=64):
    """
    The function approximation to be used for the Deep Q-Learning.

    Args:
        in_features:  The number of features that describe a state in the environment.
        out_features:  The number of actions that are possible in the environment's states.
        hidden_dim:  The dimensionality of the hidden network's neurons.

    Returns:
        A feedforward neural network that has LeakyReLU activations.
    """
    return torch.nn.Sequential(
        torch.nn.Linear(in_features, hidden_dim),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(hidden_dim, hidden_dim * 2),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(hidden_dim * 2, out_features)
    )


def train_dqn(env, config, agent):
    episode_rewards = []
    for i_episode in range(config['max_episodes']):
        episode_rewards.append(0.0)
        # Initialize the environment and state
        state, _ = env.reset()
        for timestep in count():
            # Select and perform an action
            action = agent.select_action(state, config=config)
            next_state, reward, done, _, _ = env.step(action.item())
            episode_rewards[-1] += reward
            reward = torch.tensor([reward], device=config['device'])

            if agent.memory is not None:
                # Store the transition in memory
                agent.memory.push(state, action, reward, next_state, done)
            q_values = agent.predict(state).tolist()

            if done or timestep == config['max_episode_length'] - 1:
                if agent.memory is None:  # if we are not doing experience replay
                    q_values[action.item()] = reward.item()
                    agent.update(state, q_values)
                break  # terminate episode: either we are done or we reached maximum episode length

            if agent.memory is None:  # if we're not doing experience replay
                # Update network weights using the last step only
                q_values_next = agent.predict(next_state)
                q_values[action.item()] = reward + config['gamma'] * torch.max(q_values_next).item()
                agent.update(state, q_values)
            else:  # else use previous experiences to help learn our Q-values
                agent.replay(config)

            # Move to the next state
            state = next_state

            # Update the target network, copying all weights and biases in DQN
            if agent.target_network is not None and timestep % config['target_update'] == 0:
                agent.target_network.load_state_dict(agent.policy_network.state_dict())

        config['current_eps'] = max(config['current_eps'] * config['eps_decay'], config['min_eps'])
        avg_reward = np.array(episode_rewards[-100:]).mean()

        if inside_tune():
            if (i_episode % config['report'] == 0 and i_episode > 0) or avg_reward >= config['avg_goal']:
                tune.report(performance=avg_reward)
        else:
            print('\r Episode: {}\tLength: {}\tEpsilon: {:.2f}\tAverage Reward Last 100 Episodes: {:.2f}'
                  .format(i_episode, timestep + 1, config['current_eps'], avg_reward), end='')
            if avg_reward >= config['avg_goal']:
                break

    return agent


def start(config, checkpoint_dir=None, data_dir=None):
    print('Using seed {} to solve: {}'.format(config['seed'], config['env_name']))
    # https://github.com/openai/gym/issues/2540
    # https://github.com/openai/gym/pull/2671 (merged)
    env = gym.make(config['env_name'], render_mode=None)  # change render_mode to 'human' for display, else None
    env_seed(env, config['seed'])

    # Number of states
    in_features = env.observation_space.shape[0]
    # Number of actions
    out_features = env.action_space.n

    policy_net = neural_network(in_features, out_features, hidden_dim=64).to(config['device'])
    target_net = neural_network(in_features, out_features, hidden_dim=64).to(config['device'])
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(policy_net.parameters(), config['lr'])

    agent = DQA(policy_network=policy_net, criterion=criterion, optimizer=optimizer,
                target_network=target_net, memory=ReplayMemory(config['memory_size']))

    dqn = train_dqn(env, config, agent)
    print()
    _ = evaluate_on_environment(env, n_trials=10, text=True, render=True)(dqn)


if __name__ == '__main__':
    for env_name in ['CartPole-v1', 'MountainCar-v0']:
        if debugger_is_active():
            config = configuration_settings(env_name)
            start(config)
        else:
            config = configuration_settings(env_name)

            scheduler = ASHAScheduler(
                metric="performance",
                mode="max",
                max_t=config['max_reports'],
                grace_period=1,
                reduction_factor=2)
            reporter = CLIReporter(
                parameter_columns=["lr", "batch_size"],  # add the hyperparameters that you want to display here
                metric_columns=["performance", "training_iteration"])

            result = tune.run(
                partial(start, data_dir=None),
                resources_per_trial={"cpu": 1, "gpu": 0},
                config=config,
                num_samples=5,
                scheduler=scheduler,
                progress_reporter=reporter,
            )

            best_trial = result.get_best_trial('performance', 'max', 'last')
            print("Best trial config: {}".format(best_trial.config))
            print("Best trial final validation performance: {}".format(
                best_trial.last_result["performance"]))
