import torch

from ray import tune
from utils import debugger_is_active

"""
    Discuss which hyperparameters may be important to tune and why. Then, discuss what ranges or choices might
    be reasonable for those hyperparameters you have chosen.
"""


def configuration_settings(environment_name):
    # default configurations
    config = {
        'seed': 0,  # the seed to use for the random number generators
        'lr': 3e-3,  # the learning rate, typically should be between 1e-4 to 1e-1
        'gamma': 0.99,  # the discount factor, range of values is within [0, 1] where 0 is myopic behavior
        'current_eps': 1.0,  # the rate at which the agent will take a random action
        'min_eps': 0.01,  # the minimum allowed epsilon value
        'eps_decay': 0.99,  # the rate at which we decrease epsilon, range of values is within (0, 1)
        'target_update': 10,  # the number of timestamps we wait before updating the target network
        'batch_size': 512,  # the size of the batch used in the experience replay
        'memory_size': 10000,  # the size of the agent's memory to store experiences for later replay
        'env_name': environment_name,  # the name of the Gym environment to be solved
        'max_episode_length': 200,  # the maximum length of each episode, once reached, episode will be forced to end
        'max_episodes': 100,  # the maximum number of episodes allowed for online training
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")  # enable cuda (GPU) if possible
    }
    if isinstance(environment_name, str) and environment_name == 'MountainCar-v0':
        config['max_episodes'], config['avg_goal'] = 300, -120
    elif isinstance(environment_name, str) and environment_name == 'CartPole-v1':
        config['max_episode_length'], config['avg_goal'] = 500, 200

    if not debugger_is_active():  # Ray will later be active; prepare search for hyperparameter tuning
        config['lr'] = tune.loguniform(5e-4, 1e-3)
        config['batch_size'] = tune.choice([64, 128, 256, 512])
        if isinstance(environment_name, str) and environment_name == 'MountainCar-v0':
            config['max_reports'], config['report'] = 5, 100  # report every 100 episodes, for at most 5 times
        elif isinstance(environment_name, str) and environment_name == 'CartPole-v1':
            config['max_reports'], config['report'] = 10, 50  # report every 50 episodes, for at most 10 times
    return config
