import os
import ray
import sys
import torch
import random
import numpy as np

from collections import deque


# https://github.com/ray-project/ray/issues/10839
def inside_tune():
    return ray.tune.is_session_enabled()


# https://stackoverflow.com/questions/38634988/check-if-program-runs-in-debug-mode
def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None


def set_rng(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def env_seed(env, seed):
    set_rng(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)


"""
The following code block comes from the `d3rlpy` library, 
but it has been modified to return both the average and 
standard deviation of the online evaluation.
"""


def evaluate_on_environment(env, n_trials=100, epsilon=0.0, text=True, render=False):
    """ Returns scorer function of evaluation on environment.

    This function returns scorer function, which is suitable to the standard
    scikit-learn scorer function style.
    The metrics of the scorer function is ideal metrics to evaluate the
    resulted policies.

    .. code-block:: python

        import gym

        from d3rlpy.algos import DQN
        from d3rlpy.metrics.scorer import evaluate_on_environment


        env = gym.make('CartPole-v0')

        scorer = evaluate_on_environment(env)

        cql = CQL()

        mean_episode_return = scorer(cql)


    Args:
        env (gym.Env): gym-styled environment.
        n_trials (int): the number of trials.
        epsilon (float): noise factor for epsilon-greedy policy.
        text (bool): flag to render text output.
        render (bool): flag to render environment.

    Returns:
        callable: scorer function.


    """

    def scorer(algo, *args):
        if text:
            print('Evaluating online for {} episodes.'.format(n_trials))
        scores_window = deque(maxlen=100)  # last 100 scores
        for trial_idx in range(n_trials):
            observation, info = env.reset()
            episode_reward = 0.0
            while True:
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = torch.argmax(algo.predict(torch.tensor(np.array([observation])))).item()
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

                if render:
                    env.render()

                if terminated or truncated:
                    break

            if text and len(scores_window) > 0:
                print('\rEpisode: {}\tAverage Score: {:.6f}'.format(trial_idx + 1, np.mean(scores_window)), end='')
                if trial_idx > 0 and trial_idx % 100 == 0:
                    print('\rEpisode: {}\tAverage Score: {:.6f}'.format(trial_idx + 1, np.mean(scores_window)))

            scores_window.append(episode_reward)
        if text:
            print()
        return np.mean(scores_window)

    return scorer
