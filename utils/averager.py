from multiprocessing import cpu_count
from joblib import Parallel, delayed
import numpy as np
import plotly.graph_objs as go
import plotly.offline as py

from utils import calc_batch_size, Algorithm, generate_episode


class AlgorithmFactory:
    def create(self, *args, **kwargs) -> Algorithm:
        raise NotImplementedError('Implement me')


class AveragingTask:
    def run(self, batch_size, batch_idx):
        results = self.create_results()
        for i in range(batch_size):
            self.run_single(i, results)
        return results

    def run_single(self, i, results):
        raise NotImplementedError('Implement me')

    def create_results(self):
        raise NotImplementedError('Implement me')


class GymEpisodeTask(AveragingTask):
    def __init__(self, env, n_episodes, algorithm_factory: AlgorithmFactory, algo_params):
        self.env = env
        self.n_episodes = n_episodes
        self.algorithm_factory = algorithm_factory
        self.algo_params = algo_params

    def run_single(self, i, results):
        algorithm = self.algorithm_factory.create(*self.algo_params)
        for episode in range(self.n_episodes):
            steps = generate_episode(self.env, algorithm, render=False)
            results[episode] += steps
            print('Run: {:2}, params: {}, ep: {:3}, steps: {:4}'.format(i, self.algo_params, episode, steps))

    def create_results(self):
        return np.zeros(self.n_episodes)


class TaskFactory:
    def create(self, params) -> AveragingTask:
        raise NotImplementedError('Implement me')


class GymEpisodeTaskFactory(TaskFactory):
    def __init__(self, env, n_episodes, algorithm_factory: AlgorithmFactory):
        self.env = env
        self.n_episodes = n_episodes
        self.algorithm_factory = algorithm_factory

    def create(self, params) -> AveragingTask:
        return GymEpisodeTask(self.env, self.n_episodes, self.algorithm_factory, params)


def average(results, n_avg):
    return np.sum(results, axis=0) / n_avg


class Averager:
    def __init__(self, task_factory: TaskFactory):
        self.task_factory = task_factory

    def average(self, algo_params, n_avg, n_jobs=cpu_count(), merge=average):
        with Parallel(n_jobs=n_jobs) as parallel:
            jobs = []
            for batch_idx in range(n_jobs):
                task = self.task_factory.create(algo_params)
                batch_size = calc_batch_size(n_avg, n_jobs, batch_idx)
                jobs.append(delayed(task.run)(batch_size, batch_idx))
            results = parallel(jobs)
            return merge(results, n_avg)


def plot_scatters_from_dict(results, label_format: str, x=None):
    data = []
    for label, values in results.items():
        data.append(go.Scatter(y=values, x=x, name=label_format.format(label)))
    py.plot(data)
