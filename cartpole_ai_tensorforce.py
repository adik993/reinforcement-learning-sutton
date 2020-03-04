from tensorforce.agents import PPOAgent, TRPOAgent
import gym
from time import clock

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    agent = PPOAgent(
        states_spec=dict(type='float', shape=(env.observation_space.shape[0],)),
        actions_spec=dict(type='int', num_actions=env.action_space.n),
        network_spec=[
            dict(type='dense', size=16)
        ],
        optimization_steps=5,
        likelihood_ratio_clipping=0.2,
        batch_size=4000,
        step_optimizer=dict(
            type='adam',
            learning_rate=1e-2
        ),
        summary_spec=dict(directory="./logs/ppo-4k-lr-0.01",
                          steps=50,
                          labels=['configuration',
                                  'gradients_scalar',
                                  'regularization',
                                  'inputs',
                                  'losses',
                                  'variables']
                          ),
    )
    start = clock()
    for ep in range(5000):
        steps = 0
        done = False
        obs = env.reset()
        while not done:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            agent.observe(done, reward)
            steps += 1
        print('Episode: {:4}, steps: {:3}, time: {}s'.format(ep, steps, clock() - start))
    print('Done in {}s'.format(clock() - start))
