from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import network
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
import csv
import constants
from rm_environment import ClusterEnv

tf.compat.v1.enable_v2_behavior()


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

def compute_avg_reward(environment, policy, num_episodes=10):
    total_rewards = 0.0
    total_steps = 0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_rewards = 0.0
        episode_steps = 0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_rewards += time_step.reward
            episode_steps += 1
        total_rewards += episode_rewards
        total_steps += episode_steps
    avg_reward = total_rewards / total_steps if total_steps > 0 else 0
    return avg_reward.numpy()[0]

def collect_step(environment, policy, buffer, max_priority):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    buffer.add_batch(traj)
    max_priority = max(max_priority, buffer.priority.max())

def collect_data(env, policy, buffer, steps, max_priority):
    for _ in range(steps):
        collect_step(env, policy, buffer, max_priority)



class CustomPrioritizedReplayBuffer(tf_uniform_replay_buffer.TFUniformReplayBuffer):
    def __init__(self, data_spec, batch_size, max_length, alpha=0.6):
        super(CustomPrioritizedReplayBuffer, self).__init__(data_spec, batch_size, max_length)
        self.alpha = alpha
        self.priority = np.zeros((max_length,), dtype=np.float32)
        self.max_priority = 1.0
        self.next_idx = 0
        self.num_in_buffer = 0

    def add_batch(self, items):
        indices = np.arange(self.next_idx, self.next_idx + items.observation.shape[0]) % self.capacity
        self._store_batch(items, indices)
        self.priority[indices] = self.max_priority
        self.next_idx = (self.next_idx + items.observation.shape[0]) % self.capacity
        self.num_in_buffer = min(self.num_in_buffer + items.observation.shape[0], self.capacity)

    def get_next(self, sample_batch_size):
        if self.num_in_buffer == 0:
            return super().get_next(sample_batch_size)
        probs = self.priority[:self.num_in_buffer] ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(self.num_in_buffer, sample_batch_size, p=probs)
        return self._get(indices), indices

    def update_priorities(self, indices, priorities):
        self.priority[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())

class CustomDuelingQuantileNetwork(network.Network):
    def __init__(self, input_tensor_spec, action_spec, fc_layer_params, num_quantiles, name='CustomDuelingQuantileNetwork'):
        super(CustomDuelingQuantileNetwork, self).__init__(input_tensor_spec, state_spec=(), name=name)

        self._num_quantiles = num_quantiles
        self._action_spec = action_spec

        self._encoder = [tf.keras.layers.Dense(units, activation='relu') for units in fc_layer_params]

        self._value_layer = tf.keras.layers.Dense(num_quantiles)
        self._advantage_layer = tf.keras.layers.Dense(action_spec.maximum - action_spec.minimum + 1)

    def call(self, observation, step_type=None, network_state=(), training=False):
        x = tf.cast(tf.reshape(observation, [-1, observation.shape[-1]]), tf.float32)
        for layer in self._encoder:
            x = layer(x)
        
        value = self._value_layer(x)
        advantage = self._advantage_layer(x)

        reshaped_value = tf.expand_dims(value, -1)
        reshaped_advantage = tf.expand_dims(advantage, -2)

        mean_advantage = tf.reduce_mean(reshaped_advantage, axis=-1, keepdims=True)
        q_values = reshaped_value + reshaped_advantage - mean_advantage

        return q_values, network_state

def quantile_huber_loss(quantiles, target, actions=None, gamma=0.99, huber_delta=1.0):
    quantiles = tf.convert_to_tensor(quantiles, dtype=tf.float32)
    target = tf.convert_to_tensor(target, dtype=tf.float32)

    if actions is not None:
        actions = tf.expand_dims(actions, axis=-1)

    if len(quantiles.shape) == 1:
        quantiles = tf.reshape(quantiles, [-1, 1])

    batch_size = tf.shape(quantiles)[0]
    num_quantiles = tf.shape(quantiles)[1]

    target = tf.reshape(target, [-1, num_quantiles])

    td_errors = target - quantiles

    huber_loss = tf.where(
        tf.abs(td_errors) <= huber_delta,
        0.5 * tf.square(td_errors),
        huber_delta * (tf.abs(td_errors) - 0.5 * huber_delta)
    )

    tau = (tf.range(num_quantiles, dtype=tf.float32) + 0.5) / tf.cast(num_quantiles, tf.float32)
    tau = tf.expand_dims(tau, axis=0)
    tau = tf.expand_dims(tau, axis=0)

    quantile_loss = tf.abs(tau - tf.cast(td_errors < 0.0, tf.float32)) * huber_loss
    loss = tf.reduce_sum(quantile_loss, axis=-1)

    return tf.reduce_mean(loss)

def noisy_dense(units, activation=None):
    return tf.keras.layers.Dense(units, activation=activation,
        kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.5),
        bias_initializer=tf.keras.initializers.RandomNormal(0, 0.5))



def train_rainbow(
        num_iterations=20000,
        initial_collect_steps=1000,
        collect_steps_per_iteration=10,
        replay_buffer_max_length=100000,
        fc_layer_params=(200,),
        batch_size=64,
        learning_rate=1e-3,
        log_interval=200,
        num_eval_episodes=10,
        eval_interval=1000,
        alpha=0.6,
        beta_start=0.4,
        beta_end=1.0,
        beta_annealing_steps=20000
):
    file = open(constants.root + '/output/avg_returns_' + constants.algo + '_beta_' + str(constants.beta) + '.csv', 'w', newline='')
    avg_return_writer = csv.writer(file, delimiter=',')
    avg_return_writer.writerow(["Iteration", "AVG_Return", "AVG_Reward"])

    train_py_env = ClusterEnv()
    eval_py_env = train_py_env
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    num_quantiles = 51
    q_net = CustomDuelingQuantileNetwork(
        input_tensor_spec=train_env.observation_spec(),
        action_spec=train_env.action_spec(),
        fc_layer_params=fc_layer_params,
        num_quantiles=num_quantiles,
        name='CustomDuelingQuantileNetwork'
    )

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_step_counter = tf.compat.v1.Variable(0)
    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=lambda td_targets, q_values: quantile_huber_loss(td_targets, q_values),
        train_step_counter=train_step_counter,
        target_update_period=1000,  # Update target network every 1000 steps
        gamma=0.99,  # Discount factor
        n_step_update=3,  # Multi-step learning
        epsilon_greedy=None  # Use Noisy Networks
    )
    agent.initialize()

    replay_buffer = CustomPrioritizedReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length,
        alpha=alpha
    )

    beta = beta_start
    beta_increment = (beta_end - beta_start) / beta_annealing_steps
    max_priority = 1.0

    collect_data(train_env, agent.collect_policy, replay_buffer, steps=initial_collect_steps, max_priority=max_priority)
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2
    ).prefetch(3)
    iterator = iter(dataset)

    agent.train = common.function(agent.train)
    agent.train_step_counter.assign(0)
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    avg_reward = compute_avg_reward(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]
    rewards = [avg_reward]

    for iteration in range(num_iterations):
        for _ in range(collect_steps_per_iteration):
            collect_step(train_env, agent.collect_policy, replay_buffer, max_priority)
        experience, indices = next(iterator)
        train_loss = agent.train(experience).loss
        td_errors = agent.compute_td_error(experience).numpy()
        replay_buffer.update_priorities(indices.numpy(), np.abs(td_errors) + 1e-6)

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            avg_reward = compute_avg_reward(eval_env, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}, Average Reward = {2}'.format(step, avg_return, avg_reward))
            avg_return_writer.writerow([step, avg_return, avg_reward])
            returns.append(avg_return)
            rewards.append(avg_reward)

        if iteration < beta_annealing_steps:
            beta += beta_increment

    iterations = range(0, num_iterations + 1, eval_interval)
    plt.plot(iterations, returns, label='Average Return')
    plt.plot(iterations, rewards, label='Average Reward')
    plt.ylabel('Average Return / Reward')
    plt.xlabel('Iterations')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    train_rainbow()
