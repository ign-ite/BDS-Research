from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import tensorflow as tf
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
import constants
from rm_environment import ClusterEnv
import csv
import numpy as np
from itertools import chain
import seaborn as sns
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook
import os
from datetime import datetime
tf.compat.v1.enable_v2_behavior()

# Function to save lists to CSV files
def save_to_csv(filename, header, data):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)

# ***Metrics and Evaluation ***
def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    total_steps = 0
    total_adherence = 0.0  # Initialize total adherence
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        episode_steps = 0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            episode_steps += 1

        total_return += episode_return
        total_steps += episode_steps

        # Get deadline adherence for the episode
        adherence = environment.pyenv.envs[0].get_deadline_adherence()
        total_adherence += adherence

    avg_return = total_return / num_episodes
    avg_steps_per_episode = total_steps / num_episodes
    avg_adherence = total_adherence / num_episodes  # Calculate average adherence
    return avg_return.numpy()[0], avg_steps_per_episode, avg_adherence  # Return avg adherence

def compute_avg_reward(environment, policy, num_episodes=10):
    total_rewards = 0.0
    total_steps = 0
    total_adherence = 0.0  # Initialize total adherence

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

        # Get deadline adherence for the episode
        adherence = environment.pyenv.envs[0].get_deadline_adherence()
        total_adherence += adherence

    avg_reward = total_rewards / total_steps if total_steps > 0 else 0
    avg_steps_per_episode = total_steps / num_episodes
    avg_adherence = total_adherence / num_episodes  # Calculate average adherence
    return avg_reward.numpy()[0], avg_steps_per_episode, avg_adherence  # Return avg adherence

# Data Collection
def collect_step(environment, policy, buffer, cpu_utilization_list, mem_utilization_list, adherence_list):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    buffer.add_batch(traj)

    # Record resource utilization after each step
    # Access the underlying PyEnvironment to get resource utilization
    cpu_utilization, mem_utilization = environment.pyenv.envs[0].get_resource_utilization()
    cpu_utilization_list.append(cpu_utilization)
    mem_utilization_list.append(mem_utilization)

    # Record deadline adherence after each step
    adherence = environment.pyenv.envs[0].get_deadline_adherence()
    adherence_list.append(adherence)

def collect_data(env, policy, buffer, steps, cpu_utilization_list, mem_utilization_list, adherence_list):
    for _ in range(steps):
        collect_step(env, policy, buffer, cpu_utilization_list, mem_utilization_list, adherence_list)

class IQNNetwork(tf.keras.Model):
    def __init__(self, observation_spec, action_spec, fc_layer_params, num_quantiles,name=None):
        super(IQNNetwork, self).__init__(
            input_tensor_spec= observation_spec,
            action_spec= action_spec,
            fc_layer_params= fc_layer_params,
            name=name
        )

        self._num_quantiles = num_quantiles
        self._action_spec = action_spec

        output_size = (self._action_spec.maximum - self._action_spec.minimum + 1)*num_quantiles

        # Define the fully connected layers
        self._fc_layers = [tf.keras.layers.Dense(units, activation='relu') for units in fc_layer_params]
        # Define the final output layer
        self._quantile_layer = tf.keras.layers.Dense(output_size)

    def call(self, observations, quantiles):
        x = tf.cast(tf.reshape(observations, [-1, observations.shape[-1]]), tf.float32)
        for layer in self._fc_layers:
            x = layer(x)

        quantiles_output = self._quantile_layer(x)
        reshaped_quantiles = tf.reshape(quantiles_output, [-1, self._num_quantiles, self._action_spec.maximum - self._action_spec.minimum + 1])

        mean_quantiles = tf.reduce_mean(reshaped_quantiles, axis=1)
        return mean_quantiles


class IQNAgent(tf.Module):
    def __init__(self, time_step_spec, action_spec, q_network, optimizer, num_quantiles=32, gamma=0.99):
        self._time_step_spec = time_step_spec
        self._action_spec = action_spec
        self._q_network = q_network
        self._optimizer = optimizer
        self._num_quantiles = num_quantiles
        self._gamma = gamma
        self._epsilon = tf.Variable(1.0, trainable=False)
        self._epsilon_decay = 0.995
        self._min_epsilon = 0.01

    def policy(self, time_step):
        q_values = self._q_network(time_step.observation, tf.random.uniform([self._num_quantiles]))
        mean_q_values = tf.reduce_mean(q_values, axis=1)
        if tf.random.uniform([]) < self._epsilon:
            action = tf.random.uniform([1], maxval=self._action_spec.maximum, dtype=tf.int64)
        else:
            action = tf.argmax(mean_q_values, axis=-1)
        return action

    def train(self, experience):
        observations, actions, rewards, next_observations, dones = experience
        quantiles = tf.random.uniform([tf.shape(observations)[0], self._num_quantiles])

        with tf.GradientTape() as tape:
            current_quantile_values = self._q_network(observations, quantiles)
            next_quantile_values = self._q_network(next_observations, quantiles)
            next_actions = tf.argmax(tf.reduce_mean(next_quantile_values, axis=1), axis=-1)
            target_quantile_values = tf.stop_gradient(rewards + self._gamma * (1.0 - dones) * next_quantile_values[:, :, next_actions])
            td_errors = target_quantile_values - current_quantile_values
            quantile_loss = quantile_huber_loss(td_errors, target_quantile_values, gamma=self._gamma)
            loss = tf.reduce_mean(quantile_loss)

        grads = tape.gradient(loss, self._q_network.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._q_network.trainable_variables))
        self._epsilon.assign(max(self._min_epsilon, self._epsilon * self._epsilon_decay))
        return loss


def quantile_huber_loss(quantiles, target, actions=None, gamma=0.99, huber_delta=1.0):
    quantiles = tf.convert_to_tensor(quantiles, dtype=tf.float32)
    target = tf.convert_to_tensor(target, dtype=tf.float32)

    #print tensor shapes for debugging
    #print("quantiles shape:", quantiles.shape)
    #print("target shape", target.shape)

    if actions is not None:
        actions = tf.expand_dims(actions, axis=-1)

    # ensure quantiles are 2D
    if len(quantiles.shape)== 1:
        quantiles = tf.reshape(quantiles, [-1,1])

    batch_size = tf.shape(quantiles)[0]
    num_quantiles = tf.shape(quantiles)[1]

    # ensure target has the correct shape
    target = tf.reshape(target, [-1, num_quantiles])
    # target = tf.expand_dims(target, axis=-2)

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



def train_iqn(
        num_iterations=20000,
        initial_collect_steps=1000,
        collect_steps_per_iteration=10,
        replay_buffer_max_length=100000,
        fc_layer_params=(200,),
        batch_size=128,
        learning_rate=1e-3,
        log_interval=200,
        num_eval_episodes=10,
        eval_interval=1000,
        num_quantiles=32
):

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(constants.root, 'output', timestamp)
    os.makedirs(output_dir)

    file = open(constants.root + '/output/avg_returns_' + constants.algo + '_beta_' + str(constants.beta) + '.csv', 'w', newline='')
    avg_return_writer = csv.writer(file, delimiter=',')
    avg_return_writer.writerow(["Iteration", "AVG_Return", "AVG_Reward"])

    max_possible_throughput = 100  # Set this to your baseline or maximum possible throughput

    # *** Environment***
    train_py_env = ClusterEnv()
    eval_py_env = train_py_env

    # converting pyenv to tfenv
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    # ***Agent***
    iqn_net = IQNNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params,
        num_quantiles=num_quantiles)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    agent = IQNAgent(
        time_step_spec=train_env.time_step_spec(),
        action_spec=train_env.action_spec(),
        q_network=iqn_net,
        optimizer=optimizer,
        num_quantiles=num_quantiles)

    # *** Replay Buffer ***
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.policy.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length)

    # Data Collection
    cpu_utilization_list = []
    mem_utilization_list = []
    throughput_list = []
    adherence_list = []  # Track deadline adherence
    collect_data(train_env, agent.policy, replay_buffer, initial_collect_steps, cpu_utilization_list, mem_utilization_list, adherence_list)

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2).prefetch(3)

    iterator = iter(dataset)

    # *** Agent Training ***
    agent.train = common.function(agent.train)

    # Evaluate the agent's policy once before training.
    avg_return, _, avg_adherence = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    avg_reward, _, _ = compute_avg_reward(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]
    rewards = [avg_reward]
    adherences = [avg_adherence]  # Store initial adherence
    simple_rewards = []  # List to store simple rewards
    losses = []
    avg_losses = []

    for _ in range(num_iterations):

        # Collect a few steps using collect_policy and save to the replay buffer.
        for _ in range(collect_steps_per_iteration):
            collect_step(train_env, agent.policy, replay_buffer, cpu_utilization_list, mem_utilization_list, adherence_list)
            # Log simple reward at each step
            time_step = train_env.current_time_step()
            simple_rewards.append(time_step.reward.numpy()[0])

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss
        losses.append(train_loss.numpy())

        step = agent.train_step_counter.numpy()

        if (step + 1) % eval_interval == 0:
            avg_loss = sum(losses[-eval_interval:]) / eval_interval
            avg_losses.append(avg_loss)

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        # Calculate throughput after each episode
        throughput = train_env.pyenv.envs[0].calculate_throughput()  
        throughput_percentage = (throughput) * 1000
        throughput_list.append(throughput_percentage)

        if step % eval_interval == 0:
            avg_return, _, avg_adherence = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            avg_reward, _, _ = compute_avg_reward(eval_env, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}, Average Reward = {2}'.format(step, avg_return, avg_reward))
            avg_return_writer.writerow([step, avg_return, avg_reward])
            returns.append(avg_return)
            rewards.append(avg_reward)
            adherences.append(avg_adherence)  # Append adherence

    # *** Visualizations ***
    iterations = range(0, num_iterations + 1, eval_interval)
    plt.plot(iterations, returns, label='Average Return')
    plt.plot(iterations, rewards, label='Average Reward')
    plt.ylabel('Average Return / Reward')
    plt.xlabel('Iterations') 
    plt.legend()
    plt.show()

     # Plot the simple rewards
    plt.plot(simple_rewards, label='Reward')
    plt.ylabel('Reward')
    plt.xlabel('Step')
    plt.legend()
    plt.show()

    # Plot the losses
    avg_loss_iterations = range(eval_interval, num_iterations + 1, eval_interval)
    plt.plot(avg_loss_iterations, avg_losses, label='Average Loss')
    plt.ylabel('Average Loss')
    plt.xlabel('Iterations') 
    plt.legend()
    plt.show()

# Initialize and train the IQN agent
train_iqn()
