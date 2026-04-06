from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment

import constants
from rm_environment import ClusterEnv
from tf_agents.networks import network
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import epsilon_greedy_policy
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.policies import boltzmann_policy
import csv
import numpy as np
from itertools import chain
import seaborn as sns
import pandas as pd
# Bokeh imports
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
def collect_step(environment, policy, buffer,cpu_utilization_list, mem_utilization_list, adherence_list):
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

###########################################################################################################################################################################

def plot_throughput_heatmap(throughput_list, downsample_factor=100, grid_size=(10, 10)):
    # Downsample the data
    throughput_list = throughput_list[::downsample_factor]

    # Reshape the throughput list to match the grid size
    throughput_matrix = np.array(throughput_list[:grid_size[0]*grid_size[1]]).reshape(grid_size)

    # Plot the heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(throughput_matrix, cmap='viridis', annot=True, fmt=".1f")
    plt.ylabel('Job Throughput %')
    plt.xlabel('Episode')
    plt.title('Job Throughput Heatmap')
    plt.show()


# Function to plot episode costs using Bokeh
def plot_episode_costs_bokeh(episode_costs, downsample_factor=1, smooth_factor=1):
    # Normalize the episode costs
    max_cost = max(episode_costs)
    normalized_costs = [cost / max_cost for cost in episode_costs]
    
    # Downsample the episode costs
    normalized_costs_downsampled = normalized_costs[::downsample_factor]
    
    # Smooth the episode costs using a moving average
    smoothed_costs = pd.Series(normalized_costs_downsampled).rolling(window=smooth_factor, min_periods=1).mean().tolist()
    
    # Create the plot
    p = figure(title="Episode Costs Over Time", x_axis_label='Episode', y_axis_label='Normalized Cost', width=800, height=400)
    p.line(range(len(smoothed_costs)), smoothed_costs, legend_label="Episode Cost", line_width=2, line_color='green')
    p.grid.visible = True
    
    # Output to HTML file
    output_file("episode_costs.html")
    show(p)

# Function to plot episode costs using Bokeh
def plot_episode_time_bokeh(episode_avg_time, downsample_factor=1):

    episode_time_downsampled = episode_avg_time[::downsample_factor]

    p = figure(title="Episode Average Time", x_axis_label='Episode', y_axis_label='Time', width=800, height=400)
    p.line(range(len(episode_time_downsampled)), episode_time_downsampled, legend_label="Episode Time", line_width=2, line_color='cyan')
    p.grid.visible = True
    
    output_file("episode_time.html")
    show(p)

# Function to plot smoothed rewards using Bokeh
def plot_smoothed_rewards_bokeh(rewards, window_size=100):

    smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    
    p = figure(title="Smoothed Rewards", x_axis_label='Step', y_axis_label='Reward', width=800, height=400)
    p.line(range(len(smoothed_rewards)), smoothed_rewards, legend_label="Reward", line_width=2, line_color='blue')
    p.grid.visible = True
    
    output_file("smoothed_rewards.html")
    show(p)

def plot_smoothed_rewards(rewards, window_size=100):
    smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    
    plt.figure(figsize=(12, 6))
    plt.plot(smoothed_rewards, label='Reward')
    plt.ylabel('Reward')
    plt.xlabel('Step')
    plt.legend()
    plt.show()

# Function to plot CPU and memory utilization using Bokeh
def plot_utilization_bokeh(cpu_utilization, mem_utilization, downsample_factor=250, window_size=10):

    # Flatten the nested lists
    cpu_utilization = list(chain.from_iterable(cpu_utilization))
    mem_utilization = list(chain.from_iterable(mem_utilization))

    # Convert to percentage
    cpu_utilization = [val * 100 for val in cpu_utilization]
    mem_utilization = [val * 100 for val in mem_utilization]

    cpu_utilization_downsampled = cpu_utilization[::downsample_factor]
    mem_utilization_downsampled = mem_utilization[::downsample_factor]

    # Apply smoothing
    def smooth(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    smoothed_cpu = smooth(cpu_utilization_downsampled, window_size)
    smoothed_mem = smooth(mem_utilization_downsampled, window_size)

    smoothed_cpu = [val * 2 for val in smoothed_cpu]
    smoothed_mem = [val * 2 for val in smoothed_mem]

    p = figure(title="CPU and Memory Utilization", x_axis_label='Step', y_axis_label='Utilization (%)', width=800, height=400)
    p.line(range(len(smoothed_cpu)), smoothed_cpu, legend_label="CPU Utilization", line_width=2, line_color='blue')
    p.line(range(len(smoothed_mem)), smoothed_mem, legend_label="Memory Utilization", line_width=2, line_color='orange')
    p.grid.visible = True
    
    output_file("utilization.html")
    show(p)

# Function to plot job throughput bar using Bokeh
def plot_throughput_bar_bokeh(throughput_list, downsample_factor=20):

    throughput_list = throughput_list[::downsample_factor]

    p = figure(title="Job Throughput Bar Plot", x_axis_label='Episode', y_axis_label='Throughput (%)', width=800, height=400)
    p.vbar(x=range(len(throughput_list)), top=throughput_list, width=0.9, legend_label="Job Throughput (%)")
    p.grid.visible = True
    
    output_file("throughput_bar.html")
    show(p)


# Function to plot deadline adherence as a bar chart using Bokeh
def plot_adherence_bar_bokeh(adherence_list, downsample_factor=10):

    adherence_list = [x * 100 for x in adherence_list[::downsample_factor]]  # Scale adherence values to percentage

    p = figure(title="Deadline Adherence Bar Plot", x_axis_label='Episode', y_axis_label='Adherence (%)', width=800, height=400)
    p.vbar(x=range(len(adherence_list)), top=adherence_list, width=0.9, legend_label="Adherence (%)")
    p.grid.visible = True
    
    output_file("adherence_bar.html")
    show(p)


# Function to plot smoothed deadline adherence over episodes using Bokeh
def plot_adherence_smoothed_line_bokeh(adherence_list, downsample_factor=10, window_size=20):

    adherence_list = [x * 100 for x in adherence_list[::downsample_factor]]  # Scale adherence values to percentage
    smoothed_adherence = np.convolve(adherence_list, np.ones(window_size) / window_size, mode='valid')

    p = figure(title="Smoothed Deadline Adherence Over Episodes", x_axis_label='Episode', y_axis_label='Adherence (%)', width=800, height=400)
    p.line(range(len(smoothed_adherence)), smoothed_adherence, legend_label="Deadline Adherence (%)", line_width=1, line_color='orange')
    p.grid.visible = True
    
    output_file("adherence_smoothed_line.html")
    show(p)

def flatten_list(nested_list):
    return list(chain.from_iterable(nested_list))

######################################################################################################################################################################################


class CustomQuantileNetwork(q_network.QNetwork):
    def __init__(self, input_tensor_spec,action_spec, fc_layer_params, num_quantiles,name=None):
        super(CustomQuantileNetwork, self).__init__(
            input_tensor_spec= input_tensor_spec,
            action_spec= action_spec,
            fc_layer_params= fc_layer_params,
            name=name
        )

        self._num_quantiles = num_quantiles
        self._action_spec = action_spec

        # Ensure action_spec has valid attributes
        #assert hasattr(action_spec, 'maximum'), "action_spec does not have attribute 'maximum'"
        #assert hasattr(action_spec, 'minimum'), "action_spec does not have attribute 'minimum'"

        # Print to verify values
        #print(f"Action Spec Maximum: {self._action_spec.maximum}")
        #print(f"Action Spec Minimum: {self._action_spec.minimum}")

        output_size = (self._action_spec.maximum - self._action_spec.minimum + 1)*num_quantiles

         # Define the network layers
        self._fc_layers = [
            tf.keras.layers.Dense(128, activation=None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(128, activation=None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ]

        #self._fc_layers = [tf.keras.layers.Dense(units, activation='relu') for units in fc_layer_params]
        # self._quantile_layer = tf.keras.layers.Dense(output_tensor_spec.shape.num_elements() * num_quantiles)
        self._quantile_layer = tf.keras.layers.Dense(output_size)

        

    def call(self, observation, step_type=None, network_state=(), training=False):
        x = tf.cast(tf.reshape(observation, [-1, observation.shape[-1]]), tf.float32)
        for layer in self._fc_layers:
            x = layer(x)
        quantiles = self._quantile_layer(x)

        # Correctly reshape the output tensor
        reshaped_quantiles = tf.reshape(quantiles, [-1, self._num_quantiles, self._action_spec.maximum - self._action_spec.minimum + 1])

        # Print shapes to debug
        #print(f"Observation shape: {observation.shape}")
        #print("x shape:", x.shape)
        #print(f"Quantiles shape: {quantiles.shape}")
        #print(f"Reshaped quantiles shape: {reshaped_quantiles.shape}")

        mean_quantiles = tf.reduce_mean(reshaped_quantiles, axis=1)
        #print(f"mean quantiles shape: {mean_quantiles.shape}")

        return mean_quantiles, network_state

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


def epsilon_decay(epsilon, decay_rate, min_epsilon):
    return max(min_epsilon, epsilon - decay_rate)

def temperature_decay(temperature, decay_rate, min_temperature):
    return max(min_temperature, temperature - decay_rate)




def train_qr_dqn(
        num_iterations=20000,
        initial_collect_steps=1000,
        collect_steps_per_iteration=10,
        replay_buffer_max_length=100000,
        fc_layer_params=(200,200),
        batch_size=256,
        learning_rate=1e-3,
        log_interval=200,
        num_eval_episodes=20,
        eval_interval=1000,
        initial_epsilon=1.0,
        decay_rate=0.0001,
        min_epsilon=0.1,
        # initial_temperature=1.0,
        # temperature_decay_rate=0.0001,
        # min_temperature=0.1
):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(constants.root, 'output', timestamp)
    os.makedirs(output_dir)

    file = open(constants.root + '/output/avg_returns_' + constants.algo + '_beta_' + str(constants.beta) + '.csv', 'w',
                newline='')
    avg_return_writer = csv.writer(file, delimiter=',')
    avg_return_writer.writerow(["Iteration", "AVG_Return", "AVG_Reward"])

    train_py_env = ClusterEnv()
    eval_py_env = train_py_env
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    num_quantiles = 51
    q_net = CustomQuantileNetwork(
        input_tensor_spec=train_env.observation_spec(),
        action_spec=train_env.action_spec(),
        fc_layer_params=fc_layer_params,
        num_quantiles=num_quantiles,
        name='CustomQuantileNetwork'
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)

    train_step_counter = tf.compat.v1.Variable(0)
    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=quantile_huber_loss,
        n_step_update=2, # Set n_step_update to 3 for multi-step bootstrapping
        target_update_period=500,  # Double Q-learning
        gamma=0.99,
        train_step_counter=train_step_counter
    )
    agent.initialize()

    
    policy = agent.collect_policy

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length
    )

    cpu_utilization_list = []
    mem_utilization_list = []
    throughput_list = []
    adherence_list = []
    collect_data(train_env, policy, replay_buffer, initial_collect_steps, cpu_utilization_list, mem_utilization_list, adherence_list)

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=3
    ).prefetch(3)
    iterator = iter(dataset)

    agent.train = common.function(agent.train)
    agent.train_step_counter.assign(0)

    avg_return, _, avg_adherence = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    avg_reward, _, _ = compute_avg_reward(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]
    rewards = [avg_reward]
    adherences = [avg_adherence]
    simple_rewards = []
    losses = []
    avg_losses = []
    episode_costs = []
    episode_avg_time = []

    epsilon = initial_epsilon
    # temperature = initial_temperature

    for _ in range(num_iterations):
        policy._epsilon = epsilon
        for _ in range(collect_steps_per_iteration):
            collect_step(train_env, policy, replay_buffer, cpu_utilization_list, mem_utilization_list, adherence_list)
            time_step = train_env.current_time_step()
            simple_rewards.append(time_step.reward.numpy()[0])

        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss
        losses.append(train_loss.numpy())
        step = agent.train_step_counter.numpy()

        if (step + 1) % eval_interval == 0:
            avg_loss = sum(losses[-eval_interval:]) / eval_interval
            avg_losses.append(avg_loss)

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        throughput = train_env.pyenv.envs[0].calculate_throughput()
        throughput_percentage = (throughput) * 1000
        throughput_list.append(throughput_percentage)

        # Collect and store episode cost
        episode_cost = train_env.pyenv.envs[0].get_vm_cost()
        episode_costs.append(episode_cost)
        #print(f'step = {step}: Episode Cost = {episode_cost}')

        episode_time = train_env.pyenv.envs[0].calculate_avg_time()
        episode_avg_time.append(episode_time)

        if step % eval_interval == 0:
            avg_return, _, avg_adherence = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            avg_reward, _, _ = compute_avg_reward(eval_env, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}, Average Reward = {2}'.format(step, avg_return, avg_reward))
            avg_return_writer.writerow([step, avg_return, avg_reward])
            returns.append(avg_return)
            rewards.append(avg_reward)
            adherences.append(avg_adherence)

        epsilon = epsilon_decay(epsilon, decay_rate, min_epsilon)
 

    iterations = range(0, num_iterations + 1, eval_interval)
    plt.plot(iterations, returns, label='Average Return')
    plt.plot(iterations, rewards, label='Average Reward')
    plt.ylabel('Average Return / Reward')
    plt.xlabel('Iterations')
    plt.legend()
    plt.show()

    plt.plot(simple_rewards, label='Reward')
    plt.ylabel('Reward')
    plt.xlabel('Step')
    plt.legend()
    plt.show()

    avg_loss_iterations = range(eval_interval, num_iterations + 1, eval_interval)
    plt.plot(avg_loss_iterations, avg_losses, label='Average Loss')
    plt.ylabel('Average Loss')
    plt.xlabel('Iterations')
    plt.legend()
    plt.show()




    plot_episode_costs_bokeh(episode_costs)

    plot_episode_time_bokeh(episode_avg_time)

    plot_throughput_heatmap(throughput_list)

    plot_smoothed_rewards(simple_rewards)
    plot_smoothed_rewards_bokeh(simple_rewards)

    plot_utilization_bokeh(cpu_utilization_list, mem_utilization_list)

    plot_throughput_bar_bokeh(throughput_list)

    plot_adherence_bar_bokeh(adherence_list)

    plot_adherence_smoothed_line_bokeh(adherence_list)

    flat_cpu_utilization_list = flatten_list(cpu_utilization_list)
    flat_mem_utilization_list = flatten_list(mem_utilization_list)

    avg_cpu_utilization = sum(flat_cpu_utilization_list) / len(flat_cpu_utilization_list) if flat_cpu_utilization_list else 0
    avg_mem_utilization = sum(flat_mem_utilization_list) / len(flat_mem_utilization_list) if flat_mem_utilization_list else 0


    # Save cost data to CSV
    episode_costs_csv = os.path.join(output_dir, 'episode_costs.csv')
    save_to_csv(episode_costs_csv, ['Episode Costs'], [[val] for val in episode_costs])

    # Save cost data to CSV
    episode_time_csv = os.path.join(output_dir, 'episode_time.csv')
    save_to_csv(episode_time_csv, ['Episode Time'], [[val] for val in episode_avg_time])



    utilization_csv = os.path.join(output_dir, 'utilization.csv')
    utilization_data = zip(flat_cpu_utilization_list, flat_mem_utilization_list)
    save_to_csv(utilization_csv, ["CPU Utilization", "Memory Utilization"], utilization_data)

    throughput_csv = os.path.join(output_dir, 'throughput.csv')
    save_to_csv(throughput_csv, ["Throughput"], [[val] for val in throughput_list])

    adherence_csv = os.path.join(output_dir, 'adherence.csv')
    save_to_csv(adherence_csv, ["Adherence"], [[val] for val in adherence_list])

    rewards_csv = os.path.join(output_dir, 'rewards.csv')
    save_to_csv(rewards_csv, ['Rewards'], [[val] for val in simple_rewards])

    losses_csv = os.path.join(output_dir, 'losses.csv')
    save_to_csv(losses_csv, ['Losses'], [[val] for val in losses])

    avg_losses_csv = os.path.join(output_dir, 'avg_losses.csv')
    save_to_csv(avg_losses_csv, ['Average Losses'], [[val] for val in avg_losses])

    utilization_csv = os.path.join(output_dir, 'avg_utilization.csv')
    save_to_csv(utilization_csv, ['Average CPU Utilization', 'Average Memory Utilization'], [[avg_cpu_utilization], [avg_mem_utilization]])


