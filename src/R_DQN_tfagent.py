import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.networks import network
import tensorflow_probability as tfp
from tf_agents.policies import random_tf_policy
from tf_agents.policies import categorical_q_policy
from tf_agents.policies import greedy_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
import numpy as np
import pandas as pd
import math
import logging
from rm_environment import ClusterEnv  # Make sure this is correctly defined
import constants
import csv
from itertools import chain
import seaborn as sns
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


class NoisyDense(layers.Layer):
    def __init__(self, units, sigma_init=0.017, **kwargs):
        super(NoisyDense, self).__init__(**kwargs)
        self.units = units
        self.sigma_init = sigma_init

    def build(self, input_shape):
        self.w_mu = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=tf.keras.initializers.RandomUniform(minval=-1 / math.sqrt(input_shape[-1]), maxval=1 / math.sqrt(input_shape[-1])),
            trainable=True,
            name='w_mu')
        self.w_sigma = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=tf.keras.initializers.Constant(self.sigma_init / math.sqrt(input_shape[-1])),
            trainable=True,
            name='w_sigma')
        self.b_mu = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.RandomUniform(minval=-1 / math.sqrt(input_shape[-1]), maxval=1 / math.sqrt(input_shape[-1])),
            trainable=True,
            name='b_mu')
        self.b_sigma = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.Constant(self.sigma_init / math.sqrt(input_shape[-1])),
            trainable=True,
            name='b_sigma')

    def call(self, inputs, training=False):
        if training:
            w_noise = tf.random.normal(shape=(inputs.shape[-1], self.units))
            b_noise = tf.random.normal(shape=(self.units,))
            w = self.w_mu + self.w_sigma * w_noise
            b = self.b_mu + self.b_sigma * b_noise
        else:
            w = self.w_mu
            b = self.b_mu
        return tf.matmul(inputs, w) + b




class RainbowQNetwork(network.Network):
    def __init__(self, input_tensor_spec, action_spec, num_atoms=51, fc_layer_params=(200,), name='RainbowQNetwork'):
        super(RainbowQNetwork, self).__init__(input_tensor_spec=input_tensor_spec, state_spec=(), name=name)
        self._action_spec = action_spec
        self._num_actions = action_spec.maximum - action_spec.minimum + 1
        self._num_atoms = num_atoms

        self.flatten = layers.Flatten()
        self.noisy_dense1 = NoisyDense(128)
        self.noisy_dense2 = NoisyDense(128)
        self.advantage = NoisyDense(self._num_actions * self._num_atoms)
        self.value = NoisyDense(self._num_atoms)

    @property
    def num_atoms(self):
        return self._num_atoms

    def call(self, inputs, step_type=None, network_state=(), training=False):
        x = tf.cast(inputs, tf.float32)  # Ensure inputs are float32
        x = self.flatten(x)
        x = self.noisy_dense1(x, training=training)
        x = self.noisy_dense2(x, training=training)

        advantage = self.advantage(x, training=training)
        value = self.value(x, training=training)

        advantage = tf.reshape(advantage, [-1, self._num_actions, self._num_atoms])
        value = tf.reshape(value, [-1, 1, self._num_atoms])

        # Calculate Q-values distribution
        q_values = value + advantage - tf.reduce_mean(advantage, axis=1, keepdims=True)
        q_values = tf.reshape(q_values, [-1, self._num_actions, self._num_atoms])

        # Transform Q-values to match DqnAgent expectation
        q_values_summed = tf.reduce_sum(q_values, axis=-1)

        return q_values_summed, network_state




class PrioritizedReplayBuffer(tf_uniform_replay_buffer.TFUniformReplayBuffer):
    def __init__(self, *args, alpha=0.6, beta=0.4, anneal_step=100000, **kwargs):
        super(PrioritizedReplayBuffer, self).__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.anneal_step = anneal_step
        self.priorities = []

    def add_batch(self, items, priorities=None):
        if priorities is None:
            priorities = tf.ones((items.step_type.shape[0],), dtype=tf.float32)
        priorities = tf.pow(priorities + 1e-6, self.alpha)
        self.priorities.extend(priorities.numpy())
        super(PrioritizedReplayBuffer, self).add_batch(items)

    def get_next(self, *args, **kwargs):
        if not self.priorities:
            return super(PrioritizedReplayBuffer, self).get_next(*args, **kwargs)
        probabilities = np.array(self.priorities) / np.sum(self.priorities)
        indices = np.random.choice(len(self.priorities), size=kwargs['sample_batch_size'], p=probabilities)
        samples = [self._storage[i] for i in indices]
        weights = np.power(len(self.priorities) * probabilities[indices], -self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta / self.anneal_step)
        experience = tf.nest.map_structure(lambda t: tf.stack([t[i] for i in indices]), self._storage)
        weights_tensor = tf.convert_to_tensor(weights, dtype=tf.float32)
        return experience, weights_tensor



##################################################################################################################################################################

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


def plot_metrics(returns, losses, log_interval):
    iterations = range(0, len(returns) * log_interval, log_interval)
    plt.plot(iterations, returns, label='Average Return')
    plt.plot(iterations, losses, label='Average Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Average Return / Loss')
    plt.legend()
    plt.show()



logging.basicConfig(level=logging.INFO)

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

##########################################################################################################################################################################################



def flatten_list(nested_list):
    return list(chain.from_iterable(nested_list))



def train_rainbow_dqn(
        # ***Hyperparameters***
        num_iterations=10000,  # @param {type:"integer"}
        initial_collect_steps=1000,  # @param {type:"integer"}
        collect_steps_per_iteration=10,  # @param {type:"integer"}
        replay_buffer_max_length=100000,  # @param {type:"integer"}
        fc_layer_params=(200,),
        batch_size=128,  # @param {type:"integer"}
        initial_learning_rate=9e-4,  # @param {type:"number"} try changing this to 5e-4
        log_interval=200,  # @param {type:"integer"}
        num_eval_episodes=10,  # @param {type:"integer"}
        eval_interval=1000,  # @param {type:"integer"}
    ):


     # Print all hyperparameters
    print("Training with the following hyperparameters:")
    print(f"num_iterations: {num_iterations}")
    print(f"initial_collect_steps: {initial_collect_steps}")
    print(f"collect_steps_per_iteration: {collect_steps_per_iteration}")
    print(f"replay_buffer_max_length: {replay_buffer_max_length}")
    print(f"fc_layer_params: {fc_layer_params}")
    print(f"batch_size: {batch_size}")
    print(f"learning_rate: {initial_learning_rate}")
    print(f"log_interval: {log_interval}")
    print(f"num_eval_episodes: {num_eval_episodes}")
    print(f"eval_interval: {eval_interval}")
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(constants.root, 'output', timestamp)
    os.makedirs(output_dir)

    file = open(constants.root + '/output/avg_returns_' + constants.algo + '_beta_' + str(constants.beta) + '.csv', 'w',
                newline='')
    avg_return_writer = csv.writer(file, delimiter=',')
    avg_return_writer.writerow(["Iteration", "AVG_Return", "AVG_Reward"])


    # *** Environment***s
    train_py_env = tf_py_environment.TFPyEnvironment(ClusterEnv())
    eval_py_env = tf_py_environment.TFPyEnvironment(ClusterEnv())

    # Ensure the action_spec has the necessary attributes
    action_spec = train_py_env.action_spec()
    action_spec = tf.TensorSpec(
        shape=action_spec.shape, 
        dtype=action_spec.dtype, 
        name=action_spec.name
    )
    
    # ***Agent***
    q_net = RainbowQNetwork(
        train_py_env.observation_spec(), 
        train_py_env.action_spec(), 
        num_atoms=51, 
        fc_layer_params=fc_layer_params)
    
    """learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=num_iterations,
        end_learning_rate=1e-4)"""
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=initial_learning_rate,
        #epsilon=1e-07,
        #amsgrad=False,
    )
    train_step_counter = tf.Variable(0)





#this requires q_summed_values
    agent = dqn_agent.DqnAgent(
        train_py_env.time_step_spec(),
        train_py_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_huber_loss,
        gamma=0.99,
        train_step_counter=train_step_counter,
        n_step_update=2, # Set n_step_update to 3 for multi-step bootstrapping
        target_update_period=200,  # Double Q-learning
        epsilon_greedy=lambda: tf.maximum(0.1, 1 - train_step_counter.numpy() / num_iterations)
    
    )
    agent.initialize()

    replay_buffer = PrioritizedReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_py_env.batch_size,
        max_length=replay_buffer_max_length,
        alpha=0.6,
        beta=0.4,
        anneal_step=num_iterations
    )


    # *** Policies ***
    random_policy = random_tf_policy.RandomTFPolicy(train_py_env.time_step_spec(), train_py_env.action_spec())
    greedy_eval_policy = greedy_policy.GreedyPolicy(agent.policy)
    replay_observer = [replay_buffer.add_batch]



    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=3,
        single_deterministic_pass=False).prefetch(3)  # Ensure num_steps is 2
    
    iterator = iter(dataset)
  
    # *** Agent Training ***
    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)
    agent.train_step_counter.assign(0)

    avg_return, _, avg_adherence = compute_avg_return(eval_py_env, agent.policy, num_eval_episodes)
    avg_reward, _, _ = compute_avg_reward(eval_py_env, agent.policy, num_eval_episodes)
    returns = [avg_return]
    rewards = [avg_reward]
    adherences = [avg_adherence]  # Store initial adherence
    losses = []
    avg_losses = []
    cpu_utilization_list = []
    mem_utilization_list = []
    adherence_list = []
    throughput_list = []
    simple_rewards = []  # List to store simple rewardssimple_rewards = []  # List to store simple rewards
    episode_costs = []
    episode_avg_time = []

    #losses = []

    # Collect initial data
    print("Collecting initial data...")
    collect_data(train_py_env, agent.collect_policy, replay_buffer, initial_collect_steps, cpu_utilization_list, mem_utilization_list, adherence_list)


    # Check the size of the replay buffer after initial collection
    initial_buffer_size = replay_buffer.num_frames().numpy()
    print(f"Replay buffer size after initial collection: {initial_buffer_size}")

    for _ in range(num_iterations):
        # Collect a few steps using collect_policy and save to the replay buffer.
        for _ in range(collect_steps_per_iteration):
            time_step = train_py_env.reset()
            while not time_step.is_last():
                collect_step(train_py_env, agent.collect_policy, replay_buffer, cpu_utilization_list, mem_utilization_list, adherence_list)
                # Log simple reward at each step
                time_step = train_py_env.current_time_step()
                simple_rewards.append(time_step.reward.numpy()[0])

        experience, sample_info = next(iterator)
    
        # Ensure weights are of the correct type and shape
        weights = sample_info.probabilities
        #print(f"Weights before casting: {weights}")  # Debugging
        weights = tf.cast(weights, tf.float32)
        weights = tf.reshape(weights, [-1, 1])
        #print(f"Weights after casting: {weights}")  # Debugging

        # Perform a training step with the correct weights
        train_loss = agent.train(experience, weights=weights).loss
        losses.append(train_loss)

        step = agent.train_step_counter.numpy()

        if (step + 1) % eval_interval == 0:
            avg_loss = sum(losses[-eval_interval:]) / eval_interval
            avg_losses.append(avg_loss)

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        # Calculate throughput after each episode
        throughput = train_py_env.pyenv.envs[0].calculate_throughput()
        throughput_percentage = (throughput) * 1000
        throughput_list.append(throughput_percentage)

        # Collect and store episode cost
        episode_cost = train_py_env.pyenv.envs[0].get_vm_cost()
        episode_costs.append(episode_cost)
        #print(f'step = {step}: Episode Cost = {episode_cost}')

        episode_time = train_py_env.pyenv.envs[0].calculate_avg_time()
        episode_avg_time.append(episode_time)

        if step % eval_interval == 0:
            avg_return, _, avg_adherence = compute_avg_return(eval_py_env, agent.policy, num_eval_episodes)
            avg_reward, _, _ = compute_avg_reward(eval_py_env, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}, Average Reward = {2}'.format(step, avg_return, avg_reward))
            avg_return_writer.writerow([step, avg_return, avg_reward])
            returns.append(avg_return)
            rewards.append(avg_reward)
            adherences.append(avg_adherence)  # Append adherence
    #plot_metrics(returns, losses, log_interval)

    # *** Visualizations ***
    iterations = range(0, num_iterations + 1, eval_interval)
    plt.plot(iterations, returns, label='Average Return')
    plt.plot(iterations, rewards, label='Average Reward')
    plt.ylabel('Average Return / Reward')
    plt.xlabel('Iterations') 
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

    # Flatten the utilization lists
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



    # Save utilization data to CSV
    utilization_csv = os.path.join(output_dir, 'utilization.csv')
    utilization_data = zip(flat_cpu_utilization_list, flat_mem_utilization_list)
    save_to_csv(utilization_csv, ["CPU Utilization", "Memory Utilization"], utilization_data)

    # Save throughput data to CSV
    throughput_csv = os.path.join(output_dir, 'throughput.csv')
    save_to_csv(throughput_csv, ["Throughput"], [[val] for val in throughput_list])

    # Save adherence data to CSV
    adherence_csv = os.path.join(output_dir, 'adherence.csv')
    save_to_csv(adherence_csv, ["Adherence"], [[val] for val in adherence_list])

    # Save rewards data to CSV
    rewards_csv = os.path.join(output_dir, 'rewards.csv')
    save_to_csv(rewards_csv, ['Rewards'], [[val] for val in simple_rewards])

    # Save losses data to CSV
    losses_csv = os.path.join(output_dir, 'losses.csv')
    save_to_csv(losses_csv, ['Losses'], [[val] for val in losses])

    # Save average losses data to CSV
    avg_losses_csv = os.path.join(output_dir, 'avg_losses.csv')
    save_to_csv(avg_losses_csv, ['Average Losses'], [[val] for val in avg_losses])


    # Save utilization data to CSV
    utilization_csv = os.path.join(output_dir, 'avg_utilization.csv')
    save_to_csv(utilization_csv, ['Average CPU Utilization', 'Average Memory Utilization'], [[avg_cpu_utilization], [avg_mem_utilization]])
    
# Call the training function
#train_rainbow_dqn(num_iterations=20000, log_interval=100, eval_interval=1000, num_eval_episodes=10, initial_collect_steps=1000)




        