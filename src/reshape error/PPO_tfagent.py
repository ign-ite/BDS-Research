from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import tensorflow as tf

from tf_agents.agents.ppo import ppo_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import PolicyInfo
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.networks import actor_distribution_network, value_network
import csv
import numpy as np
from itertools import chain
import seaborn as sns

import constants
from rm_environment import ClusterEnv

tf.compat.v1.enable_v2_behavior()

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
    cpu_utilization, mem_utilization = environment.pyenv.envs[0].get_resource_utilization()
    cpu_utilization_list.append(cpu_utilization)
    mem_utilization_list.append(mem_utilization)

    # Record deadline adherence after each step
    adherence = environment.pyenv.envs[0].get_deadline_adherence()
    adherence_list.append(adherence)




def collect_data(env, policy, buffer, steps, cpu_utilization_list, mem_utilization_list, adherence_list):
    for _ in range(steps):
        collect_step(env, policy, buffer, cpu_utilization_list, mem_utilization_list, adherence_list)

def plot_smoothed_rewards(rewards, window_size=100):
    smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    
    plt.figure(figsize=(12, 6))
    plt.plot(smoothed_rewards, label='Reward')
    plt.ylabel('Reward')
    plt.xlabel('Step')
    plt.legend()
    plt.show()

def plot_adherence(adherence_list, downsample_factor=1):
    # Downsample the data
    adherence_list = adherence_list[::downsample_factor]

    # Plot the adherence graph
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(adherence_list)), adherence_list, label='Deadline Adherence (%)')
    plt.ylabel('Adherence (%)')
    plt.xlabel('Episode')
    plt.legend()
    plt.title('Deadline Adherence Over Episodes')
    plt.show()

def plot_adherence_bar(adherence_list, downsample_factor=1, tick_spacing=10000, rotation_angle=45, figsize=(14, 8)):
    # Downsample the data
    adherence_list = adherence_list[::downsample_factor]

    # Plot the bar graph
    plt.figure(figsize=figsize)
    sns.barplot(x=list(range(len(adherence_list))), y=adherence_list, palette="viridis")
    plt.ylabel('Adherence (%)')
    plt.xlabel('Episode')
    plt.title('Deadline Adherence Bar Plot')

    # Set the x-axis ticks
    plt.xticks(ticks=np.arange(0, len(adherence_list), tick_spacing), rotation=rotation_angle)

    plt.show()

def plot_adherence_smoothed_line(adherence_list, downsample_factor=1, window_size=10):
    # Downsample the data
    adherence_list = adherence_list[::downsample_factor]

    # Apply smoothing
    smoothed_adherence = np.convolve(adherence_list, np.ones(window_size) / window_size, mode='valid')

    # Plot the smoothed line graph
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=list(range(len(smoothed_adherence))), y=smoothed_adherence, palette="viridis")
    plt.ylabel('Adherence (%)')
    plt.xlabel('Episode')
    plt.title('Smoothed Deadline Adherence Over Episodes')
    plt.show()

def plot_utilization(cpu_utilization, mem_utilization, downsample_factor=250, window_size=10):
    # Flatten the lists if they are nested
    cpu_utilization = list(chain.from_iterable(cpu_utilization))
    mem_utilization = list(chain.from_iterable(mem_utilization))

    # Normalize utilization values to a percentage
    cpu_utilization = [val * 100 for val in cpu_utilization]
    mem_utilization = [val * 100 for val in mem_utilization]

    # Downsample the data
    cpu_utilization_downsampled = cpu_utilization[::downsample_factor]
    mem_utilization_downsampled = mem_utilization[::downsample_factor]

    # Apply smoothing
    def smooth(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    smoothed_cpu = smooth(cpu_utilization_downsampled, window_size)
    smoothed_mem = smooth(mem_utilization_downsampled, window_size)

    # Multiply by 2 to adjust the range
    smoothed_cpu = [val * 2 for val in smoothed_cpu]
    smoothed_mem = [val * 2 for val in smoothed_mem]

    # Adjust the x-axis to match the length of the smoothed data
    steps = range(len(smoothed_cpu))

    # Set Seaborn style
    sns.set(style="whitegrid")

    # Plot the smoothed data
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=steps, y=smoothed_cpu, label='CPU Utilization', color='b')
    sns.lineplot(x=steps, y=smoothed_mem, label='Memory Utilization', color='orange')

    plt.ylabel('Utilization (%)')
    plt.xlabel('Step')
    plt.legend(loc='upper right')
    plt.title('CPU and Memory Utilization')

    # Add grid and improve aesthetics
    plt.grid(True)
    plt.tight_layout()

    plt.show()

def plot_throughput_bar(throughput_list, downsample_factor=1, tick_spacing=10000, rotation_angle=45, figsize=(14, 8)):
    # Downsample the data
    throughput_list = throughput_list[::downsample_factor]

    # Set Seaborn style
    sns.set(style="whitegrid")

    # Plot the bar graph
    plt.figure(figsize=figsize)
    sns.barplot(x=list(range(len(throughput_list))), y=throughput_list, palette='viridis')

    plt.ylabel('Throughput (%)')
    plt.xlabel('Episode')
    plt.title('Job Throughput Bar Plot')
    
    # Set the x-axis ticks
    plt.xticks(ticks=np.arange(0, len(throughput_list), tick_spacing), rotation=rotation_angle)
    plt.grid(True)
    plt.tight_layout()
    
    plt.show()

def train_ppo(
        # ***Hyperparameters***
        num_iterations=20000,  # @param {type:"integer"}
        initial_collect_steps=1000,  # @param {type:"integer"}
        collect_steps_per_iteration=1,  # @param {type:"integer"}
        replay_buffer_max_length=100000,  # @param {type:"integer"}
        batch_size=64,  # @param {type:"integer"}
        learning_rate=1e-3,  # @param {type:"number"}
        log_interval=200,  # @param {type:"integer"}
        num_eval_episodes=10,  # @param {type:"integer"}
        eval_interval=1000  # @param {type:"integer"}
):
    env = ClusterEnv()
    train_py_env = env
    eval_py_env = env

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    # Set up PPO agent
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=(64, 64)
    )
    
    value_net = value_network.ValueNetwork(
        train_env.observation_spec(),
        fc_layer_params=(64, 64)
    )

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    global_step = tf.compat.v1.train.get_or_create_global_step()

    tf_agent = ppo_agent.PPOAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        actor_net=actor_net,
        value_net=value_net,
        optimizer=optimizer,
        num_epochs=3,
        train_step_counter=global_step
    )
    tf_agent.initialize()

    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy

    

    random_policy = random_tf_policy.RandomTFPolicy(
    train_env.time_step_spec(), train_env.action_spec(), emit_log_probability=True)



    # Define the policy info spec to match what the policy generates
    policy_info_spec = {'log_probability': tensor_spec.TensorSpec(shape=(), dtype=tf.float32)}

    # Create the trajectory spec
    trajectory_spec = trajectory.Trajectory(
        step_type=train_env.time_step_spec().step_type,
        observation=train_env.time_step_spec().observation,
        action=train_env.action_spec(),
        policy_info=policy_info_spec,
        next_step_type=train_env.time_step_spec().step_type,
        reward=train_env.time_step_spec().reward,
        discount=train_env.time_step_spec().discount
    )


    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=trajectory_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length
    )

    dynamic_step_driver.DynamicStepDriver(
        train_env,
        random_policy,
        observers=[replay_buffer.add_batch],
        num_steps=initial_collect_steps
    ).run()

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        sample_batch_size=batch_size,
        num_steps=2
    ).prefetch(tf.data.experimental.AUTOTUNE)

    iterator = iter(dataset)

    # *** Training the agent***
    cpu_utilization_list = []
    mem_utilization_list = []
    adherence_list = []
    returns = []
    for _ in range(num_iterations):
        dynamic_step_driver.DynamicStepDriver(
            train_env,
            collect_policy,
            observers=[replay_buffer.add_batch],
            num_steps=collect_steps_per_iteration
        ).run()

        experience, unused_info = next(iterator)
        train_loss = tf_agent.train(experience).loss

        step = tf_agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0:
            avg_return, avg_steps, avg_adherence = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
            avg_reward, avg_steps, avg_adherence = compute_avg_reward(eval_env, tf_agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}, Average Steps per Episode = {2}, Average Reward per Step = {3}, Average Adherence = {4}%'.format(step, avg_return, avg_steps, avg_reward, avg_adherence))

            returns.append(avg_return)

    # ***Plot***
    plot_smoothed_rewards(returns)
    plot_adherence(adherence_list)
    plot_adherence_bar(adherence_list)
    plot_adherence_smoothed_line(adherence_list)
    plot_utilization(cpu_utilization_list, mem_utilization_list)
    plot_throughput_bar(adherence_list)  # Using adherence list for simplicity

