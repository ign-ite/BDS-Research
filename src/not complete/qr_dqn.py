import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.agents import TFAgent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

import constants
from rm_environment import ClusterEnv
import csv

tf.compat.v1.enable_v2_behavior()

class CustomQRDQNAgent(TFAgent):
    def __init__(self, time_step_spec, action_spec, q_network, optimizer, num_quantiles=200, epsilon_greedy=0.1, gamma=0.99):
        self._num_quantiles = num_quantiles
        self._q_network = q_network
        self._optimizer = optimizer
        self._epsilon_greedy = epsilon_greedy
        self._gamma = gamma

        super(CustomQRDQNAgent, self).__init__(
            time_step_spec,
            action_spec,
            policy=self._create_policy(),
            collect_policy=self._create_collect_policy(),
            train_sequence_length=None
        )
    
    def _create_policy(self):
        return epsilon_greedy_policy.EpsilonGreedyPolicy(self._q_network, epsilon=self._epsilon_greedy)

    def _create_collect_policy(self):
        return epsilon_greedy_policy.EpsilonGreedyPolicy(self.policy, epsilon=self._epsilon_greedy)

    def _loss(self, experience, weights=None):
        # Compute the loss for QR-DQN
        batch_size = tf.shape(experience.observation)[0]
        num_actions = self._q_network.action_spec().maximum - self._q_network.action_spec().minimum + 1

        # Forward pass through the Q network
        quantile_values, _ = self._q_network(experience.observation)
        quantile_values = tf.reshape(quantile_values, (batch_size, num_actions, self._num_quantiles))

        # Get the selected actions and their corresponding quantile values
        actions = tf.cast(experience.action, dtype=tf.int32)
        actions_one_hot = tf.one_hot(actions, num_actions, dtype=tf.float32)
        selected_quantiles = tf.reduce_sum(quantile_values * actions_one_hot[:, :, None], axis=1)

        # Compute target quantile values
        next_quantile_values, _ = self._q_network(experience.next_observation)
        next_quantile_values = tf.reshape(next_quantile_values, (batch_size, num_actions, self._num_quantiles))
        max_next_quantiles = tf.reduce_max(next_quantile_values, axis=1)

        rewards = tf.cast(experience.reward, dtype=tf.float32)
        discounts = tf.cast(experience.discount, dtype=tf.float32)
        target_quantiles = rewards[:, None] + discounts[:, None] * self._gamma * max_next_quantiles

        # Compute the Huber loss
        td_errors = target_quantiles[:, None, :] - selected_quantiles[:, :, None]
        huber_loss = tf.compat.v1.losses.huber_loss(td_errors, reduction=tf.compat.v1.losses.Reduction.NONE)
        quantile_huber_loss = tf.reduce_mean(huber_loss, axis=2)

        return tf.reduce_mean(quantile_huber_loss)

    def _train(self, experience, weights=None):
        with tf.GradientTape() as tape:
            loss = self._loss(experience, weights)
        gradients = tape.gradient(loss, self._q_network.trainable_weights)
        self._optimizer.apply_gradients(zip(gradients, self._q_network.trainable_weights))

        return loss

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

def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)

def train_qr_dqn(
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
        num_quantiles=200
):
    file = open(constants.root + '/output/avg_returns_' + constants.algo + '_beta_' + str(constants.beta) + '.csv', 'w',
                newline='')
    avg_return_writer = csv.writer(file, delimiter=',')
    avg_return_writer.writerow(["Iteration", "AVG_Return", "AVG_Reward"])

    # *** Environment***
    train_py_env = ClusterEnv()
    eval_py_env = train_py_env

    # converting pyenv to tfenv
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    # ***Agent***
    q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    train_step_counter = tf.compat.v1.Variable(0)

    agent = CustomQRDQNAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        num_quantiles=num_quantiles)

    agent.initialize()

    # *** Policies ***
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())

    # *** Replay Buffer ***
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length)

    # Data Collection
    collect_data(train_env, random_policy, replay_buffer, steps=initial_collect_steps)

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2).prefetch(3)

    iterator = iter(dataset)

    # *** Agent Training ***
    agent.train = common.function(agent.train)

    # Reset the train step
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    avg_reward = compute_avg_reward(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]
    rewards = [avg_reward]

    for _ in range(num_iterations):
        for _ in range(collect_steps_per_iteration):
            collect_step(train_env, agent.collect_policy, replay_buffer)

        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

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

    # *** Visualizations ***
    iterations = range(0, num_iterations + 1, eval_interval)
    plt.plot(iterations, returns, label='Average Return')
    plt.plot(iterations, rewards, label='Average Reward')
    plt.ylabel('Average Return / Reward')
    plt.xlabel('Iterations')
    plt.legend()
    plt.show()

train_qr_dqn()
