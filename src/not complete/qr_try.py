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
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
import csv

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

def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)




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

        self._fc_layers = [tf.keras.layers.Dense(units, activation='relu') for units in fc_layer_params]
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
        print(f"Reshaped quantiles shape: {reshaped_quantiles.shape}")

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

def train_dqn(
        num_iterations=20000,
        initial_collect_steps=1000,
        collect_steps_per_iteration=10,
        replay_buffer_max_length=100000,
        fc_layer_params=(200,),
        batch_size=64,
        learning_rate=1e-3,
        log_interval=200,
        num_eval_episodes=10,
        eval_interval=1000
):
    file = open(constants.root + '/output/avg_returns_' + constants.algo + '_beta_' + str(constants.beta) + '.csv', 'w', newline='')
    avg_return_writer = csv.writer(file, delimiter=',')
    avg_return_writer.writerow(["Iteration", "AVG_Return", "AVG_Reward"])

    train_py_env = ClusterEnv()
    eval_py_env = train_py_env
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    num_quantiles = 51
    q_net = CustomQuantileNetwork(
        input_tensor_spec=train_env.observation_spec(),
        action_spec = train_env.action_spec(),
        fc_layer_params=fc_layer_params,
        num_quantiles=num_quantiles,
        name='CustomQuantileNetwork'
    )
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_step_counter = tf.compat.v1.Variable(0)
    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn = lambda td_targets, q_values : quantile_huber_loss(td_targets, q_values),
        #td_errors_loss_fn=quantile_huber_loss,
        train_step_counter=train_step_counter
    )
    agent.initialize()

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length
    )
    collect_data(train_env, agent.collect_policy, replay_buffer, steps=initial_collect_steps)
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

    iterations = range(0, num_iterations + 1, eval_interval)
    plt.plot(iterations, returns, label='Average Return')
    plt.plot(iterations, rewards, label='Average Reward')
    plt.ylabel('Average Return / Reward')
    plt.xlabel('Iterations')
    plt.legend()
    plt.show()

