from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network, network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from rb_env import ClusterEnv
import csv

tf.compat.v1.enable_v2_behavior()

# Add Noisy Layers
class NoisyDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(NoisyDense, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        
    def build(self, input_shape):
        self.w_mu = self.add_weight(name='w_mu', shape=(input_shape[-1], self.units),
                                    initializer='glorot_uniform', trainable=True)
        self.w_sigma = self.add_weight(name='w_sigma', shape=(input_shape[-1], self.units),
                                       initializer='glorot_uniform', trainable=True)
        self.b_mu = self.add_weight(name='b_mu', shape=(self.units,),
                                    initializer='zeros', trainable=True)
        self.b_sigma = self.add_weight(name='b_sigma', shape=(self.units,),
                                       initializer='zeros', trainable=True)

    def call(self, inputs):
        epsilon_w = tf.random.normal(self.w_mu.shape)
        epsilon_b = tf.random.normal(self.b_mu.shape)
        w = self.w_mu + self.w_sigma * epsilon_w
        b = self.b_mu + self.b_sigma * epsilon_b
        output = tf.matmul(inputs, w) + b
        if self.activation is not None:
            output = self.activation(output)
        return output


# Define Dueling QNetwork
class DuelingQNetwork(network.Network):

    def __init__(self, input_tensor_spec, action_spec, fc_layer_params=(100,), name='DuelingQNetwork'):
        super(DuelingQNetwork, self).__init__(input_tensor_spec=input_tensor_spec, state_spec=(), name=name)

        self._action_spec = action_spec
        self._num_actions = action_spec.maximum - action_spec.minimum + 1

        self.conv1 = layers.Conv2D(32, (8, 8), strides=4, activation='relu')
        self.conv2 = layers.Conv2D(64, (4, 4), strides=2, activation='relu')
        self.conv3 = layers.Conv2D(64, (3, 3), strides=1, activation='relu')

        self.flatten = layers.Flatten()
        
        self.fc_layers = [layers.Dense(num_units, activation='relu') for num_units in fc_layer_params]

        # Advantage and Value streams
        self.advantage_fc = layers.Dense(self._num_actions, activation=None)
        self.value_fc = layers.Dense(1, activation=None)

    class DuelingQNetwork(network.Network):
        def __init__(self, input_tensor_spec, action_spec, name='DuelingQNetwork'):
            super(DuelingQNetwork, self).__init__(
                input_tensor_spec=input_tensor_spec, state_spec=(), name=name)

            # Define your layers here
            self.flatten = tf.keras.layers.Flatten()
            self.dense1 = tf.keras.layers.Dense(128, activation='relu')
            self.advantage = tf.keras.layers.Dense(action_spec.maximum - action_spec.minimum + 1, name='advantage')
            self.value = tf.keras.layers.Dense(1, name='value')

        def call(self, observation, step_type=None, network_state=(), training=False):
            # Assuming 'observation' is your input tensor
            x = observation

            # Flatten the input tensor if it's not already flattened
            x = self.flatten(x)

            # Apply dense layers for dueling architecture
            x = self.dense1(x)
            advantage = self.advantage(x)
            value = self.value(x)

            # Dueling Q-network calculation
            q_values = value + tf.subtract(advantage, tf.reduce_mean(advantage, axis=-1, keepdims=True))

            return q_values

# Example usage:
# Define your action spec and observation spec appropriately for your environment
#action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=9, name='action')
#observation_spec = array_spec.BoundedArraySpec(shape=(22,), dtype=np.int32, minimum=0, maximum=255, name='observation')

# Instantiate your Dueling Q-network
#dueling_q_net = DuelingQNetwork(input_tensor_spec=observation_spec, action_spec=action_spec)


# Example usage:
# input_tensor_spec = tf.TensorSpec(shape=[84, 84, 1], dtype=tf.float32)
# action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=9)
# dueling_q_network = DuelingQNetwork(input_tensor_spec, action_spec)



# Prioritized Experience Replay (PER)
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
        experience = super(PrioritizedReplayBuffer, self).get_next(*args, **kwargs)
        return experience, weights


def train_rainbow_dqn(num_iterations=20000, log_interval=200, eval_interval=1000, num_eval_episodes=10):
    train_py_env = tf_py_environment.TFPyEnvironment(ClusterEnv())
    eval_py_env = tf_py_environment.TFPyEnvironment(ClusterEnv())

    q_net = DuelingQNetwork(train_py_env.observation_spec(),
                             train_py_env.action_spec(),
                             fc_layer_params=(100,))

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
        train_py_env.time_step_spec(),
        train_py_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_huber_loss,
        gamma=0.99,
        train_step_counter=train_step_counter,
        n_step_update=3
    )
    agent.initialize()

    replay_buffer = PrioritizedReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_py_env.batch_size,
        max_length=100000,
        alpha=0.6,
        beta=0.4,
        anneal_step=100000
    )

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

    for _ in range(num_iterations):
        time_step = train_py_env.current_time_step()
        action_step = agent.collect_policy.action(time_step)
        next_time_step = train_py_env.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        replay_buffer.add_batch(traj)

        experience, _ = replay_buffer.get_next(sample_batch_size=64)
        train_loss = agent.train(experience).loss

        if train_step_counter.numpy() % log_interval == 0:
            print(f'step = {train_step_counter.numpy()}: loss = {train_loss}')

        if train_step_counter.numpy() % eval_interval == 0:
            avg_return = compute_avg_return(eval_py_env, agent.policy, num_eval_episodes)
            print(f'step = {train_step_counter.numpy()}: Average Return = {avg_return}')

    checkpoint_dir = './checkpoint'
    train_checkpointer = common.Checkpointer(
        ckpt_dir=checkpoint_dir,
        max_to_keep=1,
        agent=agent,
        policy=agent.policy,
        replay_buffer=replay_buffer,
        global_step=train_step_counter
    )
    train_checkpointer.save(train_step_counter)

    steps = range(0, num_iterations + 1, eval_interval)
    returns = [compute_avg_return(eval_py_env, agent.policy, num_eval_episodes) for _ in steps]

    plt.plot(steps, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Step')
    plt.ylim(top=250)
    plt.show()

train_rainbow_dqn()