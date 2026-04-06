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
import math
import logging
from rm_environment import ClusterEnv  # Make sure this is correctly defined

tf.compat.v1.enable_v2_behavior()

class NoisyDense(layers.Layer):
    def __init__(self, units, sigma_init=0.5, **kwargs):
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
    def __init__(self, input_tensor_spec, action_spec, num_atoms=51, fc_layer_params=(100,), name='RainbowQNetwork'):
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

    """def get_q_values(self, inputs, step_type=None, network_state=(), training=False):
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

        return q_values, network_state"""


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

def plot_metrics(returns, losses, log_interval):
    iterations = range(0, len(returns) * log_interval, log_interval)
    plt.plot(iterations, returns, label='Average Return')
    plt.plot(iterations, losses, label='Average Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Average Return / Loss')
    plt.legend()
    plt.show()

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

logging.basicConfig(level=logging.INFO)

def train_rainbow_dqn(num_iterations=20000, log_interval=100, eval_interval=1000, num_eval_episodes=10, initial_collect_steps=1000):
    train_py_env = tf_py_environment.TFPyEnvironment(ClusterEnv())
    eval_py_env = tf_py_environment.TFPyEnvironment(ClusterEnv())

    # Ensure the action_spec has the necessary attributes
    action_spec = train_py_env.action_spec()
    action_spec = tf.TensorSpec(shape=action_spec.shape, dtype=action_spec.dtype, name=action_spec.name)
    

    q_net = RainbowQNetwork(train_py_env.observation_spec(), train_py_env.action_spec(), num_atoms=51, fc_layer_params=(100,))
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
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
        target_update_period=1000,  # Double Q-learning
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

    random_policy = random_tf_policy.RandomTFPolicy(train_py_env.time_step_spec(), train_py_env.action_spec())
    replay_observer = [replay_buffer.add_batch]

    # Use a greedy policy instead of epsilon-greedy
    #greedy_policy_instance = greedy_policy.GreedyPolicy(agent.collect_policy)
    #replay_observer = [replay_buffer.add_batch]


    """class CustomCategoricalQPolicy(categorical_q_policy.CategoricalQPolicy):
        def _distribution(self, time_step, policy_state):
            q_values, _ = self._q_network(time_step.observation, step_type=policy_state, training=False)
            q_values = tf.reshape(q_values, [-1, self._num_actions, self._num_atoms])
            logits = q_values / self._tau
            return tfp.distributions.Categorical(logits=logits), policy_state
"""
#this require q_values
    """collect_policy = categorical_q_policy.CategoricalQPolicy(
        time_step_spec=agent.time_step_spec,
        action_spec=agent.action_spec,
        q_network=q_net,
        min_q_value=0.0,  # You can adjust these values based on your problem
        max_q_value=10.0,    # You can adjust these values based on your problem
    )
    replay_observer = [replay_buffer.add_batch]"""

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=64,
        num_steps=3,
        single_deterministic_pass=False).prefetch(3)  # Ensure num_steps is 2
    
    iterator = iter(dataset)
  
    agent.train = common.function(agent.train)
    agent.train_step_counter.assign(0)

    avg_return = compute_avg_return(eval_py_env, random_policy, num_eval_episodes)
    returns = [avg_return]
    losses = []

    # Collect initial data
    print("Collecting initial data...")
    for _ in range(initial_collect_steps):
        time_step = train_py_env.reset()
        while not time_step.is_last():
            action_step = random_policy.action(time_step)
            next_time_step = train_py_env.step(action_step.action)
            traj = trajectory.from_transition(time_step, action_step, next_time_step)
            replay_buffer.add_batch(traj)
            time_step = next_time_step

    # Check the size of the replay buffer after initial collection
    initial_buffer_size = replay_buffer.num_frames().numpy()
    print(f"Replay buffer size after initial collection: {initial_buffer_size}")

    for _ in range(num_iterations):
        time_step = train_py_env.reset()
        while not time_step.is_last():
            action_step = agent.collect_policy.action(time_step)
            next_time_step = train_py_env.step(action_step.action)
            traj = trajectory.from_transition(time_step, action_step, next_time_step)
            replay_buffer.add_batch(traj)
            time_step = next_time_step

        experience, sample_info = next(iterator)
    
        # Ensure weights are of the correct type and shape
        weights = sample_info.probabilities
        #print(f"Weights before casting: {weights}")  # Debugging
        weights = tf.cast(weights, tf.float32)
        weights = tf.reshape(weights, [-1, 1])
        #print(f"Weights after casting: {weights}")  # Debugging

        # Perform a training step with the correct weights
        train_loss = agent.train(experience, weights=weights).loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            avg_return = compute_avg_return(eval_py_env, agent.policy, num_eval_episodes)
            losses.append(train_loss.numpy())
            print(f'step = {step}: Average Return = {avg_return}, Loss = {train_loss}')
            returns.append(avg_return)

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_py_env, agent.policy, num_eval_episodes)
            print(f'step = {step}: Average Return = {avg_return}')
            returns.append(avg_return)

    plot_metrics(returns, losses, log_interval)

# Call the training function
#train_rainbow_dqn(num_iterations=20000, log_interval=100, eval_interval=1000, num_eval_episodes=10, initial_collect_steps=1000)