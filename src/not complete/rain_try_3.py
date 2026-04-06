from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.environments import tf_py_environment
import constants
from rm_environment import ClusterEnv
from tf_agents.networks import network
from tf_agents.networks import encoding_network
from tf_agents.networks import q_network
from tf_agents.networks import categorical_q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
import csv
import logging
import numpy

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
    try:
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        print("Step collected.")
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

       # Print debug information for trajectory shapes
        def get_shapes(obj):
            shapes = {}
            for attr in dir(obj):
                if not attr.startswith('_'):
                    value = getattr(obj, attr)
                    if isinstance(value, (tf.Tensor, tf.Variable)):
                        shapes[attr] = value.shape
            return shapes

        print("Trajectory shapes:")
        print("Time step:", get_shapes(time_step))
        print("Action step:", get_shapes(action_step))
        print("Next time step:", get_shapes(next_time_step))


        buffer.add_batch(traj)
        print("Trajectory added to buffer.")
    except Exception as e:
        print(f"Error in collect_step: {e}")

def collect_data(env, policy, buffer, steps):
    for step in range(steps):
        print(f"Collecting step {step + 1}/{steps}")
        collect_step(env, policy, buffer)




class NoisyDense(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(NoisyDense, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        sigma_init = 0.5 / tf.sqrt(float(input_dim))
        self.w_mu = self.add_weight(
            name='w_mu', shape=(input_dim, self.units),
            initializer=tf.random_uniform_initializer(-0.1, 0.1))
        self.w_sigma = self.add_weight(
            name='w_sigma', shape=(input_dim, self.units),
            initializer=tf.constant_initializer(sigma_init.numpy()))
        self.b_mu = self.add_weight(
            name='b_mu', shape=(self.units,),
            initializer=tf.random_uniform_initializer(-0.1, 0.1))
        self.b_sigma = self.add_weight(
            name='b_sigma', shape=(self.units,),
            initializer=tf.constant_initializer(sigma_init.numpy()))
        self._input_shape = input_shape

    def call(self, inputs, training=False):
        if training:
            epsilon_w = tf.random.normal(self.w_mu.shape)
            epsilon_b = tf.random.normal(self.b_mu.shape)
            w = self.w_mu + self.w_sigma * epsilon_w
            b = self.b_mu + self.b_sigma * epsilon_b
        else:
            w = self.w_mu
            b = self.b_mu
        return tf.matmul(inputs, w) + b

class CustomCategoricalQNetwork(network.Network):
    def __init__(self, input_tensor_spec, action_spec, num_atoms, fc_layer_params, name='CustomCategoricalQNetwork'):
        super(CustomCategoricalQNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)
        self._action_spec = action_spec
        self._num_actions = action_spec.maximum - action_spec.minimum + 1
        self._num_atoms = num_atoms

        self._encoding_network = encoding_network.EncodingNetwork(
            input_tensor_spec,
            fc_layer_params=fc_layer_params,
            activation_fn=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode='fan_in', distribution='truncated_normal'))

        self._advantage_dense_layers = [NoisyDense(num_units) for num_units in fc_layer_params]
        self._value_dense_layers = [NoisyDense(num_units) for num_units in fc_layer_params]

        self._advantage = tf.keras.layers.Dense(
            self._num_actions * self._num_atoms,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.03, maxval=0.03),
            bias_initializer=tf.keras.initializers.Constant(-0.2),
            name='advantage')
        self._value = tf.keras.layers.Dense(
            self._num_atoms,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.03, maxval=0.03),
            bias_initializer=tf.keras.initializers.Constant(-0.2),
            name='value')

    @property
    def num_atoms(self):
        return self._num_atoms

    def call(self, observation, step_type=(), network_state=(), training=False):
        state, network_state = self._encoding_network(
            observation, step_type=step_type, network_state=network_state, training=training)

        advantage = state
        for layer in self._advantage_dense_layers:
            advantage = layer(advantage, training=training)
        advantage = self._advantage(advantage)
        advantage = tf.reshape(advantage, [-1, self._num_actions, self._num_atoms])

        value = state
        for layer in self._value_dense_layers:
            value = layer(value, training=training)
        value = self._value(value)
        value = tf.reshape(value, [-1, 1, self._num_atoms])

        q_logits = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
        q_values = tf.nn.softmax(q_logits, axis=-1)
        return q_values, network_state



class TFPrioritizedReplayBuffer(tf_uniform_replay_buffer.TFUniformReplayBuffer):
    def __init__(self, data_spec, batch_size, max_length, alpha=0.6):
        super(TFPrioritizedReplayBuffer, self).__init__(
            data_spec=data_spec,
            batch_size=batch_size, 
            max_length=max_length,
        )

        self._index = 0
        self.alpha = alpha
        self._priorities = tf.Variable(
            tf.zeros([batch_size, max_length], dtype=tf.float32),
            trainable=False, name='priorities')
        
        # initialize the data attribute
        self.data = tf.nest.map_structure(
            lambda spec: tf.TensorArray(
                dtype=spec.dtype,
                size=max_length,
                dynamic_size=False,
                element_shape=spec.shape
            ),
            data_spec
        )



    def add_batch(self, items):
        super(TFPrioritizedReplayBuffer, self).add_batch(items)
        print("Batch added to replay buffer.")
        self._priorities.assign(tf.tensor_scatter_nd_update(
            self._priorities, [[0, self._index]], tf.ones([self._batch_size], dtype=tf.float32)))
        #self._index += 1


        #flatten the items and ensure they match the expected shapes
        flattened_items = tf.nest.flatten(items)
        flattened_data = tf.nest.flatten(self.data)

       # Update the data attribute with new items
        for i, item in enumerate(flattened_items):
            expected_shape = flattened_data[i].element_shape
            print(f"Item {i} shape before reshape: {item.shape}, expected shape: {expected_shape}")
            if item.shape != expected_shape:
                item = tf.reshape(item, expected_shape)
                print(f"Item {i} shape after reshape: {item.shape}")
            flattened_data[i] = flattened_data[i].write(self._index, item)
        
        #reconstruct the nested structure of self.data
        self.data = tf.nest.pack_sequence_as(self.data, flattened_data)


        # Increment the index and wrap around if necessary
        self._index = (self._index + 1) % self._max_length
        print(f"Index after adding batch: {self._index}")

    def get_next(self, sample_batch_size=None, num_steps=None, time_stacked=True):
        print("Getting next batch from replay buffer...")
        if sample_batch_size is None:
            sample_batch_size = self._batch_size
        if num_steps is None:
            num_steps = 1

        print("Calculating priorities...")
        priorities = tf.pow(self._priorities[:self._batch_size, :self._max_length], self.alpha)
        tf.print("Priorities shape:", tf.shape(priorities), "values:", priorities)
    
        print("Calculating total priority...")
        total_priority = tf.reduce_sum(priorities)
        tf.print("Total priority:", total_priority)
    
        print("Calculating sampling probabilities...")
        probs = priorities / total_priority
        tf.print("Sampling probabilities shape:", tf.shape(probs), "values:", probs)

        print("Sampling indices...")
        indices = tf.random.categorical(tf.math.log(probs), sample_batch_size)
        tf.print("Sampled indices shape:", tf.shape(indices), "values:", indices)
    
        indices = tf.reshape(indices, [sample_batch_size])
        tf.print("Reshaped indices:", indices)

        # ensure the indices are within the bounds
        indices = tf.clip_by_value(indices, 0, self._max_length - 1)
        tf.print("Clipped indices:", indices)

        print("Gathering items from replay buffer...")
        
        """items = tf.nest.map_structure(lambda x: tf.gather(x.stack(), indices), self.data)
        print("Next batch obtained.")
        tf.print("Items shape:", {k: tf.shape(v) for k, v in items.items()})

        """
        try:
            items = tf.nest.map_structure(lambda x: tf.gather(x.stack(), indices), self.data)
            print("Next batch obtained.")
            tf.print("Items shape:", {k: tf.shape(v) for k, v in items.items()})
        except Exception as e:
            tf.print("Error in gathering items:", e)
            return None, None
        
        return items, indices
        


    def update_priorities(self, indices, priorities):
        priorities = tf.pow(priorities, self.alpha)
        self._priorities.assign(tf.tensor_scatter_nd_update(self._priorities, indices, priorities))




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


logging.basicConfig(level=logging.INFO)  # Set logging level to INFO
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
    """Train the DQN agent."""
    print("Training started...")
    file = open(constants.root + '/output/avg_returns_' + constants.algo + '_beta_' + str(constants.beta) + '.csv', 'w', newline='')
    avg_return_writer = csv.writer(file, delimiter=',')
    avg_return_writer.writerow(["Iteration", "AVG_Return", "AVG_Reward"])

    train_py_env = ClusterEnv()
    eval_py_env = train_py_env
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    
    # create a custom categoricalqnetwork with deuling architecture and noisy layers
    q_net = CustomCategoricalQNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        num_atoms=51, # the number of atoms for the distributional q-learning
        fc_layer_params=fc_layer_params)

    # Use Adam optimizer
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_step_counter = tf.compat.v1.Variable(0)


    
    # Use CategoricalDqnAgent for Distributional Q-Learning
    agent = categorical_dqn_agent.CategoricalDqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        categorical_q_network=q_net,
        optimizer=optimizer,
        min_q_value=-10,
        max_q_value=10,
        n_step_update=3,  # Multi-Step Returns
        epsilon_greedy=0.1,
        target_update_period=2000,
        gamma=0.99,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter,
        name='CategoricalDqnAgent')

    agent.initialize()

    # Use Prioritized Replay Buffer
    replay_buffer = TFPrioritizedReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length)
    

    print("Starting initial data collection...")
    try:
        collect_data(train_env, agent.collect_policy, replay_buffer, steps=initial_collect_steps)
    except Exception as e:
        logging.error(f"Error during initial data collection: {e}")
        return

    print("Initial data collection completed.")

    print("Creating dataset and iterator...")
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2
    ).prefetch(3)
    iterator = iter(dataset)
    print("Dataset and iterator created successfully.")


    agent.train = common.function(agent.train)
    agent.train_step_counter.assign(0)
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    avg_reward = compute_avg_reward(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]
    rewards = [avg_reward]


    print("Starting training loop...")
    for iteration in range(num_iterations):
        print(f"Starting iteration {iteration + 1}/{num_iterations}...")
        for _ in range(collect_steps_per_iteration):
            collect_step(train_env, agent.collect_policy, replay_buffer)
        print(f"Iteration {iteration + 1}: Data collection completed.")
        
        try:
            experience, unused_info = next(iterator)
        except StopIteration:
            logging.error("Dataset iterator stopped unexpectedly.")
            break
        
        print(f"Iteration {iteration + 1}: Data sampled from buffer.")
        

        train_loss = agent.train(experience).loss
        step = agent.train_step_counter.numpy()
        print(f"Iteration {iteration + 1}: Agent trained. Loss = {train_loss}, Step = {step}")


        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            avg_reward = compute_avg_reward(eval_env, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}, Average Reward = {2}'.format(step, avg_return, avg_reward))
            avg_return_writer.writerow([step, avg_return, avg_reward])
            returns.append(avg_return)
            rewards.append(avg_reward)
        
    print("Training completed.")

    iterations = range(0, num_iterations + 1, eval_interval)
    plt.plot(iterations, returns, label='Average Return')
    plt.plot(iterations, rewards, label='Average Reward')
    plt.ylabel('Average Return / Reward')
    plt.xlabel('Iterations')
    plt.legend()
    plt.show()