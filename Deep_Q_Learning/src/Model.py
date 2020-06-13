import tensorflow as tf
import numpy as np
import random


def Q_network_structure(state_dim,action_dim):
  W1 = tf.Variable(tf.truncated_normal(shape=[state_dim, 15]))
  b1 = tf.Variable(tf.constant(0.01, shape=[15]))
  W2 = tf.Variable(tf.truncated_normal(shape=[15, action_dim]))
  b2 = tf.Variable(tf.constant(0.01, shape=[action_dim]))

  state_input = tf.placeholder("float", [None, state_dim])  # input layer
  h_layer = tf.nn.relu(tf.matmul(state_input, W1) + b1)  # hidden layers
  Q_value = tf.matmul(h_layer, W2) + b2  # Q Value layer
  return state_input,Q_value

def sample_minibatch(replay_memory, BATCH_SIZE):
  Batch_Data = random.sample(replay_memory, BATCH_SIZE)
  state_batch = [d[0] for d in Batch_Data]
  action_batch = [d[1] for d in Batch_Data]
  reward_batch = [d[2] for d in Batch_Data]
  next_state_batch = [d[3] for d in Batch_Data]
  done_batch = [d[4] for d in Batch_Data]
  return state_batch, action_batch, reward_batch, next_state_batch, done_batch


class Q_network_model:
  def __init__(self,state_dim,action_dim):
    self.state_dim=state_dim
    self.action_dim=action_dim
    self.sample_minibatch=sample_minibatch
    self.Q_network_structure=Q_network_structure
    self.state_input, self.Q_value = Q_network_structure(self.state_dim, self.action_dim)
    #self.Q_network_structure()
    self.create_training_method()

  def __call__(self,replay_memory, BATCH_SIZE, GAMMA):
    # sample random minibatch of transitions from D

    state_batch, action_batch, reward_batch, next_state_batch, done_batch=self.sample_minibatch(replay_memory,BATCH_SIZE)
    # calculate y_value
    y_value_batch = []
    Q_value_batch = self.Q_value.eval(feed_dict={self.state_input:next_state_batch})
    for i in range(0,BATCH_SIZE):
      done = done_batch[i]
      if not done:
        y_value_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))
      else:
        y_value_batch.append(reward_batch[i])

    #action_input,y_input,optimizer=create_training_method(self.state_dim,self.Q_value)
    self.optimizer.run(feed_dict={
      self.y_input:y_value_batch,
      self.action_input:action_batch,
      self.state_input:state_batch})




  def create_training_method(self):
    self.action_input = tf.placeholder("float",[None,self.action_dim]) # one hot presentation
    self.y_input = tf.placeholder("float",[None])
    Q_action = tf.reduce_sum(tf.multiply(self.Q_value,self.action_input),reduction_indices = 1)
    loss = tf.reduce_mean(tf.square(self.y_input - Q_action))
    self.optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)


'''
  def Q_network_structure(self):
    W1 = tf.Variable(tf.truncated_normal(shape=[self.state_dim, 15]))
    b1 = tf.Variable(tf.constant(0.01, shape=[15]))
    W2 = tf.Variable(tf.truncated_normal(shape=[15, self.action_dim]))
    b2 = tf.Variable(tf.constant(0.01, shape=[self.action_dim]))

    self.state_input = tf.placeholder("float", [None, self.state_dim])  # input layer
    h_layer = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)  # hidden layers
    self.Q_value = tf.matmul(h_layer, W2) + b2  # Q Value layer
'''




