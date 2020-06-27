import tensorflow as tf
import numpy as np
import random
from collections import deque

def NNstructure(state_dim, action_dim):
  W1 = tf.Variable(tf.truncated_normal(shape=[state_dim, 15]))
  b1 = tf.Variable(tf.constant(0.01, shape=[15]))
  W2 = tf.Variable(tf.truncated_normal(shape=[15, action_dim]))
  b2 = tf.Variable(tf.constant(0.01, shape=[action_dim]))

  state_input = tf.placeholder("float", [None, state_dim])  # input layer
  h_layer = tf.nn.relu(tf.matmul(state_input, W1) + b1)     # hidden layers
  Q_value = tf.matmul(h_layer, W2) + b2                     # Q Value layer
  return state_input,Q_value

def SampleMinibatch(replay_memory, BATCH_SIZE):
  Batch_Data = random.sample(replay_memory, BATCH_SIZE)
  state_batch = [d[0] for d in Batch_Data]
  action_batch = [d[1] for d in Batch_Data]
  reward_batch = [d[2] for d in Batch_Data]
  next_state_batch = [d[3] for d in Batch_Data]
  done_batch = [d[4] for d in Batch_Data]
  return state_batch, action_batch, reward_batch, next_state_batch, done_batch

def TrainingNetwork(action_dim, Q_value, lr):
  action_input = tf.placeholder("float",[None,action_dim]) # one hot presentation
  y_input = tf.placeholder("float",[None])
  Q_action = tf.reduce_sum(tf.multiply(Q_value,action_input),reduction_indices = 1)
  loss = tf.reduce_mean(tf.square(y_input - Q_action))
  optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
  return action_input,y_input,optimizer


class QNetworkModel:
  def __init__(self,state_dim,action_dim,param_set):
    self.action_dim=action_dim
    self.lr=param_set['lr']
    self.sample_minibatch=SampleMinibatch
    self.Q_network_structure=NNstructure
    self.state_input, self.Q_value = NNstructure(state_dim, action_dim)
    self.action_input,self.y_input,self.optimizer=TrainingNetwork(action_dim, self.Q_value, self.lr)

    self.initial_epsilon = param_set['initial_epsilon']
    self.final_epsilon = param_set['final_epsilon']
    self.gamma = param_set['gamma']
    self.REPLAY_SIZE = param_set['REPLAY_SIZE']
    self.BATCH_SIZE = param_set['BATCH_SIZE']

    self.replay_memory = deque()
    self.time_step = 0
    self.epsilon = self.initial_epsilon
    self.session = tf.InteractiveSession()
    self.session.run(tf.initialize_all_variables())

  def UpdateNetwork(self, state, action, reward, next_state, done):
    one_hot_action = np.zeros(self.action_dim)
    one_hot_action[action] = 1

    # store transition in D (replay memory)
    self.replay_memory.append((state, one_hot_action, reward, next_state, done))
    if len(self.replay_memory) > self.REPLAY_SIZE:
      self.replay_memory.popleft()
    if len(self.replay_memory) > self.BATCH_SIZE:
      self.time_step += 1
      state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.sample_minibatch(self.replay_memory,self.BATCH_SIZE)
      # calculate y_value
      y_value_batch = []
      Q_value_batch = self.Q_value.eval(feed_dict={self.state_input: next_state_batch})
      for i in range(0, self.BATCH_SIZE):
        done = done_batch[i]
        if not done:
          y_value_batch.append(reward_batch[i] + self.gamma * np.max(Q_value_batch[i]))
        else:
          y_value_batch.append(reward_batch[i])

      # action_input,y_input,optimizer=create_training_method(self.state_dim,self.Q_value)
      self.optimizer.run(feed_dict={
        self.y_input: y_value_batch,
        self.action_input: action_batch,
        self.state_input: state_batch})

  def EgreedyAction(self, state):
    Q_value = self.Q_value.eval(feed_dict = {self.state_input:[state]})[0]
    self.epsilon -= (self.initial_epsilon - self.final_epsilon) / 10000   # epsilon decay
    if random.random() <= self.epsilon:
      return random.randint(0,self.action_dim - 1)  # randomly select one
    else:
      return np.argmax(Q_value)

  def GetMaxAction(self, state):
    return np.argmax(self.Q_value.eval(feed_dict = {self.state_input:[state]})[0])
