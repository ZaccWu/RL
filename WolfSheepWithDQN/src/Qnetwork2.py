import tensorflow as tf
import numpy as np
import random
from collections import deque

class dqnModel2:
  def __init__(self, stateDim, actionDim, paramSet):
    # init experience replay
    self.replayBuffer = deque()

    # init some parameters
    self.timeStep = 0
    self.stateDim = stateDim
    self.actionDim = actionDim

    # hyper-parameters
    self.BATCH_SIZE = paramSet['BATCH_SIZE']
    self.REPLAY_SIZE = paramSet['REPLAY_SIZE']
    self.REPLACE_TARGET_FREQ = paramSet['REPLACE_TARGET_FREQ']
    self.GAMMA = paramSet['GAMMA']
    self.INITIAL_EPSILON = paramSet['INITIAL_EPSILON']
    self.FINAL_EPSILON = paramSet['FINAL_EPSILON']
    self.HIDDEN_LAYER_WIDTH = paramSet['HIDDEN_LAYER_WIDTH']
    self.LR = paramSet['LR']

    # parameters for local usage
    self.epsilon = self.INITIAL_EPSILON

    # function initialization
    self.QNetworkStructure()
    self.CreateTrainingMethod()

    # initialize session
    self.session = tf.InteractiveSession()
    self.session.run(tf.global_variables_initializer())
    self.targetReplaceOp = None

  def QNetworkStructure(self):
    # input layer
    self.stateInput = tf.placeholder("float", [None, self.stateDim])
    hWidth = self.HIDDEN_LAYER_WIDTH

    # network weights
    with tf.variable_scope('current_net'):
        W1e = self.WeightVariable([self.stateDim, hWidth])
        b1e = self.BiasVariable([hWidth])
        W2e = self.WeightVariable([hWidth,hWidth])
        b2e = self.BiasVariable([hWidth])
        W3e = self.WeightVariable([hWidth, self.actionDim])
        b3e = self.BiasVariable([self.actionDim])
        # hidden layers
        hiddenLayer1 = tf.nn.relu(tf.matmul(self.stateInput, W1e) + b1e)
        hiddenLayer2 = tf.nn.relu(tf.matmul(hiddenLayer1, W2e) + b2e)
        # Q Value layer
        self.QValue = tf.matmul(hiddenLayer2, W3e) + b3e

    with tf.variable_scope('target_net'):
        W1t = self.WeightVariable([self.stateDim, hWidth])
        b1t = self.BiasVariable([hWidth])
        W2t = self.WeightVariable([hWidth,hWidth])
        b2t = self.BiasVariable([hWidth])
        W3t = self.WeightVariable([hWidth, self.actionDim])
        b3t = self.BiasVariable([self.actionDim])
        # hidden layers
        hiddenLayerTarget1 = tf.nn.relu(tf.matmul(self.stateInput, W1t) + b1t)
        hiddenLayerTarget2 = tf.nn.relu(tf.matmul(hiddenLayerTarget1, W2t) + b2t)
        # Q Value layer
        self.targetQValue = tf.matmul(hiddenLayerTarget2, W3t) + b3t

    EvalNetParam = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='current_net')
    TargetNetParam = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')

    with tf.variable_scope('soft_replacement'):
        self.targetReplaceOp = [tf.assign(t, e) for t, e in zip(TargetNetParam[:4], EvalNetParam[:4])]

  def CreateTrainingMethod(self):
    self.action_input = tf.placeholder("float", [None, self.actionDim]) # one hot presentation
    self.yInput = tf.placeholder("float", [None])
    qAction = tf.reduce_sum(tf.multiply(self.QValue, self.action_input), reduction_indices = 1)
    self.cost = tf.reduce_mean(tf.square(self.yInput - qAction))
    self.optimizer = tf.train.AdamOptimizer(self.LR).minimize(self.cost)

  def Update(self, state, action, reward, nextState, done):
    OneHotAction = np.zeros(self.actionDim)
    OneHotAction[action] = 1
    self.replayBuffer.append((state, OneHotAction, reward, nextState, done))
    if len(self.replayBuffer) > self.REPLAY_SIZE:
      self.replayBuffer.popleft()
    if len(self.replayBuffer) > self.BATCH_SIZE:
      self.SampleAndTrain()

  def SampleAndTrain(self):
    self.timeStep += 1
    # Sample MiniBatch from replay memory
    minibatch = random.sample(self.replayBuffer, self.BATCH_SIZE)
    stateBatch = [data[0] for data in minibatch]
    actionBatch = [data[1] for data in minibatch]
    rewardBatch = [data[2] for data in minibatch]
    nextStateBatch = [data[3] for data in minibatch]

    # Calculate y
    yBatch = []
    QValueBatch = self.targetQValue.eval(feed_dict={self.stateInput:nextStateBatch})
    for i in range(0,self.BATCH_SIZE):
      done = minibatch[i][4]
      if done:
        yBatch.append(rewardBatch[i])
      else :
        yBatch.append(rewardBatch[i] + self.GAMMA * np.max(QValueBatch[i]))

    self.optimizer.run(feed_dict={
      self.yInput:yBatch,
      self.action_input:actionBatch,
      self.stateInput:stateBatch
      })

  def UpdateTargetQNetwork(self, episode):
    if episode % self.REPLACE_TARGET_FREQ == 0:
        self.session.run(self.targetReplaceOp)

  def WeightVariable(self, shape):
    initial = tf.truncated_normal(shape)
    return tf.Variable(initial)

  def BiasVariable(self, shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

  def egreedyAction(self, state):
    QValue = self.QValue.eval(feed_dict = {self.stateInput:[state]})[0]
    # epsilon decay
    self.epsilon -= (self.INITIAL_EPSILON - self.FINAL_EPSILON) / 10000
    self.INITIAL_EPSILON = self.epsilon
    if random.random() <= self.epsilon:
        return random.randint(0, self.actionDim - 1)
    else:
        return np.argmax(QValue)

  def GetMaxAction(self,state):
    MaxValueAction=np.argmax(self.QValue.eval(feed_dict = {self.stateInput:[state]})[0])
    return MaxValueAction