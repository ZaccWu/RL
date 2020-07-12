import tensorflow as tf
import numpy as np
import random
from collections import deque

class dqnModel:
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

  def QNetworkStructure(self):
    # input layer
    self.stateInput = tf.placeholder("float", [None, self.stateDim])
    hWidth = self.HIDDEN_LAYER_WIDTH

    # network weights
    with tf.variable_scope('current_net'):
        W1 = self.WeightVariable([self.stateDim, hWidth])
        b1 = self.BiasVariable([hWidth])
        W2 = self.WeightVariable([hWidth, self.actionDim])
        b2 = self.BiasVariable([self.actionDim])
        # hidden layers
        hiddenLayer = tf.nn.relu(tf.matmul(self.stateInput, W1) + b1)
        # Q Value layer
        self.QValue = tf.matmul(hiddenLayer, W2) + b2

    with tf.variable_scope('target_net'):
        W1t = self.WeightVariable([self.stateDim, hWidth])
        b1t = self.BiasVariable([hWidth])
        W2t = self.WeightVariable([hWidth, self.actionDim])
        b2t = self.BiasVariable([self.actionDim])
        # hidden layers
        hiddenLayerTarget = tf.nn.relu(tf.matmul(self.stateInput, W1t) + b1t)
        # Q Value layer
        self.targetQValue = tf.matmul(hiddenLayerTarget, W2t) + b2t

    TargetNetParam = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
    EvalNetParam = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='current_net')

    with tf.variable_scope('soft_replacement'):
        self.targetReplaceOp = [tf.assign(t, e) for t, e in zip(TargetNetParam, EvalNetParam)] # Here may have an issue

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
