import tensorflow as tf
import numpy as np
import random
from collections import deque

from Model import Q_network_model
from RLEnv.Cartpole import CartPoleEnvSetup,visualizeCartpole
from RLEnv.Cartpole import resetCartpole,CartPoleTransition,CartPoleReward,isTerminal


# Hyper Parameters for DQN
GAMMA = 0.9           # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
REPLAY_SIZE = 10000   # experience replay buffer size N
BATCH_SIZE = 32       # size of minibatch

class DQN():
  # DQN Agent
  def __init__(self, env):
    self.replay_memory = deque()
    self.time_step = 0
    self.epsilon = INITIAL_EPSILON
    self.state_dim = env.observation_space.shape[0]
    self.action_dim = env.action_space.n

    self.Q_network=Q_network_model(self.state_dim,self.action_dim)
    self.state_input=self.Q_network.state_input
    self.Q_value=self.Q_network.Q_value

    self.session = tf.InteractiveSession()
    self.session.run(tf.initialize_all_variables())

  def update_Q(self,state,action,reward,next_state,done):
    '''
    This function is used to update the Q network
    '''
    one_hot_action = np.zeros(self.action_dim)
    one_hot_action[action] = 1

    # store transition in D (replay memory)
    self.replay_memory.append((state, one_hot_action, reward, next_state, done))
    if len(self.replay_memory) > REPLAY_SIZE:
      self.replay_memory.popleft()
    if len(self.replay_memory) > BATCH_SIZE:
      self.time_step += 1
      self.Q_network.train_Q_network(self.replay_memory, BATCH_SIZE, GAMMA)

  def choose_egreedy_action(self,state):
    '''
    This function is used to choose the action a_t according to e_greedy strategy
    '''
    Q_value = self.Q_value.eval(feed_dict = {self.state_input:[state]})[0]
    self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000   # epsilon decay
    if random.random() <= self.epsilon:
      return random.randint(0,self.action_dim - 1)  # randomly select one
    else:
      return np.argmax(Q_value)

  def get_max_action(self,state):
    '''
    This function is for testing code
    We will get the action just accord to the neural network (with out exploration or update).
    '''
    return np.argmax(self.Q_value.eval(feed_dict = {self.state_input:[state]})[0])

# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = ['CartPole-v0','Acrobot-v1','Pendulum-v0','MountainCar-v0']
EPISODE = 10000 # Episode limitation
STEP = 400      # Step limitation in an episode
TEST = 10       # The number of experiment test every 100 episode

def main():
  env = CartPoleEnvSetup()      # initialize OpenAI Gym environment
  visualizeCp = visualizeCartpole()     # for visualization
  resetCp = resetCartpole()         # for state reset
  transitionCp = CartPoleTransition()  # transition function
  rewardCp = CartPoleReward()        # reward function
  isterminalCp = isTerminal()            # judge whether is terminal
  agent = DQN(env)              # initialize dqn agent

  # the outer for loop
  for episode in range(EPISODE):
    state = resetCp() # initialize task
    # the inner for loop
    for t in range(STEP):
      action = agent.choose_egreedy_action(state)         # e-greedy action for train
      next_state=transitionCp(state, action)
      done = isterminalCp(next_state)
      reward = rewardCp(done)
      agent.update_Q(state,action,reward,next_state,done) # renew the parameter according to the reward, observe the environment
      state = next_state
      if done:
        break

    # Test every 100 episodes
    if episode % 100 == 0:
      rewards = 0   # total reward
      for i in range(TEST):
        state = resetCp()
        for j in range(STEP):
          visualizeCp(state)
          action = agent.get_max_action(state)      # direct action for test
          next_state=transitionCp(state, action)
          done = isterminalCp(next_state)
          reward = rewardCp(done)
          state=next_state
          rewards += reward
          if done:
            break
      ave_reward = rewards/TEST
      print('episode: ',episode,'Evaluation Average Reward:',ave_reward)
      if ave_reward >= 200:
        visualizeCp.close()
        break

if __name__ == '__main__':
  main()

