from src.Qnetwork import QNetworkModel
from Env.Cartpole import CartPoleEnvSetup,visualizeCartpole
from Env.Cartpole import resetCartpole,CartPoleTransition,CartPoleReward,isTerminal

param_set = {
  'initial_epsilon': 0.5,
  'final_epsilon': 0.01,
  'gamma': 0.9,
  'REPLAY_SIZE': 10000,
  'BATCH_SIZE': 32,
  'lr': 0.001,
}
EPISODE = 10000  # Episode limitation
STEP = 400  # Step limitation in an episode
TEST = 10  # The number of experiment test every 100 episode

def main():
  env = CartPoleEnvSetup()      # initialize OpenAI Gym environment
  visualizeCp = visualizeCartpole()     # for visualization
  resetCp = resetCartpole()         # for state reset
  transitionCp = CartPoleTransition()  # transition function
  rewardCp = CartPoleReward()        # reward function
  isterminalCp = isTerminal()            # judge whether is terminal

  state_dim = env.observation_space.shape[0]
  action_dim = env.action_space.n
  dqn = QNetworkModel(state_dim, action_dim, param_set)

  # the outer for loop
  for episode in range(EPISODE):
    state = resetCp() # initialize task
    # the inner for loop
    for t in range(STEP):
      action = dqn.EgreedyAction(state)         # e-greedy action for train
      next_state=transitionCp(state, action)
      done = isterminalCp(next_state)
      reward = rewardCp(done)
      dqn.UpdateNetwork(state, action, reward, next_state, done) # renew the parameter according to the reward, observe the environment
      state = next_state
      if done:
        break

    # Test every 100 episodes (for visualization)
    if episode % 100 == 0:
      rewards = 0   # total reward
      for i in range(TEST):
        state = resetCp()
        for j in range(STEP):
          visualizeCp(state)
          action = dqn.GetMaxAction(state)      # direct action for test
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

