from src.Model import Q_network_model
from Env.Cartpole import CartPoleEnvSetup,visualizeCartpole
from Env.Cartpole import resetCartpole,CartPoleTransition,CartPoleReward,isTerminal

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

  state_dim = env.observation_space.shape[0]
  action_dim = env.action_space.n
  dqn = Q_network_model(state_dim,action_dim)

  # the outer for loop
  for episode in range(EPISODE):
    state = resetCp() # initialize task
    # the inner for loop
    for t in range(STEP):
      action = dqn.choose_egreedy_action(state)         # e-greedy action for train
      next_state=transitionCp(state, action)
      done = isterminalCp(next_state)
      reward = rewardCp(done)
      dqn.update_Q_network(state, action, reward, next_state, done) # renew the parameter according to the reward, observe the environment
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
          action = dqn.get_max_action(state)      # direct action for test
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

