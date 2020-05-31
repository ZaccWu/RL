import unittest
from ddt import ddt,data,unpack
from RLEnv.Cartpole import *
from RLEnv.gymCartpole import *

import sys
sys.path.append("..")

@ddt
class TestEnv(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    @data(
        ([0.3, 1.2, 0.2, 0.1],0),
        ([0.3, 1.2, -0.2, -0.1], 1),
        ([0.4, 1.3, -0.2, 1.5], 0),
        ([-0.1, 0.4, 0.1, 0.6], 1)
    )
    @unpack
    def testTransition(self,state,action):
        originenv=CartPoleEnv()
        originenv.state=state
        next_stateGym,rewardGym,doneGym,_=originenv.step(action)

        transitionCp=CartPoleTransition()
        next_state=transitionCp(state,action)

        self.assertEqual(tuple(next_stateGym),tuple(next_state))


    @data(
        ([0.3, 1.2, 0.2, 0.1],0),
        ([0.3, 1.2, -0.2, -0.1], 1),
        ([0.4, 1.3, -0.2, 1.5], 0),
        ([-0.1, 0.4, 0.1, 0.6], 1)
    )
    @unpack
    def testIsterminal(self,state,action):
        originenv=CartPoleEnv()
        originenv.state=state
        next_stateGym,rewardGym,doneGym,_=originenv.step(action)

        transitionCp=CartPoleTransition()
        next_state=transitionCp(state,action)
        isterminalCp=isTerminal()
        done=isterminalCp(next_state)

        self.assertEqual(doneGym,done)


    @data(
        ([0.3, 1.2, 0.2, 0.1],0),
        ([0.3, 1.2, -0.2, -0.1], 1),
        ([0.4, 1.3, -0.2, 1.5], 0),
        ([-0.1, 0.4, 0.1, 0.6], 1)
    )
    @unpack
    def testReward(self,state,action):
        originenv=CartPoleEnv()
        originenv.state=state
        next_stateGym,rewardGym,doneGym,_=originenv.step(action)

        transitionCp=CartPoleTransition()
        next_state=transitionCp(state,action)
        isterminalCp=isTerminal()
        done=isterminalCp(next_state)
        rewardCp=CartPoleReward()
        reward=rewardCp(done)

        self.assertEqual(rewardGym,reward)



if __name__ == '__main__':
    unittest.main()