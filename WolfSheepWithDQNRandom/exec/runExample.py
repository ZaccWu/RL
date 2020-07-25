import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import random
import numpy as np
import pickle
from collections import OrderedDict
import pandas as pd
from matplotlib import pyplot as plt
import itertools as it
import math
import pygame as pg
from pygame.color import THECOLORS

from src.visualization.drawDemo import DrawBackground, DrawState, VisualizeTraj, InterpolateStateForVisualization
from src.analyticGeometryFunctions import transCartesianToPolar, transPolarToCartesian
from src.MDPChasing.env import IsTerminal, IsLegalInitPositions, ResetState, PrepareSheepVelocity, PrepareWolfVelocity, PrepareDistractorVelocity, \
PrepareAllAgentsVelocities, StayInBoundaryByReflectVelocity, TransitWithInterpolation
from src.MDPChasing.reward import RewardFunctionTerminalPenalty
from src.MDPChasing.policies import RandomPolicy
from src.chooseFromDistribution import sampleFromDistribution, maxFromDistribution
from src.trajectory import ForwardOneStep, SampleTrajectory
from src.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.Qnetwork import dqnModel

def flatten(state):
    return state.flatten()

def composeFowardOneTimeStepWithRandomSubtlety(numOfAgent):
    # experiment parameter for env
    numMDPTimeStepPerSecond = 5  # change direction every 200ms
    distanceToVisualDegreeRatio = 20
    minSheepSpeed = int(17.4 * distanceToVisualDegreeRatio / numMDPTimeStepPerSecond)
    maxSheepSpeed = int(23.2 * distanceToVisualDegreeRatio / numMDPTimeStepPerSecond)
    warmUpTimeSteps = 10 * numMDPTimeStepPerSecond  # 10s to warm up
    prepareSheepVelocity = PrepareSheepVelocity(minSheepSpeed, maxSheepSpeed, warmUpTimeSteps)

    minWolfSpeed = int(8.7 * distanceToVisualDegreeRatio / numMDPTimeStepPerSecond)
    maxWolfSpeed = int(14.5 * distanceToVisualDegreeRatio / numMDPTimeStepPerSecond)
    wolfSubtleties = [500, 11, 3.3, 1.83, 0.92, 0.31, 0.001]  # 0, 30, 60, .. 180
    initWolfSubtlety = np.random.choice(wolfSubtleties)
    prepareWolfVelocity = PrepareWolfVelocity(minWolfSpeed, maxWolfSpeed, warmUpTimeSteps, initWolfSubtlety,
                                              transCartesianToPolar, transPolarToCartesian)

    minDistractorSpeed = int(8.7 * distanceToVisualDegreeRatio / numMDPTimeStepPerSecond)
    maxDistractorSpeed = int(14.5 * distanceToVisualDegreeRatio / numMDPTimeStepPerSecond)
    prepareDistractorVelocity = PrepareDistractorVelocity(minDistractorSpeed, maxDistractorSpeed, warmUpTimeSteps,
                                                          transCartesianToPolar, transPolarToCartesian)

    sheepId = 0
    wolfId = 1
    distractorsIds = list(range(2, numOfAgent))
    prepareAllAgentsVelocities = PrepareAllAgentsVelocities(sheepId, wolfId, distractorsIds, prepareSheepVelocity,
                                                            prepareWolfVelocity, prepareDistractorVelocity)

    xBoundary = [0, 640]
    yBoundary = [0, 480]
    stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)

    killzoneRadius = 2.5 * distanceToVisualDegreeRatio
    isTerminal = IsTerminal(sheepId, wolfId, killzoneRadius)

    numFramePerSecond = 30  # visual display fps
    numFramesToInterpolate = int(numFramePerSecond / numMDPTimeStepPerSecond - 1)  # interpolate each MDP timestep to multiple frames; check terminal for each frame
    transitFunction = TransitWithInterpolation(initWolfSubtlety, numFramesToInterpolate, prepareAllAgentsVelocities,
                                               stayInBoundaryByReflectVelocity, isTerminal)
    aliveBonus = 0.01
    deathPenalty = -1
    rewardFunction = RewardFunctionTerminalPenalty(aliveBonus, deathPenalty, isTerminal)
    forwardOneStep = ForwardOneStep(transitFunction, rewardFunction)

    return transitFunction,rewardFunction,forwardOneStep

def initializeEnvironment(numOfAgent):
    sheepId = 0
    wolfId = 1
    distractorsIds = list(range(2, numOfAgent))
    distanceToVisualDegreeRatio = 20
    minInitSheepWolfDistance = 9 * distanceToVisualDegreeRatio
    minInitSheepDistractorDistance = 2.5 * distanceToVisualDegreeRatio  # no distractor in killzone when init
    isLegalInitPositions = IsLegalInitPositions(sheepId, wolfId, distractorsIds, minInitSheepWolfDistance,
                                                minInitSheepDistractorDistance)
    xBoundary = [0, 640]
    yBoundary = [0, 480]
    resetState = ResetState(xBoundary, yBoundary, numOfAgent, isLegalInitPositions, transPolarToCartesian)

    killzoneRadius = 2.5 * distanceToVisualDegreeRatio
    isTerminal = IsTerminal(sheepId, wolfId, killzoneRadius)

    return resetState,isTerminal


class SampleTrajectoriesForCoditions:
    def __init__(self, numTrajectories, composeFowardOneTimeStepWithRandomSubtlety,initializeEnvironment,parameters):
        self.numTrajectories = numTrajectories
        self.composeFowardOneTimeStepWithRandomSubtlety = composeFowardOneTimeStepWithRandomSubtlety
        self.initializeEnvironment = initializeEnvironment


        self.numOfAgent = parameters['numOfAgent']
        self.transitFunction, self.rewardFunction, self.forwardOneStep = self.composeFowardOneTimeStepWithRandomSubtlety(
            self.numOfAgent)
        self.resetState, self.isTerminal= self.initializeEnvironment(self.numOfAgent)

    def __call__(self,sampleAction):
        self.sampleAction = sampleAction
        numMDPTimeStepPerSecond = 5
        maxRunningSteps = 25 * numMDPTimeStepPerSecond

        trajectories = []
        sampleTrajecoty = SampleTrajectory(maxRunningSteps, self.isTerminal, self.resetState, self.forwardOneStep)
        # randomPolicy = RandomPolicy(actionSpace)
        # sampleAction = lambda state: sampleFromDistribution(randomPolicy(state)) # random policy
        # sampleAction = lambda state: actionSpace[dqn.GetMaxAction(flatten(state))]  # dqn
        for trajectoryId in range(self.numTrajectories):
            trajectory = sampleTrajecoty(self.sampleAction)
            trajectories.append(trajectory)
        return trajectories

def trainTask(dqn,actionSpace):
    EPISODE = 500 # Episode limitation
    STEP = 125  # Step limitation in an episode
    trainPlot = 1 # plot while training
    testPlot = 200 # plot while testing

    numTrajectories = 5
    param={'numOfAgent': 25}
    sampleTrajectoriesForCoditions = SampleTrajectoriesForCoditions(numTrajectories,
                                                                    composeFowardOneTimeStepWithRandomSubtlety,
                                                                    initializeEnvironment, param)
    results=[]
    meanRewards = 0

    for episode in range(EPISODE):
        sampleTrajectoriesForCoditions = SampleTrajectoriesForCoditions(numTrajectories,
                                                                        composeFowardOneTimeStepWithRandomSubtlety,
                                                                        initializeEnvironment, param)
        transitFunction = sampleTrajectoriesForCoditions.transitFunction
        rewardFunction = sampleTrajectoriesForCoditions.rewardFunction
        isTerminal = sampleTrajectoriesForCoditions.isTerminal
        resetState = sampleTrajectoriesForCoditions.resetState

        epiRewards = 0
        state = resetState()
        for t in range(STEP):
            actionId = dqn.egreedyAction(flatten(state))
            nextState = transitFunction(state, actionSpace[actionId])
            done = isTerminal(nextState)
            reward = rewardFunction(state, actionSpace[actionId], nextState)
            dqn.Update(flatten(state), actionId, reward, flatten(nextState), done)
            state = nextState
            meanRewards += 1
            epiRewards += 1
            if done:
                break
        # plot when training
        if episode % trainPlot == 0 and episode != 0:
            results.append(meanRewards / trainPlot)
            print("mean rewards:{},episode:{}".format(meanRewards / trainPlot, episode))
            meanRewards = 0



        # plot when testing
        if episode == EPISODE-1:
            count=0
            for i in range(testPlot):
                # all functions we need
                sampleTrajectoriesForCoditions = SampleTrajectoriesForCoditions(numTrajectories,
                                                                                composeFowardOneTimeStepWithRandomSubtlety,
                                                                                initializeEnvironment, param)
                transitFunction = sampleTrajectoriesForCoditions.transitFunction
                rewardFunction = sampleTrajectoriesForCoditions.rewardFunction
                isTerminal = sampleTrajectoriesForCoditions.isTerminal
                resetState = sampleTrajectoriesForCoditions.resetState
                state = resetState()
                rewards=0
                for j in range(STEP):
                    actionId = dqn.GetMaxAction(flatten(state))
                    nextState = transitFunction(state, actionSpace[actionId])
                    done = isTerminal(nextState)
                    reward = rewardFunction(state, actionSpace[actionId], nextState)
                    state = nextState
                    rewards += reward
                    if done:
                        break
                if rewards>=1.25:
                    count+=1
            print(count)
            print(results)

    # for visualization
    visualize = True
    if visualize:
        sampleAction = lambda state: actionSpace[dqn.GetMaxAction(flatten(state))]
        trajectoriesMultipleConditions = [sampleTrajectoriesForCoditions(sampleAction)]
        visualConditionIndex = 0
        trajectoriesToVisualize = trajectoriesMultipleConditions[visualConditionIndex]

        screenWidth = 640
        screenHeight = 480
        screen = pg.display.set_mode((screenWidth, screenHeight))
        screenColor = THECOLORS['black']
        xBoundary = [0, 640]
        yBoundary = [0, 480]
        lineColor = THECOLORS['white']
        lineWidth = 4
        drawBackground = DrawBackground(screen, screenColor, xBoundary, yBoundary, lineColor, lineWidth)

        numOfAgent = 25
        numDistractors = numOfAgent - 2
        circleColorSpace = [[0, 255, 0], [255, 0, 0]] + [[255, 255, 255]] * numDistractors
        circleSize = 10
        positionIndex = [0, 1]
        agentIdsToDraw = list(range(numOfAgent))
        saveImage = True
        dirPYFile = os.path.dirname(__file__)
        imageSavePath = os.path.join(dirPYFile, '..', 'data', 'forDemo')
        if not os.path.exists(imageSavePath):
            os.makedirs(imageSavePath)
        FPS = 30
        drawState = DrawState(FPS, screen, circleColorSpace, circleSize, agentIdsToDraw, positionIndex,
                              saveImage, imageSavePath, drawBackground)

        # MDP Env
        xBoundary = [0, 640]
        yBoundary = [0, 480]
        stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
        distanceToVisualDegreeRatio = 20
        killzoneRadius = 2.5 * distanceToVisualDegreeRatio
        sheepId = 0
        wolfId = 1
        isTerminal = IsTerminal(sheepId, wolfId, killzoneRadius)
        numMDPTimeStepPerSecond = 5  # change direction every 200ms
        numFramesToInterpolate = int(FPS / numMDPTimeStepPerSecond - 1)  # interpolate each MDP timestep to multiple frames; check terminal for each frame

        interpolateStateForVisualization = InterpolateStateForVisualization(numFramesToInterpolate,
                                                                            stayInBoundaryByReflectVelocity, isTerminal)
        stateIndexInTimeStep = 0
        actionIndexInTimeStep = 1
        nextStateIndexInTimeStep = 2
        visualizeTraj = VisualizeTraj(stateIndexInTimeStep, actionIndexInTimeStep, nextStateIndexInTimeStep,
                                      drawState, interpolateStateForVisualization)
        [visualizeTraj(trajectory) for trajectory in trajectoriesToVisualize]


def main():
    numActionDirections = 8
    actionSpace = [(np.cos(directionId * 2 * math.pi / numActionDirections),
                    np.sin(directionId * 2 * math.pi / numActionDirections))
                   for directionId in range(numActionDirections)]
    actionSpace.append((0,0))
    stateDim = 100
    actionDim = len(actionSpace)

    I=64
    J=30
    K=0.0001
    paramSet = {
        'INITIAL_EPSILON': 0.4,
        'FINAL_EPSILON': 0.01,
        'GAMMA': 0.99,
        'REPLAY_SIZE': 10000,
        'BATCH_SIZE': I,
        'REPLACE_TARGET_FREQ': 20,
        'HIDDEN_LAYER_WIDTH': J,
        'LR': K,
    }

    dqn = dqnModel(stateDim, actionDim, paramSet)
    print("#--------------------------")
    print("# batch size{},width:{},lr:{}".format(I,J,K))
    trainTask(dqn,actionSpace)


if __name__ == '__main__':
    main()