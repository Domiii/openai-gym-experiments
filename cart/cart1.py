# basic Q-Learning implementation

import time
from matplotlib import pyplot as plt
from time import perf_counter
from os.path import dirname, abspath, basename, splitext
import sys
import gym
import numpy as np
from gym import wrappers
from functools import reduce


##########################################################################################
# Configuration
##########################################################################################

# parameters we can play with
nIterations = 100  # how long do we want to train for?
alpha = 0.1
gamma = 0.99
defaultResolution = 4


# parameters we know about the environment
envName = 'CartPole-v0'
maxScore = 200
stateSpace = (
    (-4.8, 4.8, defaultResolution),
    (-5, 5, defaultResolution),
    (-.418, .418, defaultResolution),
    (-5, 5, defaultResolution)
)
actionSpace = (0, 1)

##########################################################################################
# Q-learning
##########################################################################################

# This implementation works for small action-state spaces
# as it does not use a sparse representation.
MinFloat = np.finfo.min
class Q:
  def __init__(self, env, alpha, gamma, stateSpace, actionSpace, defaultQ = 0):
    self.env = env
    self.alpha = alpha
    self.gamma = gamma
    self.lastStateIndex = None
    self.nextAction = None
    
    # prepare state space (simplest version: just dense linear spaces)
    stateSpace = self.stateSpace = (
      np.linspace(s[0], s[1], s[2]) for s in stateSpace
    )
    self.actionSpace = actionSpace
    self.maxActions = [None] * len(actionSpace)

    # allocate q vector
    stateSize = reduce(lambda acc, next: acc*next, (len(s) for s in stateSpace))
    actionSize = len(actionSpace)
    qLen = stateSize * actionSize
    self.q = np.zeros(qLen) * defaultQ

  def reset(self):
    observation = self.env.reset()
    iState = self.lastStateIndex = self.getStateIndex(observation)
    self.nextAction = self.getMaxAction(iState)

  # convert observation data into a single number representing the 1D index of the state-value space
  def getStateIndex(self, observation):
    iState = 0
    qRes = len(actionSpace)
    for i in range(len(self.stateSpace)):
      x = observation[i]
      s = self.stateSpace[i]
      iBin = np.searchsorted(s, x)
      iState += iBin * qRes
      qRes *= len(s)
    return iState


  # any action that leads to a state of max expected value
  def getMaxAction(self, iState):
    # find the max action(s) from that state
    maxActions, nMax = self.getMaxActions(iState)

    # pick any of the max actions at random
    iAction = np.random.randint(0, nMax)
    return maxActions[iAction]

    # q = self.q
    # actionCount = len(maxActions)
    # maxQ = MinFloat
    # maxAction = 0
    # for i in range(actionCount):
    #   v = q[iState + i]
    #   if v > maxQ:
    #     maxAction = i
    # return maxAction


  # all actions that lead to the states of max expected value
  # NOTE: There can be multiple.
  def getMaxActions(self, iState):
    q = self.q
    maxActions = self.maxActions
    actionCount = len(maxActions)
    maxQ = MinFloat
    nMax = 0  # amount of max actions
    for i in range(actionCount):
      v = q[iState + i]
      if v > maxQ:
        nMax = 1
        maxActions[nMax-1] = i
      elif v == maxQ:
        nMax += 1
        maxActions[nMax-1] = i
    return maxActions, nMax


  def stepRandom(self):
    # pick an action at random
    action = np.random.choice(self.actionSpace)
    return self.stepAction(action)
    

  def step(self):
    # go with the action we already determined to be optimal last step
    return self.stepAction(self.nextAction)


  def stepAction(self, action):
    q, alpha, gamma = self.q, self.alpha, self.gamma

    # take step
    observation, reward, done, info = self.env.step(action)
    iState = self.lastStateIndex = self.getStateIndex(observation)
    nextAction = self.nextAction = self.getMaxAction(iState)

    # the max value to be expected from here on forward
    nextMax = q[iState + nextAction]

    # update Q value of new state
    oldVal = q[self.lastStateIndex + action]
    newVal = reward * gamma * nextMax
    q[iState + action] = (1-alpha) * (oldVal) + (alpha) * newVal

    return reward, done, info



##########################################################################################
# Run simulation with current Q (state-action value function)
##########################################################################################

def runOnce(q, env, eps):
  q.reset()
  done = False
  iStep = 0
  score = 0
  startTime = perf_counter()

  # run simulation for this trial (max 200 steps)
  while not done:
    iStep += 1

    # TODO: epsilon-greedy!

    reward, done, _ = q.step()
    score += reward

  elapsed = perf_counter() - startTime

  return score, elapsed


# run!

def runN(n):
  env = gym.make(envName)
  q = Q(env, alpha, gamma, stateSpace, actionSpace)

  for i in range(n):
    runOnce(q, epsilon)

