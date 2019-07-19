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

# parameters we know about the environment
envName = 'CartPole-v0'
maxScore = 200
stateSpace = (
    (-4.8, 4.8),
    (-5, 5),
    (-.418, .418),
    (-5, 5)
)
actionSpace = (0, 1)

# parameters we can play with
nIterations = 100  # how long do we want to train for?
alpha = 0.5
gamma = 0.98
stateResolution = 2 # amount of bins for each state variable

# epsilon is the probability to favor exploration over exploitation
# (meaning: take random instead of "good" step)
def epsilon(iStep, score):
  return .05 + (1 - (iStep/maxScore)**2) * .5




##########################################################################################
# Q-learning
##########################################################################################


# This implementation works for small action-state spaces
# as it does not use a sparse representation.
MinFloat = np.finfo('float32').min
class Q:
  def __init__(self, env, alpha, gamma, stateSpace, actionSpace, initialQ = 0):
    self.env = env
    self.alpha = alpha
    self.gamma = gamma
    self.lastStateIndex = None
    self.nextAction = None
    
    # prepare state space (simplest version: just dense linear spaces)
    stateSpace = self.stateSpace = [
        np.linspace(s[0], s[1], stateResolution) for s in stateSpace
    ]
    self.actionSpace = actionSpace
    self.maxActions = [None] * len(actionSpace)

    # allocate q vector
    stateSize = reduce(lambda acc, next: acc*next, (len(s) for s in stateSpace))
    actionSize = len(actionSpace)
    qLen = stateSize * actionSize
    self.q = np.zeros(qLen) * initialQ

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
      thisQ = q[iState + i]
      if thisQ > maxQ:
        nMax = 1
        maxActions[nMax-1] = i
        maxQ = thisQ
      elif thisQ == maxQ:
        nMax += 1
        maxActions[nMax-1] = i
        maxQ = thisQ
    return maxActions, nMax


  def stepRandom(self):
    # pick an action at random
    action = np.random.choice(self.actionSpace)
    return self.stepAction(action)
    

  def step(self):
    # go with the action we already determined to be optimal last step
    return self.stepAction(self.nextAction)


  def stepAction(self, action):
    q, alpha, gamma, iState = self.q, self.alpha, self.gamma, self.lastStateIndex

    # take step
    observation, reward, done, info = self.env.step(action)
    nextIState = self.lastStateIndex = self.getStateIndex(observation)
    nextAction = self.nextAction = self.getMaxAction(nextIState)

    # the max value to be expected from here on forward
    nextMax = q[nextIState + nextAction]
    newVal = reward + gamma * nextMax

    # update Q value of new state
    q[iState + action] = (1-alpha) * q[iState + action] + (alpha) * newVal

    return reward, done, info

  # Make sure, some basic things are still in order
  # E.g.: all numbers are in sane range.
  def doSanityChecks(self):
    assert all(x >= 0 for x in self.q), "Q state-action value function has insane values"
  
  def getMetaData(self):
    return self.q



##########################################################################################
# Run simulation with current Q (state-action value function)
##########################################################################################

def runOnce(q, epsilon):
  q.reset()
  done = False
  iStep = 0
  nRandom = 0
  score = 0
  startTime = perf_counter()

  # run simulation for this trial (max 200 steps)
  while not done:
    iStep += 1

    if np.random.uniform() < epsilon(iStep, score):
      nRandom += 1
      reward, done, _ = q.stepRandom()
    else:
      reward, done, _ = q.step()

    score += reward

  elapsed = perf_counter() - startTime

  return score, elapsed, nRandom


# run!
def runN(n):
  env = gym.make(envName)
  q = Q(env, alpha, gamma, stateSpace, actionSpace)

  # setup plot
  #plt.ion()
  fig, axs = plt.subplots(2, 1, constrained_layout=True)

  # draw it once
  # animated scatter plot link: https://stackoverflow.com/questions/43674917/animation-in-matplotlib-with-scatter-and-using-set-offsets-autoscale-of-figure
  ys = q.q
  xs = np.array(range(len(ys)))
  qShape = axs[0].scatter(xs, ys, s=5)
  scoreShape, = axs[1].plot([1], [0])
  axs[1].set_ylim([0, maxScore])
  scores = []
  scoreXs = []
  plt.ion()
  plt.show(block=False)

  for i in range(n):
    score, elapsed, nRandom = runOnce(q, epsilon)
    scores.append(score)
    scoreXs.append(i)

    #print(f'runOnce, score: {score}, elapsed: {elapsed}, rand: {nRandom/score}, maxQ: {max(q.q)}')

    if not (i % 1000):
      # re-draw it!
      Qs = np.c_[xs, q.q]
      # xs = 0.9 * np.random.rand(512)
      # ys = 0.9 * np.random.rand(512)
      # Qs = np.c_[xs, ys]
      qShape.set_offsets(Qs)

      # relim does not work for scatter; see: https://stackoverflow.com/a/51327480
      axs[0].ignore_existing_data_limits = True
      axs[0].update_datalim(qShape.get_datalim(axs[0].transData))
      axs[0].autoscale_view()

      axs[1].set_xlim([0, i+1])
      scoreShape.set_data(scoreXs, scores)

      fig.canvas.draw()
      fig.canvas.flush_events()
      plt.pause(0.0001)

    # make sure, shit's still ok
    q.doSanityChecks()

    # done!
    if score >= maxScore:
      print('GOOD!')
      break

  plt.pause(2)


runN(5000)

#input('Press ENTER to exit...')

