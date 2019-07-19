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
goodConfigs = [
  {
    'nIterations': 1000,  # how long do we want to train for?
    'stateResolution': 4,  # amount of bins for each state variable
    'alpha': 0.01,
    'gamma': 0.9
  },
  {
    'nIterations': 100,  # how long do we want to train for?
    'stateResolution': 6,  # amount of bins for each state variable
    'alpha': 0.01,
    'gamma': 0.9
  },
  {
    'nIterations': 200,  # how long do we want to train for?
    'stateResolution': 6,  # amount of bins for each state variable
    'alpha': 0.1,
    'gamma': 0.99
  },
  {
    'nIterations': 100,  # how long do we want to train for?
    'stateResolution': 10,  # amount of bins for each state variable
    'alpha': 0.01,
    'gamma': 0.9
  }
]
configIndex = 2


config = goodConfigs[configIndex]

nIterations = config['nIterations']
stateResolution = config['stateResolution']
alpha = config['alpha']
gamma = config['gamma']

nLogSteps = nIterations/20 * 100
deathValue = -100


# epsilon is the probability to favor exploration over exploitation
# (meaning: take random instead of "good" step)
def epsilon(iRun):
  return (1 - (iRun/nIterations))**2




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
    self.q = np.zeros(qLen) + initialQ

  def hasFinished(self):
    return self.nSteps >= maxScore

  def reset(self):
    observation = self.env.reset()
    iState = self.lastStateIndex = self.getStateIndex(observation)
    self.nextAction = self.getMaxAction(iState)
    self.nSteps = 0

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
    return maxActions, nMax


  def stepRandom(self):
    # pick an action at random
    action = np.random.choice(self.actionSpace)
    results = self.stepAction(action)
    return results
    

  def step(self):
    action = self.getMaxAction(self.lastStateIndex)
    # go with the action we already determined to be optimal last step
    # action = self.nextAction
    return self.stepAction(action)


  def stepAction(self, action):
    q, alpha, gamma, iState = self.q, self.alpha, self.gamma, self.lastStateIndex

    # take step
    observation, reward, done, info = self.env.step(action)
    nextIState = self.lastStateIndex = self.getStateIndex(observation)
    maxNextAction = self.nextAction = self.getMaxAction(nextIState)
    self.nSteps += 1

    # death should be punished hard
    died = done and self.nSteps < maxScore
    if died:
      q[iState + action] = reward = deathValue
    else:
      # the max value to be expected from here on forward
      nextMax = q[nextIState + maxNextAction]
      newVal = reward + gamma * nextMax

      # update Q value of new state
      oldQ = q[iState + action]

      #print(f'{iState+action} -> {nextIState + nextAction}')

      q[iState + action] = (1-alpha) * oldQ + (alpha) * newVal

    return reward, done

  # Make sure, some basic things are still in order
  # E.g.: all numbers are in sane range.
  def doSanityChecks(self):
    assert all(x >= deathValue for x in self.q), "Q state-action value function has insane values"
  
  def getMetaData(self):
    return self.q



##########################################################################################
# Run simulation with current Q (state-action value function)
##########################################################################################

def runOnce(q, eps):
  q.reset()
  done = False
  iStep = 0
  nRandom = 0
  score = 0

  # run simulation for this trial (max 200 steps)
  while not done:
    iStep += 1

    if np.random.uniform() < eps:
      nRandom += 1
      reward, done = q.stepRandom()
    else:
      reward, done = q.step()

    score += reward

  return score, nRandom


'''
TODO:
1. multithreading and cleaner code for plotting and other debugging stuff
2. add semi-automated parameter tuning
3. improve epsilon:
  -> maybe for later? actually do research on better epsilon functions!
  -> consider current maturity of trained results instead of nIterations?
  -> restart if not found within given amount of rounds?
'''

env = gym.make(envName)
q = Q(env, alpha, gamma, stateSpace, actionSpace)

# setup plot
fig, axs = plt.subplots(2, 1, constrained_layout=True)

# draw it once
# animated scatter plot link: https://stackoverflow.com/questions/43674917/animation-in-matplotlib-with-scatter-and-using-set-offsets-autoscale-of-figure
ys = q.q
xs = np.array(range(len(ys)))
qShape = axs[0].scatter(xs, ys, s=1)
lines = [
    axs[1].plot([1], [0])[0],
    axs[1].plot([1], [0])[0]
]
axs[1].set_ylim([0, maxScore])
scores = []
randoms = []
times = []
plt.ion()
plt.show(block=False)

def drawResults(i):
  #print(f'runOnce, score: {score}, rand: {nRandom/q.nSteps}')
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
  lines[0].set_data(times, scores)
  lines[1].set_data(times, randoms)

  fig.canvas.draw()
  fig.canvas.flush_events()
  plt.pause(0.0001)

# run!
def runN(q, n):
  totalElapsed = 0
  startTime = perf_counter()
  for i in range(n):
    eps = epsilon(i)
    score, nRandom = runOnce(q, eps)
    # scores.append(q.nSteps)
    # randoms.append(nRandom)
    # times.append(i)

    # make sure, shit's still ok
    q.doSanityChecks()

    # show some stuff
    if i % nLogSteps == 0:
      drawResults(i)
      pass

    if q.hasFinished():
        # done!
      drawResults(i)
      break

  elapsed = perf_counter() - startTime
  print(f'finished episode, took {elapsed}s')
  return q.hasFinished()
  #plt.pause(3)



while not runN(q, nIterations): pass

print('Done!')

plt.pause(3)

#input('Press ENTER to exit...')

