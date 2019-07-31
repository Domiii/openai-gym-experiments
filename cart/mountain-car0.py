# basic Q-Learning implementation

import sys
import gym
import numpy as np
from matplotlib import pyplot as plt
from time import perf_counter, sleep
from os.path import dirname, abspath, basename, splitext
from gym import wrappers
from functools import reduce
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager, JoinableQueue, Pipe
import multiprocessing
import logging
import traceback
from collections.abc import Iterable

##########################################################################################
# Configuration
##########################################################################################

# parameters we know about the environment:
# https://github.com/openai/gym/wiki/MountainCar-v0
# https://github.com/openai/gym/tree/master/gym/envs/classic_control/mountain_car.py
envName = 'MountainCar-v0'
maxSteps = 200
stateSpace = (
    (-1.2, 0.6),    # position
    (-.8, .8)       # velocity
)
#actionSpace = (0, 1, 2) # left, noop, right
actionSpace = (0, 2) # left, noop, right

def hasDied(done, nSteps):
  return done and nSteps >= maxSteps

# parameters we can play with
goodConfigs = [
    {
        'nIterations': 200,  # how long do we want to train for?
        'stateResolution': 100,  # amount of bins for each state variable
        'alpha': 0.01,
        'gamma': 0.9
    }
]
configIndex = 0


config = goodConfigs[configIndex]

nIterations = config['nIterations']
stateResolution = config['stateResolution']
alpha = config['alpha']
gamma = config['gamma']

#nLogSteps = nIterations
nLogSteps = 20
deathValue = -1000

stateSpaceSize = stateResolution ** len(stateSpace) * len(actionSpace)

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
  def __init__(self, env, alpha, gamma, stateSpace, actionSpace, initialQ=0):
    self.success = False
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
    stateSize = reduce(lambda acc, next: acc*next, (len(s)
                                                    for s in stateSpace))
    actionSize = len(actionSpace)
    qLen = stateSize * actionSize
    self.q = np.zeros(qLen) + initialQ

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

    # add-on for certain state spaces: don't even try a really shitty move
    iTries = 3
    while self.q[self.lastStateIndex + action] <= deathValue * 0.5 and iTries > 0:
      action = np.random.choice(self.actionSpace)
      iTries -= 1

    results = self.stepAction(action)
    return results

  def step(self):
    #action = self.getMaxAction(self.lastStateIndex)
    # go with the action we already determined to be optimal last step
    action = self.nextAction
    return self.stepAction(action)

  def stepAction(self, action):
    q, alpha, gamma, iState = self.q, self.alpha, self.gamma, self.lastStateIndex

    # render
    # window_still_open = self.env.render()
    # if not window_still_open: return deathValue, True

    # take step
    observation, reward, done, info = self.env.step(action)
    nextIState = self.lastStateIndex = self.getStateIndex(observation)
    maxNextAction = self.nextAction = self.getMaxAction(nextIState)
    self.nSteps += 1

    # death should be punished hard
    died = hasDied(done, self.nSteps)
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

    if done and not died:
      self.success = True

    return reward, done

  # Make sure, some basic things are still in order
  # E.g.: all numbers are in sane range.
  def doSanityChecks(self):
    pass
    # assert all(
    #     x >= deathValue for x in self.q), "Q state-action value function has insane values"

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
    #print(f'stepped {iStep}')

  return score, nRandom


'''
TODO:
0. Clean up plotting code
1. add ThreadPoolExecutor: https://docs.python.org/dev/library/concurrent.futures.html#threadpoolexecutor
2. add semi-automated parameter tuning?
3. Record final result to video?
4. improve epsilon:
  -> maybe for later? actually do research on better epsilon functions!
  -> consider current maturity of trained results instead of nIterations?
  -> restart if not found within given amount of rounds?
'''


# setup plot
fig, axs = plt.subplots(2, 1, constrained_layout=True)

# draw it once
# animated scatter plot link: https://stackoverflow.com/questions/43674917/animation-in-matplotlib-with-scatter-and-using-set-offsets-autoscale-of-figure
ys = [0] * stateSpaceSize
xs = np.array(range(stateSpaceSize))
qShape = axs[0].scatter(xs, ys, s=1)
lines = [
    axs[1].plot([1], [0])[0],   # scores of per run
    axs[1].plot([1], [0])[0]    # ratio of random moves per run
]
axs[1].set_ylim([0, 1])

scores = []
randoms = []
plt.ion()
plt.show(block=False)


def drawResults():
  #print(f'runOnce, score: {score}, rand: {nRandom/q.nSteps}')

  # relim does not work for scatter; see: https://stackoverflow.com/a/51327480
  axs[0].ignore_existing_data_limits = True
  axs[0].update_datalim(qShape.get_datalim(axs[0].transData))
  axs[0].autoscale_view()

  n = len(scores)
  axs[1].set_xlim([0, n+1])
  #axs[1].relim()
  times = range(n)
  lines[0].set_data(times, scores)
  lines[1].set_data(times, randoms)

  fig.canvas.draw()
  fig.canvas.flush_events()
  plt.pause(0.1)


def logRun(newScores, newNRandoms, qOffsets):
  #scores.extend(newScores)
  #randoms.extend(newNRandoms[i]/newScores[i] for i in range(len(newNRandoms)))
  qShape.set_offsets(qOffsets)

# run!


def runN(q, n, epsilon):
  totalElapsed = 0
  maxScore = -1000000
  scores = []
  nRandoms = []
  for i in range(n):
    eps = epsilon(i)
    score, nRandom = runOnce(q, eps)
    scores.append(score)
    nRandoms.append(nRandom)
    maxScore = max(maxScore, score)

    # make sure, shit's still ok
    q.doSanityChecks()

    # show some stuff
    #if i % nLogSteps == nLogSteps-1:
      #taskQueue.put(drawResults)
      #pass

    #if q.success: # done!
      #break
  qOffsets = np.c_[xs, q.q]
  taskQueue.put((logRun, scores, nRandoms, qOffsets), block=False)
  taskQueue.put(drawResults)
  return q.success, n, maxScore
  #plt.pause(3)


# trains + learns only (no blocking distractions)
def run():
  try:
    env = gym.make(envName)
    q = Q(env, alpha, gamma, stateSpace, actionSpace)
    success = False
    totalIterations = 0
    totalElapsed = 0

    startTime = perf_counter()

    # first, run until terminal state is hit at least once
    while not success:
      success, n, maxScore = runN(q, nIterations, epsilon)
      totalIterations += n
      print(f'run #{totalIterations}, maxScore={maxScore}')

    # # run a few more times with exploration
    # success, n, maxScore = runN(q, 5 * nIterations, epsilon)
    # totalIterations += n

    # # run a few more times without exploration
    # success, n, maxScore = runN(q, nIterations, lambda i: 0)
    # totalIterations += n

    elapsed = perf_counter() - startTime
    totalElapsed += elapsed

    print(f'finished {totalIterations} iterations, maxScore={maxScore},' +
      f' took {totalElapsed:.2f}s, {(totalElapsed/totalIterations*1000):02}ms per iteration')
  except Exception as e:
    error(traceback.format_exc())

  taskQueue.close()
  #taskQueue.join()


logger = multiprocessing.log_to_stderr()
logger.setLevel(logging.WARNING)


def error(msg, *args):
  return logger.error(msg, *args)


# start (in parallel)!
nCores = max(1, multiprocessing.cpu_count()-1)
#print(nCores)
with ProcessPoolExecutor(max_workers=nCores) as workers, Manager() as manager:
  taskQueue = JoinableQueue()

  # run in serial (TODO: plot right away instead of using the taskQueue):
  # run()
  # if True:

  # only do one run in parallel
  result = workers.submit(run)
  while not result.done():
    
    taskArgs = taskQueue.get(block=True)
    if isinstance(taskArgs, Iterable):
      # process returned a function with arguments
      task, *args = taskArgs
      task(*args)
    elif taskArgs is not None:
      # process returned a function without arguments
      taskArgs()

#run()
print('Done!')
sleep(10)

