# see: https://github.com/openai/gym/wiki/CartPole-v0
# Tutorial: https://www.youtube.com/watch?v=ZipAjLSNlQc&list=PL-9x0_FO_lglnlYextpvu39E7vWzHhtNO

# NOTE: we need to stay in the air for at least 195 steps in 100 consecutive trials to beat this challenge!

from time import perf_counter
from os.path import dirname, abspath, basename, splitext
import sys
import gym
import numpy as np
from gym import wrappers

##########################################################################################
# Configuration
##########################################################################################
splitext
maxScore = 200 # defined by the environment
nPolicies = 100 # how many different policies do we want to try?
nTrials = 100 # how many episodes (aka trial) to run for each policy?
envName = 'CartPole-v0'


##########################################################################################
# Run simulation with given policy
##########################################################################################

def runOnce(weights):
  observation = env.reset()
  done = False
  iStep = 0
  score = 0
  startTime = perf_counter()

  # run simulation for this trial (max 200 steps)
  while not done:
    iStep += 1

    # Why does this part work?
    # Because all 4 observation variables indicate: bias toward the right when positive, so we want to go right to counter-balance.
    action = 1 if np.dot(weights, observation) > 0 else 0
    observation, reward, done, info = env.step(action)
    score += reward
  
  endTime = perf_counter() - startTime
  elapsed = endTime - startTime

  return score, elapsed


##########################################################################################
# Do the whole thing
##########################################################################################

print('starting policy evaluation...')

# setup
env = gym.make(envName)
averageScores = []
bestScore = 0
bestWeights = []
allTimes = []

# evaluate all policies and pick the best one!
for iPolicy in range(nPolicies):
  weights = np.random.uniform(-1, 1, 4)
  trialScores = []
  times = []

  # run trial (each trial runs the simulation once)
  for iTrial in range(nTrials):
    score, elapsed = runOnce(weights)
    trialScores.append(score)
    times.append(elapsed)

  # done with this episode - update bookkeeping
  avgScore = sum(trialScores) / len(trialScores)
  averageScores.append(avgScore)
  allTimes.append(times)

  print(f'policy #{iPolicy}: {avgScore}')

  if avgScore > bestScore:
    # found a better policy! hooray!
    bestWeights = weights
    bestScore = avgScore

    if bestScore >= maxScore:
      # we made it!
      break


##########################################################################################
# Wrap things up
##########################################################################################

print('finished!')

# now, record the best run to file!
cwd = dirname(abspath('.'))
fpath = sys.argv[0]

# cwd = abspath('.')
fpath = __file__
fname = splitext(basename(fpath))[0] # name of current .py file (without .py extension)
env = wrappers.Monitor(env, f'_results/{fname}', force=True)
runOnce(bestWeights)
