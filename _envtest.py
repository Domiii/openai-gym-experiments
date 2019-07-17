# To check all env available, uninstalled ones are also shown
# Env list: https://github.com/openai/gym/wiki
from gym import envs

allNames = [env.id for env in envs.registry.all()]

def printEnvs(names):
  print('\n'.join(names))

def printEnvCategories(names):
  names = [name.split('-')[0] for name in names]
  names = set(names)
  print(f'{len(names)} env categories found:', ', '.join(names))


printEnvCategories(allNames)
