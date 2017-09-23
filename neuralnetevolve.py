import numpy as np
import gym
from functools import reduce
env = gym.make('LunarLander-v2')
env = gym.wrappers.Monitor(env, 'evolve', force=True)
env.reset()

#-------------------------   Environment <<<
actionDim = env.action_space.shape
print(actionDim)
obsDim = env.observation_space.shape[0]
#-------------------------    >>>
#-------------------------    Functions <<<
def mapl  (func, xs) :
    return list(map(func, xs))

def zipl (*xs):
    return list(zip(*xs))
    
def sigmoid(x):
    return 1/(1+np.exp(-x))
# Neural net
def propagate (layers, funcs, xs): 
    res = xs
    for layer, func in zipl(layers, funcs) :
        res = mapl(func, np.matmul(layer, res))
    return [i/sum (res) for i in res]

def proportsigns(x) :
    x = int(x)
    if np.random.rand() < x/100 :
        ret = [-1 for i in range(x)] + [1 for i in range(100-x)]
    else :
        ret = [1 for i in range(100)]
    return ret

# Genetic Algorithm -- CHANGE IF NETWORK MADE MORE COMPLEX
def mutate(layers):
    print('Mutating '+ str(layers))
    ret = mapl(lambda x: x*np.random.choice(proportsigns(np.round(mutationAmount*100))),  (layers + mutationAmount * (np.random.rand(len(layers), len(layers[0]), len(layers[0][0])) - 0.5)))
    print('Mutated to' + str(ret))
    return ret
def breed(layers1, layers2):
    ret = []
    layerss = [layers1, layers2]
    for i in range(len(layers1)):
        ret.append([])
        for j in range(len(layers1[0])): 
            ret[i].append(layerss[np.random.randint(2)][i][j])
    return ret
    
# Run trial
def runtrial (layers, funcs):
    obs = env.reset()
    points = 0
    action = 0
    while True:
        ys = propagate(layers, funcs, obs)
        action = ys.index(max(ys))
        obs, reward, done, info = env.step(action)
        
        points += reward
        if done: return points
    
            
#------------------------- >>>
#------------------------- Constants <<<
learningRate = 1
numNeurons = 4
numGens = 500
trialsPerGen = 50
numBest = 4
mutationAmount = 0.4
mutationChance = 0.4
funcs = [np.exp]
# ------------------------  >>>
# ------------------------ Variables <<<

population = [(0, [[[(np.random.rand()  - 0.5)*2 for i in range(obsDim)] for j in range(numNeurons)]]) for trial in range(trialsPerGen)]
best = [(0, [[[(np.random.rand()  - 0.5)*2 for i in range(obsDim)] for j in range(numNeurons)]]) for trial in range(trialsPerGen)]
avg = 0
# ------------------------ >>>
# ------------------------ Main <<<

bestavg = 0
for gen in range(numGens) :
    tot = 0  
    bestCounter = 0
    for trial in range(trialsPerGen):
        if (bestCounter < numBest) :
            layers = best[trial][1]
            bestCounter += 1
        elif (np.random.rand() < mutationChance) :
            layers = mutate(best[np.random.randint(numBest)][1])
        else :
            layers =best[np.random.randint(numBest)][1]
        points = runtrial(layers, funcs)
        population[trial] = (points, layers)
        print(layers)
        tot += points
        print('Trial: ' + str(trial + gen * numGens))
        print('Points: ' + str(points))
        print('Previous avg: ' + str(avg))
        print('Mutation amount: ' + str(mutationAmount))
        print('Best average: ' + str(bestavg))
    best = sorted(population, key=lambda x: x[0])[::-1][:numBest]
    print('================================================== \n Gen: ' + str(gen))
    print('Best: ' + str(best[0]))
    avg = float(tot)/trialsPerGen
    besttot = 0
    for trial in best :
        besttot += trial[0]
    bestavg = besttot / numBest
    mutationAmount = 0.05**(0.8+ bestavg/500)
    if mutationAmount > 1 : mutationAmount = 1

