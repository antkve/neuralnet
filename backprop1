import numpy as np
import gym
env = gym.make('CartPole-v1')
#env = gym.make('LunarLander-v2')
env = gym.wrappers.Monitor(env, 'lunarlander3', force=True)
env.reset()

#-------------------------   Environment <<<
actionDim = env.action_space.shape
obsDim = env.observation_space.shape[0]
#-------------------------    >>>
#-------------------------    Exportable functions <<<
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

# Backprop for a single frame, returns delta weight gradient
def inv (x):
    if x==0: return 1
    else : return 0
def backpropsinglelayer (layer, func, xs, correctcat, ys):
    dweights = np.zeros((len(layer), len(layer[0])))
    dweights[correctcat] = [x * (1 - max(ys)) for x in xs]
    return dweights
     
# Adjust neural net based on a trial
def adjust (layers, functions, xss, correctcats, yss):
    grad = np.zeros((len(layers), len(layers[0]), len(layers[0][0])))
    for xs, yscat, ys in zipl(xss, correctcats, yss) : 
        grad[0] += backpropsinglelayer (layers[0], functions[0], xs, yscat, ys)
    grad[0] *= learningRate
    layers += grad
    return layers

# Run trial
def runtrial (layers, funcs):
    obs = env.reset()
    points = 0
    xss = []
    yss = []
    action = 0
    while True:
        ys = propagate(layers, funcs, obs)
        yss.append(ys)
        if all(obs): xss.append(obs)
        action = ys.index(max(ys))
        obs, reward, done, info = env.step(action)
        
        points += reward
        if done: return (points, xss, yss)
    
            
#------------------------- >>>
#------------------------- Constants <<<
learningRate = 1
numNeurons = 2
numGens = 500
trialsPerGen = 50
funcs = [np.exp]
# ------------------------  >>>
# ------------------------ Variables <<<
layers = [[[np.random.rand()  - 0.5 for i in range(obsDim)] for j in range(numNeurons)]]
dummylayers = layers
trials = []
prevavg = 100

# ------------------------ >>>
# ------------------------ Main <<<


for gen in range(numGens) :
    tot = 0  
    for trial in range(trialsPerGen):
        points, xss, yss = runtrial(layers, funcs)
        trials.append((points, layers, xss, yss))
        
        yscats = [ys.index(max(ys)) for ys in yss]
        if (points > int(prevavg)) :
            dummylayers = adjust(layers, funcs, xss, yscats, yss)
        else :
            layers = adjust(layers, funcs, xss, mapl(inv, yscats), yss)
        print(layers)
        tot += points
        print('Trial: ' + str(trial + gen * numGens))
        print('Points: ' + str(points))
        print('Previous avg: ' + str(prevavg))
        
    print('Gen: ' + str(gen))
    prevavg = float(tot)/trialsPerGen
