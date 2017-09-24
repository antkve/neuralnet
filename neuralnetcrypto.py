import numpy as np
import matplotlib.pyplot as plt
import time
import poloniex
from fileinput import close

polo = poloniex.Poloniex()

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
    
# Genetic Algorithm -- CHANGE IF NETWORK MADE MORE COMPLEX
def randsign (p) :
    if np.random.rand() < p :
        return -1
    else : return 1
def mutate(layers):
    print('Mutating '+ str(layers))
    mutatedlayers = layers + mutationAmount * (np.random.rand(len(layers), len(layers[0]), len(layers[0][0])) - 0.5)
    
    ret = mapl(lambda x: x*randsign(mutationAmount),  mutatedlayers)
    print('Mutated to' + str(ret))
    return ret
def breed(layers1, layers2):
    ret = []
    layerss = [layers1, layers2]
    for i in range(len(layers1)):
        for j in range(len(layers1[0])): 
            ret[i].append(layerss[np.random.randint(2)][i][j])
    return ret



# Run trial
def runtrial (layers, funcs, trialNum, prices):
    BTC = 1
    prices = prices[trialNum*288:(trialNum + 1)*288]
  
    frame = 1
    position = 0
    numCoins = 0
    while True:
        close = float(prices[frame-1]['close'])
        pricediff = close - float(prices[frame - 2]['close'])
        numCoins = (position * BTC)/close
        BTC = BTC + ( pricediff *numCoins)
        if (frame > 14) : last14 = prices[(frame - 14): frame]
        else : last14 = prices[0:frame]
        data = rsi(last14), -movingavgdiff(last14)*200000, 1
        ys = propagate(layers, funcs, data)
        
        
        position = ys[0]
        if (position > 1) : position = 1
        if (position < 0) : position = 0
        
        frame +=1
        if frame >= int(float(8500)/300):  
            
            return BTC
# ---------------------------------- Stock indicator functions <<<

def rsi(lastPrices):
    gains = 0
    losses = 0
    lastclose = float(lastPrices[0]['close'])
    for price in lastPrices :
        close = float(price['close'])
        diff = lastclose - close
        if diff > 0:
            gains += diff
        else :
            losses -= diff
    avggain = gains / len(lastPrices)
    avgloss = losses / len(lastPrices)
    if (avgloss == 0): avgloss = 0.00001
    RS = avggain/avgloss
    return 100 - 100/(1 + RS)

def movingavgdiff (lastPrices):
    ret = 0
    for price in lastPrices :
        close = float(price['close'])
        ret += close
        final = close
    return final - ret/len(lastPrices)
# ---------------------------------- >>>

#------------------------- >>>



#------------------------- Constants <<<
learningRate = 1
numNeurons = 2
numGens = 5000
trialsPerGen = 50
numBest = 8
mutationAmount = 0.4
mutationChance = 0.4
funcs = [np.exp]
starttime = time.time()
# ------------------------  >>>
# ------------------------ Variables <<<

population = [(0, [[[(np.random.rand()  - 0.5)*2 for i in range(3)] for j in range(numNeurons)] for i in [0,1]] ) for trial in range(trialsPerGen)]
best = [(0, [[[(np.random.rand()  - 0.5)*2 for i in range(3)] for j in range(numNeurons)]]) for trial in range(trialsPerGen)]
avg = 0
avgs = []
bestavg = 0
prices = polo.returnChartData('BTC_ETH', 300, starttime - (86400)*trialsPerGen, starttime)
# ------------------------ >>>
# ------------------------ Main <<<
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
        points = runtrial(layers, funcs, trial, prices)
        population[trial] = (points, layers)
        tot += points
        print('Trial: ' + str(trial ) +', Gen: ' + str(gen))
        print('Points: ' + str(points))
        print('Previous avg: ' + str(avg))
        print('Mutation amount: ' + str(mutationAmount))
        print('Best average: ' + str(bestavg))
    best = sorted(population, key=lambda x: x[0])[::-1][:numBest]
    print('================================================== \n Gen: ' + str(gen))
    print('Best: ' + str(best[0]))
    avg = float(tot)/trialsPerGen
    avgs.append(avg)
    if gen%1000 ==0 :
        plt.plot(avgs)
        plt.show()
    besttot = 0
    for trial in best :
        besttot += trial[0]
    bestavg = besttot / numBest
    mutationAmount = 0.05**((avg - 1)*10+0.8)
    if mutationAmount > 1 : mutationAmount = 1
