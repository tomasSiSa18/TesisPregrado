import itertools
from Experiment.classes.NoiseMechanisms import NoiseType, ApplyNoise, GADNoiseTeo, PADNoiseTeo, DepolarizingNoiseTeo
from Experiment.classes.ProportionalDistance import getEpsilonExperimental
import pennylane as qp


def LocalEpsilonCategoricalExperiment(stateList: list, noise: NoiseType, noiseParams: dict, accuracy: float, tolerance: float):
    
    if noise == NoiseType.PAD:
            assert noiseParams["p"] == 0.5 and noiseParams["l"] <= noiseParams["g"]
    elif noise == NoiseType.GAD:
        assert noiseParams["p"] == 0.5
    
    stateNoiseList = []
    distanceList = []
    distanceNoiseList = []
    epsilonList = []
    maxExperimentalEpsilon = -1
    bestPair = None
    
    for state in stateList:
        stateNoiseList.append(ApplyNoise(state, noise, noiseParams))
        
    combinations = itertools.combinations(range(len(stateList)), 2)
    
    for (i, j) in combinations:
        rho = stateList[i]
        sigma = stateList[j]
        
        rhoPrime = stateNoiseList[i]
        sigmaPrime = stateNoiseList[j]

        d = qp.math.trace_distance(rho, sigma)
        dNoise = qp.math.trace_distance(rhoPrime, sigmaPrime)
        
        distanceList.append(d)
        distanceNoiseList.append(dNoise)
        
        epsilonExperimental = getEpsilonExperimental(rhoPrime, sigmaPrime, accuracy)
        epsilonList.append(epsilonExperimental)
        
        if epsilonExperimental > maxExperimentalEpsilon: 
            bestPair = (i, j)
            maxExperimentalEpsilon = epsilonExperimental

    max_trace = max(distanceList)
        
    if(noise == NoiseType.DEP):
        theoreticalEpsilon = DepolarizingNoiseTeo(noiseParams["p"], max_trace, rho.shape[0])
    elif(noise == NoiseType.GAD):
        theoreticalEpsilon = GADNoiseTeo(max_trace, noiseParams["g"])
    elif(noise == NoiseType.PAD):
        theoreticalEpsilon = PADNoiseTeo(max_trace, noiseParams["g"], noiseParams["l"])
        
    experimentalEpsilon = max(epsilonList)
    
    assert experimentalEpsilon <= theoreticalEpsilon + tolerance
    
    delta_ang = abs(((360*bestPair[0])/len(stateList))- ((360*bestPair[1])/len(stateList)))
    max_angle = min(360-delta_ang, delta_ang)
        
    return experimentalEpsilon, theoreticalEpsilon, epsilonList, distanceList, distanceNoiseList, bestPair, delta_ang, max_angle


    