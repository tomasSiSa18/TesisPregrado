from enum import Enum
import pennylane as qp

from Experiment.classes import NoiseMechanisms, QuantProcesses, Encoder
from tqdm import tqdm

from Experiment.classes.NoiseMechanisms import NoiseType, ApplyNoise

from Experiment.classes.ProportionalDistance import getEpsilonExperimental


def CentralEpsilonExperiment(setRho, noise: NoiseType, noiseParams: dict, tolerance: float, accuracy: float = 0.001):
        
    if noise == NoiseType.PAD:
            assert noiseParams["p"] == 0.5 and noiseParams["l"] <= noiseParams["g"]
    elif noise == NoiseType.GAD:
        assert noiseParams["p"] == 0.5
    
    distanceList = []
    distanceNoiseList = []
    epsilonList = []

    rhoK = QuantProcesses.Aggregate(setRho)
    
    rhoKPrime = ApplyNoise(rhoK, noise, noiseParams)

    for i in tqdm(range(len(setRho))):
        
        setSigma = Encoder.encodeExcludeOne(i, setRho)
        sigmaK = QuantProcesses.Aggregate(setSigma)
        sigmaKPrime = ApplyNoise(sigmaK, noise, noiseParams)
        distanceList.append(qp.math.trace_distance(rhoK, sigmaK))
        distanceNoiseList.append(qp.math.trace_distance(rhoKPrime, sigmaKPrime))
        
        
        iter_max = getEpsilonExperimental(rhoKPrime, sigmaKPrime, accuracy)
        epsilonList.append(iter_max)
    
    
    max_trace = max(distanceList)
    
    if(noise == NoiseType.DEP):
        theoreticalEpsilon = NoiseMechanisms.DepolarizingNoiseTeo(noiseParams["p"], max_trace, rhoK.shape[0])
    elif(noise == NoiseType.GAD):
        theoreticalEpsilon = NoiseMechanisms.GADNoiseTeo(max_trace, noiseParams["g"])
    elif(noise == NoiseType.PAD):
        theoreticalEpsilon = NoiseMechanisms.PADNoiseTeo(max_trace, noiseParams["g"], noiseParams["l"])
    
    experimentalEpsilon = max(epsilonList)
    
    assert experimentalEpsilon <= theoreticalEpsilon + tolerance
        
    return experimentalEpsilon, theoreticalEpsilon, epsilonList, distanceList, distanceNoiseList
    
    