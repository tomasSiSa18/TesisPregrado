from enum import Enum
import pennylane as qp

from Experiment.classes import NoiseMechanisms, ProportionalDistance, QuantProcesses, Encoder
from tqdm import tqdm

class NoiseType(Enum):
    DEP = 1
    GAD = 2
    PAD = 3

def ApplyNoise(density_matrix, noise: NoiseType, noiseParams):
    
    result = None
    
    if(noise == NoiseType.DEP):
        result = NoiseMechanisms.DepolarizingNoise(density_matrix, noiseParams["p"])
    elif(noise == NoiseType.GAD):
        result = NoiseMechanisms.GADNoiseMultiQubit(density_matrix, noiseParams["p"], noiseParams["g"])
    elif(noise == NoiseType.PAD):
        result = NoiseMechanisms.PADNoiseMultiQubit(density_matrix, noiseParams["p"], noiseParams["g"], noiseParams["l"])
        
    return result

def CentralEpsilonExperiment(setRho, noise: NoiseType, noiseParams: dict, tolerance: float):
        
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
        rho_to_sigma_dPD = ProportionalDistance.dPD(rhoKPrime.copy(), sigmaKPrime.copy(), 0.001)
        sigma_to_rho_dPD = ProportionalDistance.dPD(sigmaKPrime.copy(), rhoKPrime.copy(), 0.001)
        
        iter_max = max(rho_to_sigma_dPD, sigma_to_rho_dPD)
        epsilonList.append(iter_max)
    
    
    max_trace = max(distanceList)
    theoreticalEpsilon = None
    if(noise == NoiseType.DEP):
        theoreticalEpsilon = NoiseMechanisms.DepolarizingNoiseTeo(noiseParams["p"], max_trace, rhoK.shape[0])
    elif(noise == NoiseType.GAD):
        theoreticalEpsilon = NoiseMechanisms.GADNoiseTeo(max_trace, noiseParams["g"])
    elif(noise == NoiseType.PAD):
        theoreticalEpsilon = NoiseMechanisms.PADNoiseTeo(max_trace, noiseParams["g"], noiseParams["l"])
    
    experimentalEpsilon = max(epsilonList)
    
    assert experimentalEpsilon <= theoreticalEpsilon + tolerance
        
    return experimentalEpsilon, theoreticalEpsilon, epsilonList, distanceList, distanceNoiseList
    
    