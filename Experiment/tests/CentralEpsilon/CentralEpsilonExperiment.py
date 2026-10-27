from enum import Enum
import pandas as pd
import pennylane as qp

from Experiment.classes import NoiseMechanisms, ProportionalDistance, QuantProcesses, Encoder
from tqdm import tqdm

class NoiseType(Enum):
    DEP = 1
    GAD = 2
    PAD = 3

def CentralEpsilonExperiment(noise: NoiseType, noiseParams: dict, data: str, tolerance: float):
        
    if noise == NoiseType.PAD:
            assert noiseParams["p"] == 0.5 and noiseParams["l"] <= noiseParams["g"]
    
    device = qp.device('default.mixed', wires=1)

    dataset = pd.read_csv(data)
    experimentalEpsilonTotal=0
    theoreticalEpsilonTotal = 0
    
    epsilons = {}
    
    circuit = qp.QNode(Encoder.encodeOneCol, device)
    
    for k in tqdm(range(7)):
    
        distanceList = []
        epsilonList = []

        #Codifico D
        Qd = Encoder.encodeAllSingle(dataset, circuit, k)
        rhoK = QuantProcesses.Aggregate(Qd)
        
        if(noise == NoiseType.DEP):
            rhoKPrime = NoiseMechanisms.DepolarizingNoise(rhoK, noiseParams["p"])
        elif(noise == NoiseType.GAD):
            rhoKPrime = NoiseMechanisms.GADNoiseMultiQubit(rhoK, noiseParams["p"], noiseParams["g"])
        elif(noise == NoiseType.PAD):
            rhoKPrime = NoiseMechanisms.PADNoiseMultiQubit(rhoK, noiseParams["p"], noiseParams["g"], noiseParams["l"])

        #Codifico
        for i in tqdm(range(1000)):
            
            Qd_prime = Encoder.encodeExcludeOne(i, Qd)
            sigmaK = QuantProcesses.Aggregate(Qd_prime)
            if(noise == NoiseType.DEP):
                sigmaKPrime = NoiseMechanisms.DepolarizingNoise(sigmaK, noiseParams["p"])
            elif(noise == NoiseType.GAD):
                sigmaKPrime = NoiseMechanisms.GADNoiseMultiQubit(sigmaK, noiseParams["p"], noiseParams["g"])
            elif(noise == NoiseType.PAD):
                sigmaKPrime = NoiseMechanisms.PADNoiseMultiQubit(sigmaK, noiseParams["p"], noiseParams["g"], noiseParams["l"])
            distanceList.append(qp.math.trace_distance(rhoK, sigmaK))
            rho_to_sigma_dPD = ProportionalDistance.dPD(rhoKPrime.copy(), sigmaKPrime.copy(), 0.001, i)
            sigma_to_rho_dPD = ProportionalDistance.dPD(sigmaKPrime.copy(), rhoKPrime.copy(), 0.001, i)
            
            iter_max = max(rho_to_sigma_dPD, sigma_to_rho_dPD)
            epsilonList.append(iter_max)
        
        
        max_trace = max(distanceList)
        theoreticalEpsilonK = None
        if(noise == NoiseType.DEP):
           theoreticalEpsilonK = NoiseMechanisms.DepolarizingNoiseTeo(noiseParams["p"], max_trace, 2)
        elif(noise == NoiseType.GAD):
            theoreticalEpsilonK = NoiseMechanisms.GADNoiseTeo(max_trace, noiseParams[0])
        elif(noise == NoiseType.PAD):
            theoreticalEpsilonK = NoiseMechanisms.PADNoiseTeo(max_trace, noiseParams["g"], noiseParams["l"])
        
        theoreticalEpsilonTotal += theoreticalEpsilonK
        
        experimentalEpsilonK = max(epsilonList)
        experimentalEpsilonTotal += experimentalEpsilonK
        
        assert experimentalEpsilonK <= theoreticalEpsilonK + tolerance
        
        epsilons[cols[k]] = [cols[k],max_trace,theoreticalEpsilonK, experimentalEpsilonK, theoreticalEpsilonK/experimentalEpsilonK]
        
    return epsilons, theoreticalEpsilonTotal, experimentalEpsilonTotal
    
    