import math

import pandas as pd
import pennylane as qp
import matplotlib.pyplot as plt
import numpy as np

from Experiment.classes import NoiseMechanisms, Encoder

from Experiment.classes.NoiseMechanisms import NoiseType, getStretchNoise

noises = {NoiseType.DEP: {"p": 0.5}, NoiseType.PAD: {"p": 0.5, "g": 0.5, "l": 0.25}, NoiseType.GAD: {"p": 0.5, "g": 0.5}}

device = qp.device('default.mixed', wires=1)

dataset = pd.read_csv("/Users/tomassierra/Documents/Universidad/Tesis/TesisPregrado/ProcesamientoDataset/german_credit_data_for_quant.csv")

dict_df = dataset.to_dict(orient="records")
circuit = qp.QNode(Encoder.encodeOneCol, device)

data_encoded = Encoder.encodeAllSingle(dataset, circuit, 0)

plusState = np.array([1/math.sqrt(2), 1/math.sqrt(2)])
plusStateTranspose = plusState.conj().T

probsNormal = []


randomGen = np.random.default_rng(8)

uniforms = randomGen.random(size=(2000, 1000))

for rho in data_encoded:
        
    probsNormal.append((plusStateTranspose @ rho @ plusState).real)
probsNormal = np.array(probsNormal)
comparison_normal = uniforms < probsNormal

correctedByNoise = {}
for noiseMechanisms in noises.keys():
    probsNoise = []
    
    noiseParams = noises[noiseMechanisms]
    for rho in data_encoded:
         
        rhoNoise = NoiseMechanisms.ApplyNoise(rho, noiseMechanisms, noiseParams)
        
        probsNoise.append((plusStateTranspose @ rhoNoise @ plusState).real)
        
    probsNoise = np.array(probsNoise)

    #print(np.unique(np.round(probsNormal, decimals=5)))
    #print(np.unique(np.round(probsNoise, 5)))

    
    comparison = uniforms < probsNoise

    avgNormalValues = np.mean(comparison_normal, axis=1)
    
    avgNoiseValues = np.mean(comparison, axis=1)
    
    avgNoiseCorrectedValues = 1/2 + ((avgNoiseValues - (1/2))/getStretchNoise(noiseMechanisms, noiseParams))

    avgNormal = np.mean(avgNormalValues)
    
    avgNoise = np.mean(avgNoiseValues)
    
    avgNoiseCorrected = np.mean(avgNoiseCorrectedValues)

    stdNormal = np.std(avgNormalValues)
    stdNoise = np.std(avgNoiseValues)
    stdCorrected = np.std(avgNoiseCorrectedValues)

    print(f"F_B: Media: {avgNormal} Std: {stdNormal}")
    print(f"F_C: Media: {avgNoise} Std: {stdNoise}")
    print(f"F^: Media: {avgNoiseCorrected} Std: {stdCorrected}")
        
    correctedByNoise[noiseMechanisms] = avgNoiseCorrectedValues

names = {NoiseType.DEP: "Depolarizing", NoiseType.PAD: "Phase-amplitude damping", NoiseType.GAD: "Generalized amplitude damping"}

FB = avgNormalValues
allValues = np.concatenate([FB, *correctedByNoise.values()])

# F can only take the values k/n, so F_hat lives on a grid of spacing (1/n)/s.
# Bin edges are placed on that grid (binSteps grid points per bin) for every histogram;
# otherwise bins catch 2 or 3 grid points alternately and the histogram looks like a comb.
n = uniforms.shape[1]
binSteps = 4
edgesF = (np.arange(-1, n // binSteps + 2) * binSteps - 0.5) / n

fig, axes = plt.subplots(len(correctedByNoise), 1, sharex=True, sharey=True, figsize=(6.4, 7.5))

for ax, (noiseMechanism, FH) in zip(axes, correctedByNoise.items()):
    s = getStretchNoise(noiseMechanism, noises[noiseMechanism])
    epsilon = math.log((1 + s) / (1 - s))
    edgesHat = 1/2 + (edgesF - 1/2) / s

    ax.hist(FB, bins=edgesF, density=True, color="grey", alpha=0.5, label=rf"$F_B$ (no noise), std = {FB.std():.4f}")
    ax.hist(FH, bins=edgesHat, density=True, color="C0", alpha=0.6, label=rf"$\hat{{F}}$ (noisy, corrected), std = {FH.std():.4f}")
    ax.axvline(probsNormal.mean(), color="black", linestyle="--", linewidth=1, label=rf"$E[F_B]$ = {probsNormal.mean():.3f}")
    ax.set_title(rf"{names[noiseMechanism]}: $s$ = {s:.3f}, $\varepsilon$ = {epsilon:.2f}", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)

axes[0].set_xlim(allValues.min(), allValues.max())
fig.supxlabel("Fraction of + outcomes per repetition")
fig.supylabel("Density of repetitions")
fig.tight_layout()
fig.savefig("Experiment/imgs/histograma_collateral_mecanismos.png", dpi=300)
plt.show()

    
    