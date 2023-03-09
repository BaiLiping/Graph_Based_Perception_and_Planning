import numpy as np

def predictAnchors(posteriorParticlesAnchors, parameters):
    numParticles = parameters['numParticles']
    anchorNoiseVariance = parameters['anchorRegularNoiseVariance']
    survivalProbability = parameters['survivalProbability']

    numAnchors = posteriorParticlesAnchors.shape[1]
    predictedParticlesAnchors = np.zeros((2, numParticles, numAnchors))
    weightsAnchor = np.zeros((numParticles, numAnchors))

    for anchor in range(numAnchors):
        weightsAnchor[:, anchor] = survivalProbability * posteriorParticlesAnchors[:, anchor, 2]
        anchorNoise = np.sqrt(anchorNoiseVariance) * np.random.randn(2, numParticles)
        predictedParticlesAnchors[:, :, anchor] = posteriorParticlesAnchors[:, anchor, 1] + anchorNoise

    return predictedParticlesAnchors, weightsAnchor
