import numpy as np

def predictMeasurements(predictedParticles, anchorPositions, weightsAnchor):
    _, numParticles, numAnchors = anchorPositions.shape

    predictedMeans = np.zeros((numAnchors,))
    predictedUncertainties = np.zeros((numAnchors,))
    predictedRange = np.zeros((numParticles, numAnchors))

    for anchor in range(numAnchors):
        predictedRange[:, anchor] = np.sqrt((predictedParticles[0, :] - anchorPositions[0, :, anchor])**2 + (predictedParticles[1, :] - anchorPositions[1, :, anchor])**2).T
        predictedMeans[anchor] = np.dot(predictedRange[:, anchor], weightsAnchor[:, anchor]) / np.sum(weightsAnchor[:, anchor])
        predictedUncertainties[anchor] = np.dot((predictedRange[:, anchor] - predictedMeans[anchor])**2, weightsAnchor[:, anchor]) / np.sum(weightsAnchor[:, anchor])

    return predictedMeans, predictedUncertainties, predictedRange
