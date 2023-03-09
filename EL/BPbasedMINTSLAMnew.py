import numpy as np
from numpy.random import rand

def BPbasedMINTSLAMnew(dataVA, clutteredMeasurements, parameters, trueTrajectory):
    numSteps, numSensors = clutteredMeasurements.shape
    numSteps = min(numSteps, parameters['maxSteps'])
    numParticles = parameters['numParticles']
    detectionProbability = parameters['detectionProbability']
    priorMean = parameters['priorMean']
    survivalProbability = parameters['survivalProbability']
    undetectedAnchorsIntensity = parameters['undetectedAnchorsIntensity'] * np.ones(numSensors)
    birthIntensity = parameters['birthIntensity']
    clutterIntensity = parameters['clutterIntensity']
    unreliabilityThreshold = parameters['unreliabilityThreshold']
    execTimePerStep = np.zeros(numSteps)
    known_track = parameters['known_track']
    
    # allocate memory
    estimatedTrajectory = np.zeros((4, numSteps))
    numEstimatedAnchors = np.zeros((2, numSteps))
    storing_idx = np.arange(30, numSteps+1, 30)
    posteriorParticlesAnchorsstorage = [None]*len(storing_idx)
    
    # initial state vectors
    if known_track:
        posteriorParticlesAgent = np.tile(np.hstack((trueTrajectory[:, 0], [0, 0])), (numParticles, 1)).T
    else:
        posteriorParticlesAgent = np.zeros((4, numParticles))
        posteriorParticlesAgent[:2] = drawSamplesUniformlyCirc(priorMean[:2], parameters['UniformRadius_pos'], parameters['numParticles'])
        posteriorParticlesAgent[2:] = np.tile(priorMean[2:], (2, parameters['numParticles'])) + 2*parameters['UniformRadius_vel']*rand(2, parameters['numParticles']) - parameters['UniformRadius_vel']
    estimatedTrajectory[:, 0] = np.mean(posteriorParticlesAgent, axis=1)
    
    estimatedAnchors, posteriorParticlesAnchors = initAnchors(parameters, dataVA, numSteps, numSensors)
    for sensor in range(numSensors):
        numEstimatedAnchors[sensor, 0] = len(estimatedAnchors[sensor][0])
    
    for step in range(1, numSteps):
        # perfrom prediction step
        if known_track:
            predictedParticlesAgent = np.tile(np.hstack((trueTrajectory[:, step], [0, 0])), (numParticles, 1)).T
        else:
            predictedParticlesAgent = performPrediction(posteriorParticlesAgent, parameters)
        weightsSensors = np.full((numParticles, numSensors), np.nan)
        for sensor in range(numSensors):
            estimatedAnchors[sensor][step] = estimatedAnchors[sensor][step-1]
            measurements = clutteredMeasurements[step, sensor]
            numMeasurements = len(measurements)
            
            # predict undetected anchors intensity
            undetectedAnchorsIntensity[sensor] = undetectedAnchorsIntensity[sensor] * survivalProbability + birthIntensity
            
            # predict "legacy" anchors
            predictedParticlesAnchors, weightsAnchor = predictAnchors(posteriorParticlesAnchors[sensor], parameters)
            
            # create new anchors (one for each measurement)
            newParticlesAnchors, newInputBP = generateNewAnchors(measurements, undetectedAnchorsIntensity[sensor], predictedParticlesAgent, parameters)
            
            # predict measurements from anchors
            predictedMeasurements, predictedUncertainties, predictedRange = predictMeasurements(predictedParticlesAgent, predictedParticlesAnchors, weightsAnchor)[2]

            # compute association probabilities
            associationProbabilities, associationProbabilitiesNew, messagelhfRatios, messagesNew = calculateAssociationProbabilitiesGA(measurements, predictedMeasurements, predictedUncertainties, weightsAnchor, newInputBP, parameters)
            
            # perform message multiplication for agent state
            numAnchors = predictedParticlesAnchors.shape[2]
            weights = np.zeros((numParticles, numAnchors))
            for anchor in range(numAnchors):
                weights[:, anchor] = np.full(numParticles, 1 - detectionProbability)
                for measurement in range(numMeasurements):
                    measurementVariance = measurements[1, measurement]
                    factor = 1 / np.sqrt(2 * np.pi * measurementVariance) * detectionProbability / clutterIntensity
                    weights[:, anchor] += factor * messagelhfRatios[measurement, anchor] * np.exp(-1 / (2 * measurementVariance) * (measurements[0, measurement] - predictedRange[:, anchor]) ** 2)
                
                # compute anchor predicted existence probability
                predictedExistence = np.sum(weightsAnchor[:, anchor])
                
                aliveUpdate = np.sum(predictedExistence * 1 / numParticles * weights[:, anchor])
                deadUpdate = 1 - predictedExistence
                posteriorParticlesAnchors[sensor][anchor].posteriorExistence = aliveUpdate / (aliveUpdate + deadUpdate)
                
                # compute anchor belief
                idx_resampling = resampleSystematic(weights[:, anchor] / np.sum(weights[:, anchor]), numParticles)
                posteriorParticlesAnchors[sensor][anchor].x = predictedParticlesAnchors[:, idx_resampling[:numParticles], anchor]
                posteriorParticlesAnchors[sensor][anchor].w = posteriorParticlesAnchors[sensor][anchor].posteriorExistence / numParticles * np.ones(numParticles)
                
                estimatedAnchors[sensor][step][anchor].x = np.mean(posteriorParticlesAnchors[sensor][anchor].x, axis=1)
                estimatedAnchors[sensor][step][anchor].posteriorExistence = posteriorParticlesAnchors[sensor][anchor].posteriorExistence
                
                weights[:, anchor] = predictedExistence * weights[:, anchor] + deadUpdate
                weights[:, anchor] = np.log(weights[:, anchor])
                weights[:, anchor] = weights[:, anchor] - np.max(weights[:, anchor])
            
            numEstimatedAnchors[sensor, step] = len(estimatedAnchors[sensor][step])
            weightsSensors[:, sensor] = np.sum(weights, axis=1)
            weightsSensors[:, sensor] = weightsSensors[:, sensor] - np.max(weightsSensors[:, sensor])
            
            # update undetected anchors intensity
            undetectedAnchorsIntensity[sensor] = undetectedAnchorsIntensity[sensor] * (1 - parameters['detectionProbability'])
            
            # update new anchors
            for measurement in range(numMeasurements):
                posteriorParticlesAnchors[sensor].append(AssociatedAnchor())
                posteriorParticlesAnchors[sensor][numAnchors + measurement].posteriorExistence = messagesNew[measurement] * newParticlesAnchors[measurement].constant / (messagesNew[measurement] * newParticlesAnchors[measurement].constant + 1)
                posteriorParticlesAnchors[sensor][numAnchors + measurement].x = newParticlesAnchors[measurement].x
                posteriorParticlesAnchors[sensor][numAnchors + measurement].w = posteriorParticlesAnchors[sensor][numAnchors + measurement].posteriorExistence / numParticles
                estimatedAnchors[sensor][step].append(EstimatedAnchor(x=np.mean(newParticlesAnchors[measurement].x, axis=1), posteriorExistence=posteriorParticlesAnchors[sensor][numAnchors + measurement].posteriorExistence, generatedAt=step))
            
            # delete unreliable anchors
            estimatedAnchors[sensor][step], posteriorParticlesAnchors[sensor] = deleteUnreliableVA(estimatedAnchors[sensor][step], posteriorParticlesAnchors[sensor], unreliabilityThreshold)
            numEstimatedAnchors[sensor, step] = len(estimatedAnchors[sensor][step])
        
        weightsSensors = np.sum(weightsSensors, axis=1)
        weightsSensors = weightsSensors - np.max(weightsSensors)
        weightsSensors = np.exp(weightsSensors)
        weightsSensors = weightsSensors / np.sum(weightsSensors)
        
        if step in storing_idx:
            posteriorParticlesAnchorsstorage[storing_idx.index(step)] = deepcopy(posteriorParticlesAnchors)
        
        if known_track:
            estimatedTrajectory[:, step] = np.mean(predictedParticlesAgent, axis=1)
            posteriorParticlesAgent = predictedParticlesAgent
        else:
            estimatedTrajectory[:, step] = np.sum(predictedParticlesAgent * weightsSensors[:, np.newaxis], axis=0)
            posteriorParticlesAgent = predictedParticlesAgent[:, resampleSystematic(weightsSensors, numParticles)]
        
        # error output
        execTimePerStep[step] = time.time() - tic
        error_agent = calcDistance(trueTrajectory[:2, step], estimatedTrajectory[:2, step])
        print(f'Time instance: {step}')
        print(f'Number of Anchors Sensor 1: {numEstimatedAnchors[0, step]}')
        print(f'Number of Anchors Sensor 2: {numEstimatedAnchors[1, step]}')
        print(f'Position error agent: {error_agent}')
        print(f'Execution Time: {execTimePerStep[step]:.4f}')
        print('---------------------------------------------------\n')

    return estimatedTrajectory, estimatedAnchors, posteriorParticlesAnchorsstorage, numEstimatedAnchors

