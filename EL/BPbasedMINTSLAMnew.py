import numpy as np

def BPbasedMINTSLAMnew(dataVA, clutteredMeasurements, parameters, trueTrajectory):
    numSteps, numSensors = clutteredMeasurements.shape
    numSteps = min(numSteps, parameters["maxSteps"])
    numParticles = parameters["numParticles"]
    detectionProbability = parameters["detectionProbability"]
    priorMean = parameters["priorMean"]
    survivalProbability = parameters["survivalProbability"]
    undetectedAnchorsIntensity = parameters["undetectedAnchorsIntensity"] * np.ones((numSensors,))
    birthIntensity = parameters["birthIntensity"]
    clutterIntensity = parameters["clutterIntensity"]
    unreliabilityThreshold = parameters["unreliabilityThreshold"]
    execTimePerStep = np.zeros((numSteps,))
    known_track = parameters["known_track"]

    # allocate memory
    estimatedTrajectory = np.zeros((4, numSteps))
    numEstimatedAnchors = np.zeros((2, numSteps))
    storing_idx = np.arange(30, numSteps + 1, 30)
    posteriorParticlesAnchorsstorage = [None] * len(storing_idx)
    # initial state vectors
    if known_track:
        posteriorParticlesAgent = np.tile(np.hstack((trueTrajectory[:, 0], 0, 0)), (numParticles, 1)).T
    else:
        posteriorParticlesAgent = np.vstack((drawSamplesUniformlyCirc(priorMean[:2], parameters["UniformRadius_pos"], parameters["numParticles"]), 
                                             np.tile(priorMean[2:], (1, parameters["numParticles"])) + 2 * parameters["UniformRadius_vel"] * np.random.rand(2, parameters["numParticles"]) - parameters["UniformRadius_vel"]))
    estimatedTrajectory[:, 0] = np.mean(posteriorParticlesAgent, axis=1)

    estimatedAnchors, posteriorParticlesAnchors = initAnchors(parameters, dataVA, numSteps, numSensors)
    for sensor in range(numSensors):
        numEstimatedAnchors[sensor, 0] = estimatedAnchors[sensor][0].shape[1]

    ## main loop
    for step in range(1, numSteps):
        # perform prediction step
        if known_track:
            predictedParticlesAgent = np.tile(np.hstack((trueTrajectory[:, step], 0, 0)), (numParticles, 1)).T
        else:
            predictedParticlesAgent = performPrediction(posteriorParticlesAgent, parameters)

        weightsSensors = np.full((numParticles, numSensors), np.nan)
        for sensor in range(numSensors):
            estimatedAnchors[sensor][step] = estimatedAnchors[sensor][step - 1]
            measurements = clutteredMeasurements[step, sensor]
            numMeasurements = measurements.shape[1]

            # predict undetected anchors intensity
            undetectedAnchorsIntensity[sensor] = undetectedAnchorsIntensity[sensor] * survivalProbability + birthIntensity

            # predict "legacy" anchors
            predictedParticlesAnchors, weightsAnchor = predictAnchors(posteriorParticlesAnchors[sensor], parameters)

            # create new anchors (one for each measurement)
            newParticlesAnchors, newInputBP = generateNewAnchors(measurements, undetectedAnchorsIntensity[sensor], predictedParticlesAgent, parameters)

            # predict measurements from anchors
            predictedMeasurements, predictedUncertainties, predictedRange = predictMeasurements(predictedParticlesAgent, predictedParticlesAnchors, weightsAnchor)

