import numpy as np
import scipy.io as sio
from generateMeasurements import generateMeasurements
from generateClutteredMeasurements import generateClutteredMeasurements
from BPbasedMINTSLAMnew import BPbasedMINTSLAMnew
from plotAll import plotAll

# clear variables
import gc
gc.collect()

# general parameters
parameters = {'known_track': 0}
data = sio.loadmat('scenarioCleanM2_new.mat')
dataVA = data['dataVA'][0]

# cast visibilities to 1
for sensor in range(len(dataVA)):
    dataVA[sensor]['visibility'] = np.ones((dataVA[sensor]['visibility'].shape[0], len(dataVA[sensor]['visibility'][0])), dtype=int)

# algorithm parameters
parameters['maxSteps'] = 900
trueTrajectory = data['trueTrajectory'][:, :parameters['maxSteps']]
parameters['lengthStep'] = 0.03
parameters['scanTime'] = 1
v_max = parameters['lengthStep'] / parameters['scanTime']
parameters['drivingNoiseVariance'] = (v_max / 3 / parameters['scanTime']) ** 2
parameters['measurementVariance'] = .1 ** 2
parameters['measurementVarianceLHF'] = .15 ** 2
parameters['detectionProbability'] = 0.95
parameters['regionOfInterestSize'] = 30
parameters['meanNumberOfClutter'] = 1
parameters['clutterIntensity'] = parameters['meanNumberOfClutter'] / parameters['regionOfInterestSize']
parameters['meanNumberOfBirth'] = 10 ** (-4)
parameters['birthIntensity'] = parameters['meanNumberOfBirth'] / (2 * parameters['regionOfInterestSize']) ** 2
parameters['meanNumberOfUndetectedAnchors'] = 6
parameters['undetectedAnchorsIntensity'] = parameters['meanNumberOfUndetectedAnchors'] / (2 * parameters['regionOfInterestSize']) ** 2
parameters['numParticles'] = 100000
parameters['upSamplingFactor'] = 1

# SLAM parameters
parameters['detectionThreshold'] = 0.5
parameters['survivalProbability'] = 0.999
parameters['unreliabilityThreshold'] = 1e-4
parameters['priorKnownAnchors'] = [1, 1]
parameters['priorCovarianceAnchor'] = .001 ** 2 * np.eye(2)
parameters['anchorRegularNoiseVariance'] = 1e-4 ** 2

# agent parameters
parameters['UniformRadius_pos'] = .5
parameters['UniformRadius_vel'] = .05

# random seed
np.random.seed(1)

# draw starting position
parameters['priorMean'] = np.hstack((trueTrajectory[0:2, 0], np.array([0, 0])))

# generate measurements
measurements = generateMeasurements(trueTrajectory, dataVA, parameters)

# generate cluttered measurements
clutteredMeasurements = generateClutteredMeasurements(measurements, parameters)

# perform estimation with data association uncertainty
(estimatedTrajectory, estimatedAnchors, posteriorParticlesAnchors, numEstimatedAnchors) = BPbasedMINTSLAMnew(dataVA, clutteredMeasurements, parameters, trueTrajectory)

# plot results
plotAll(trueTrajectory, estimatedTrajectory, estimatedAnchors, posteriorParticlesAnchors[-1], numEstimatedAnchors, dataVA, parameters, 0, parameters['maxSteps'])

