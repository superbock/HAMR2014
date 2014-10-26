#!/usr/bin/env python
# encoding: utf-8
"""
Copyright (c) Sebastian Böck <sebastian.boeck@jku.at>


"""
import numpy as np
from scipy.signal import argrelmin


# cython stuff
cimport numpy as np
cimport cython
from libc.math cimport log

# parallel processing stuff
from cython.parallel cimport prange
import multiprocessing as mp
NUM_THREADS = mp.cpu_count()


class ConductionTracking(object):
    """
    Conduction tracking with a dynamic Bayesian Network (DBN).

    """
    # some default values
    GMM_MODEL = None
    NUM_BEAT_STATES = 2400
    TEMPO_CHANGE_PROBABILITY = 0.1
    NORM_OBSERVATIONS = False
    MIN_BPM = 2
    MAX_BPM = 30

    def __init__(self, features, fps, gmm_model=GMM_MODEL):
        """
        Instantiate a conduction tracking object.

        :param features:  features as numpy array of file (handle)
        :param fps:       frame rate of the features
        :param gmm_model: list with fitted GMMs or file (handle)
        """
        self.gmm_model = gmm_model
        self.fps = fps

        # check the type of the given data
        if isinstance(features, np.ndarray):
            # use them directly
            self.features = features
        elif isinstance(features, (basestring, file)):
            # read from file or file handle
            self.features = np.load(features).astype(np.float)
        else:
            raise TypeError("wrong input data for features")
        # other variables
        self.densities = None
        self.path = None
        self.bar_start_positions = None

    def track(self, num_bar_states=NUM_BEAT_STATES,
              min_bpm=MIN_BPM, max_bpm=MAX_BPM, gmm_model=GMM_MODEL,
              tempo_change_probability=TEMPO_CHANGE_PROBABILITY,
              norm_observations=NORM_OBSERVATIONS):
        """
        Track the conduction with a dynamic Bayesian network.

        Parameters for the transition model:

        :param num_bar_states:           number of cells for one beat period
        :param tempo_change_probability: probability of a tempo change between
                                         two adjacent observations
        :param min_bpm:                  minimum tempo used for beat tracking
        :param max_bpm:                  maximum tempo used for beat tracking

        Parameters for the observation model:

        :param gmm_model:                load the fitted GMM model from the
                                         given file
        :param norm_observations:        normalise the observations

        :return:                         detected beat positions

        """
        # convert timing information to tempo spaces
        max_tempo = int(np.ceil(max_bpm * num_bar_states / (60. * self.fps)))
        min_tempo = int(np.floor(min_bpm * num_bar_states / (60. * self.fps)))
        tempo_states = np.arange(min_tempo, max_tempo)
        print tempo_states
        # transition model
        tm = TransitionModel(num_bar_states=num_bar_states,
                             tempo_states=tempo_states,
                             tempo_change_probability=tempo_change_probability)
        # observation model
        om = ObservationModel(gmm_model, self.features,
                              num_states=tm.num_states,
                              num_bar_states=tm.num_bar_states,
                              norm_observations=norm_observations)
        # init the DBN
        dbn = DBN(transition_model=tm, observation_model=om)
        # save some information (mainly for visualisation)
        self.densities = om.densities.astype(np.float)
        self.path = dbn.bar_states_path.astype(np.int)
        self.bar_start_positions = argrelmin(self.path, mode='wrap')[0] / \
                                   float(self.fps)
        # also return the bar start positions
        return self.bar_start_positions

    def write(self, filename):
        """
        Write the detected bar start positions to a file.

        :param filename: output file name or file handle

        """
        # open file if needed
        if isinstance(filename, basestring):
            f = fid = open(filename, 'w')
        else:
            f = filename
            fid = None
        # write the start positions
        f.writelines('%g\n' % e for e in self.bar_start_positions)
        # close the file if needed
        if fid:
            fid.close()

cdef class DBN(object):
    """
    Dynamic Bayesian network for beat tracking.

    """
    # define some variables which are also exported as Python attributes
    cdef public TransitionModel transition_model
    cdef public ObservationModel observation_model
    cdef public np.ndarray initial_states
    cdef public unsigned int num_threads
    cdef public double path_probability
    # hidden variable
    cdef np.ndarray _path


    def __init__(self, transition_model=None, observation_model=None,
                 initial_states=None, num_threads=NUM_THREADS):
        """
        Construct a new dynamic Bayesian network.

        :param transition_model:  TransitionModel instance or file
        :param observation_model: ObservationModel instance or observations
        :param initial_states:    initial state distribution; a uniform
                                  distribution is assumed if None is given
        :param num_threads:       number of parallel threads

        """
        # save number of threads
        self.num_threads = num_threads
        # transition model
        if isinstance(transition_model, TransitionModel):
            # already a TransitionModel
            self.transition_model = transition_model
        else:
            # instantiate a new or load an existing TransitionModel
            self.transition_model = TransitionModel(transition_model)
        num_states = self.transition_model.num_states
        # observation model
        if isinstance(observation_model, ObservationModel):
            # already a ObservationModel
            self.observation_model = observation_model
        else:
            # instantiate a new ObservationModel
            self.observation_model = ObservationModel(observation_model,
                                                      num_states)
        # initial state distribution
        if initial_states is None:
            self.initial_states = np.ones(num_states, dtype=np.float)
        else:
            self.initial_states = np.ascontiguousarray(initial_states,
                                                       dtype=np.float)

    @property
    def bar_states_path(self):
        """Bar states path."""
        return self.path % self.transition_model.num_bar_states

    @property
    def tempo_states_path(self):
        """Tempo states path."""
        states = self.path / self.transition_model.num_bar_states
        return self.transition_model.tempo_states[states]


cdef class TransitionModel(object):
    """
    Transition model for conduction tracking with a DBN.

    The transition model is defined similar to a scipy compressed sparse row
    matrix and holds all transition probabilities from one state to an other.

    All state indices for row state s are stored in
    states[pointers[s]:pointers[s+1]]
    and their corresponding probabilities are stored in
    probabilities[pointers[s]:pointers[s+1]].

    This allows for a parallel computation of the viterbi path.

    This class should be either used for loading saved transition models or
    being sub-classed to define a new transition model.

    """
    # define some variables which are also exported as Python attributes
    cdef public np.ndarray probabilities
    cdef public np.ndarray states
    cdef public np.ndarray pointers
    # hidden list with attributes to save/load
    cdef list attributes
    # define some variables which are also exported as Python attributes
    cdef public unsigned int num_bar_states
    cdef public np.ndarray tempo_states
    cdef public double tempo_change_probability

    def __init__(self, num_bar_states, tempo_states,
                 tempo_change_probability):
        """
        Construct a transition model instance suitable for conduction tracking.

        :param num_bar_states:           number of states for one bar period
        :param tempo_states:             array with tempo states (number of
                                         bar states to progress from one
                                         observation value to the next one)
        :param tempo_change_probability: probability of a tempo change from
                                         one observation to the next one

        """
        # compute transitions
        self.num_bar_states = num_bar_states
        self.tempo_states = np.ascontiguousarray(tempo_states, dtype=np.int32)
        self.tempo_change_probability = tempo_change_probability
        # compute the transition matrix
        self.transition_model(self.num_bar_states, self.tempo_states,
                              self.tempo_change_probability)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def transition_model(self, unsigned int num_bar_states,
                         int [::1] tempo_states,
                         double tempo_change_probability):
        """
        Compute the transition probabilities and store them in a sparse format.

        :param num_bar_states:           number of states for one bar period
        :param tempo_states:             array with tempo states (number of
                                         bar states to progress from one
                                         observation value to the next one)
        :param tempo_change_probability: probability of a tempo change from
                                         one observation to the next one
        :return:                         (probabilities, states, prev_states)

        """
        # number of tempo & total states
        cdef unsigned int num_tempo_states = len(tempo_states)
        cdef unsigned int num_states = num_bar_states * num_tempo_states
        # transition probabilities
        cdef double same_tempo_prob = 1. - tempo_change_probability
        cdef double change_tempo_prob = 0.5 * tempo_change_probability
        # counters etc.
        cdef unsigned int state, prev_state, bar_state, tempo_state, tempo
        # number of transition states
        # num_tempo_states * 3 because every state has a transition from the
        # same tempo and from the slower and faster one, -2 because the slowest
        # and the fastest tempi can't have transitions from outside the tempo
        # range
        cdef int num_transition_states = (num_bar_states *
                                          (num_tempo_states * 3 - 2))
        # arrays for transitions matrix creation
        cdef unsigned int [::1] states = \
            np.empty(num_transition_states, np.uint32)
        cdef unsigned int [::1] prev_states = \
            np.empty(num_transition_states, np.uint32)
        cdef double [::1] probabilities = \
            np.empty(num_transition_states, np.float)
        cdef int i = 0
        # loop over all states
        for state in range(num_states):
            # position inside bar & tempo
            bar_state = state % num_bar_states
            tempo_state = state / num_bar_states
            tempo = tempo_states[tempo_state]
            # for each state check the 3 possible transitions
            # previous state with same tempo
            # Note: we add num_bar_states before the modulo operation so
            #       that it can be computed in C (which is faster)
            prev_state = ((bar_state + num_bar_states - tempo) %
                          num_bar_states +
                          (tempo_state * num_bar_states))
            # probability for transition from same tempo
            states[i] = state
            prev_states[i] = prev_state
            probabilities[i] = same_tempo_prob
            i += 1
            # transition from slower tempo
            if tempo_state > 0:
                # previous state with slower tempo
                prev_state = ((bar_state + num_bar_states -
                               (tempo - 1)) % num_bar_states +
                              ((tempo_state - 1) * num_bar_states))
                # probability for transition from slower tempo
                states[i] = state
                prev_states[i] = prev_state
                probabilities[i] = change_tempo_prob
                i += 1
            # transition from faster tempo
            if tempo_state < num_tempo_states - 1:
                # previous state with faster tempo
                # Note: we add num_bar_states before the modulo operation
                #       so that it can be computed in C (which is faster)
                prev_state = ((bar_state + num_bar_states -
                               (tempo + 1)) % num_bar_states +
                              ((tempo_state + 1) * num_bar_states))
                # probability for transition from faster tempo
                states[i] = state
                prev_states[i] = prev_state
                probabilities[i] = change_tempo_prob
                i += 1
        # save them in sparse format
        from scipy.sparse import csr_matrix
        # convert everything into a sparse CSR matrix
        transitions = csr_matrix((probabilities, (states, prev_states)))
        # save the sparse matrix as 3 linear arrays
        self.states = transitions.indices.astype(np.uint32)
        self.pointers = transitions.indptr.astype(np.uint32)
        self.probabilities = transitions.data.astype(dtype=np.float)
        # return the arrays
        return probabilities, states, prev_states


cdef class ObservationModel(object):
    """
    Observation model for GMM based conduction tracking with a DBN.

    An observation model is defined as two plain numpy arrays, densities and
    pointers.

    The 'densities' is a 2D numpy array with the number of rows being equal
    to the length of the observations and the columns representing the
    different observation probability densities. The type must be np.float.

    The 'pointers' is a 1D numpy array and has a length equal to the number of
    states of the DBN and points from each state to the corresponding column
    of the 'densities' array. The type must be np.uint32.

    """
    # define some variables which are also exported as Python attributes
    cdef public np.ndarray densities
    cdef public np.ndarray pointers
    # default values for conduction tracking
    NORM_OBSERVATIONS = True

    def __init__(self, model, activations, num_states, num_bar_states,
                 norm_observations=NORM_OBSERVATIONS):
        """
        Construct a observation model instance using Gaussian Mixture Models
        (GMMs).

        :param model:             load the fitted GMM(s) of the observation
                                  model from the given file
        :param activations:       time sequence of feature values
        :param num_states:        number of DBN states
        :param num_bar_states:    number of DBN bar states
        :param norm_observations: normalise the observations

        """
        # load the model (i.e. the fitted GMMs)
        if model is not None:
            import cPickle
            with open(model, 'r') as f:
                self.gmms = cPickle.load(f)
        else:
            raise ValueError('model must be given')
        # convert the given activations to an contiguous array
        self.activations = np.ascontiguousarray(activations, dtype=np.float)
        # normalise the activations
        if norm_observations:
            self.activations /= np.max(self.activations)
        # save the given parameters
        self.norm_observations = norm_observations
        # generate the observation model
        self.densities = None
        self.pointers = None
        self.observation_model(self.activations, num_states, num_bar_states)

    def observation_model(self, activations, num_states, num_bar_states,
                          min_probability=0.3, max_probability=0.7):
        """
        Compute the observation probability densities using (a) GMM(s).

        :param activations:     time sequence of feature values
        :param num_states:      number of states
        :param num_bar_states:  number of bar states
        :param min_probability: minimum probability for the densities
        :param max_probability: maximum probability for the densities

        """
        # counter, etc.
        num_observations = len(activations)
        num_gmms = len(self.gmms)
        # init observation densities
        densities = np.zeros((num_observations, num_gmms), dtype=np.float)
        # define the observation states
        for i, gmm in enumerate(self.gmms):
            # get the predictions of each GMM for the activations
            # Note: the GMMs return weird probabilities, at least exp(p) is
            #       somehow predictable, since it is > 0
            densities[:, i] = np.exp(gmm.score(activations))
        # scale the densities
        densities = np.log(densities + 1)
        densities /= np.max(densities)
        densities = np.maximum(max_probability * densities, min_probability)

        # init the observation pointers
        pointers = np.zeros(num_states, dtype=np.uint32)
        # distribute the observation densities defined by the GMMs uniformly
        # across the entire state space
        for i in range(num_states):
            pointers[i] = ((i + num_bar_states) % num_bar_states) // num_gmms
        # save everything
        self.densities = densities
        self.pointers = pointers


def parser():
    """
    Create a parser and parse the arguments.

    :return: the parsed arguments

    """
    import sys
    import argparse

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    If invoked without any parameters, the software tries to determine where
    a conductor wants his orchestra to start a measure bar.

    ''')

    # input/output options
    # general options
    p.add_argument('input', type=argparse.FileType('r'),
                   help='input feature file')
    p.add_argument('output', nargs='?', type=argparse.FileType('w'),
                   default=sys.stdout, help='output (file) [default: STDOUT]')
    # add arguments for DBNs
    g = p.add_argument_group('dynamic Bayesian Network arguments')
    # add a transition parameters
    g.add_argument('--num_bar_states', action='store', type=int,
                   default=ConductionTracking.NUM_BEAT_STATES,
                   help='number of states for one bar period '
                        '[default=%(default)i]')
    g.add_argument('--min_bpm', action='store', type=float,
                   default=ConductionTracking.MIN_BPM,
                   help='minimum tempo [bpm, default=%(default).2f]')
    g.add_argument('--max_bpm', action='store', type=float,
                   default=ConductionTracking.MAX_BPM,
                   help='maximum tempo [bpm,  default=%(default).2f]')
    g.add_argument('--tempo_change_probability', action='store', type=float,
                   default=ConductionTracking.TEMPO_CHANGE_PROBABILITY,
                   help='probability of a tempo between two adjacent '
                        'observations [default=%(default).4f]')
    # observation model stuff
    g.add_argument('--gmm_model', action='store', type=str,
                   default=ConductionTracking.GMM_MODEL,
                   help='Fitted GMM models')
    g.add_argument('--norm_obs', dest='norm_observations', action='store_true',
                   default=ConductionTracking.NORM_OBSERVATIONS,
                   help='normalize the observations of the DBN')
    p.add_argument('--fps', action='store', type=int, default=15,
                   help='frames per second [default=%(default)i]')
    # version
    p.add_argument('--version', action='version', version='ConductionTracker')
    # parse arguments
    args = p.parse_args()
    # print arguments
    if args.verbose:
        print args
    # return
    return args


def main():
    """
    Simple ConductionTracker which reads in the motion features from a file
    and determines the starting position of the bars."""

    # parse arguments
    args = parser()

    # load features
    t = ConductionTracking(args.input, fps=args.fps, gmm_model=args.gmm_model)
    # track it
    t.track(num_bar_states=args.num_bar_states, gmm_model=args.gmm_model,
            tempo_change_probability=args.tempo_change_probability,
            min_bpm=args.min_bpm, max_bpm=args.max_bpm,
            norm_observations=args.norm_observations)

    # plot it
    import matplotlib.pyplot as plt
    plt.imshow(t.densities.T, aspect='auto', interpolation='none',
               origin='lower')
    plt.plot(60 * t.path / np.max(t.path), 'w-')
    plt.colorbar()
    plt.show()

    # save detections
    t.write(args.output)

if __name__ == '__main__':
    main()
