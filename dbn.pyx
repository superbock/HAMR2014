#!/usr/bin/env python
# encoding: utf-8
"""
Copyright (c) Sebastian Böck <sebastian.boeck@jku.at>

This is a more or less quick and dirty implementation of the bar pointer model
with a dynamic Bayesian network (DBN) originally proposed in [1] and its
adaption (using GMMs for the observation models) proposed in [2].

Please note that the GMM stuff is not tuned for speed yet, so using many GMMs
may result in low performance.

References:

[1] N. Whiteley, A. T. Cemgil, and S. Godsill.
    Bayesian modelling of temporal structure in musical audio.
    In Proceedings of the 7th International Conference on Music Information
    Retrieval (ISMIR 2006), pages 29–34, Victoria, BC, Canada, October 2006.

[2] F. Krebs, S. Böck, and G. Widmer.
    Rhythmic pattern modeling for beat and downbeat tracking in musical audio.
    In Proceedings of the 14th International Society for Music Information
    Retrieval Conference (ISMIR 2013), pages 227–232, Curitiba, Brazil,
    November 2013.

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

    @property
    def num_states(self):
        """Number of states."""
        return len(self.pointers) - 1


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
    cdef public np.ndarray observations
    cdef public list gmms
    cdef public bint norm_observations
    # default values
    NORM_OBSERVATIONS = False
    LOG_PROBABILITY = False
    NORM_PROBABILITY = False
    MIN_PROBABILITY = 0.
    MAX_PROBABILITY = 1.

    def __init__(self, model, observations, num_states, num_bar_states,
                 norm_observations=NORM_OBSERVATIONS,
                 log_probability=LOG_PROBABILITY,
                 norm_probability=NORM_PROBABILITY,
                 min_probability=MIN_PROBABILITY,
                 max_probability=MAX_PROBABILITY):
        """
        Construct a observation model instance using Gaussian Mixture Models
        (GMMs).

        :param model:             load the fitted GMM(s) of the observation
                                  model from the given file
        :param observations:      time sequence of observed feature values
        :param num_states:        number of DBN states
        :param num_bar_states:    number of DBN bar states
        :param norm_observations: normalise the observations
        :param log_probability:   scale the probabilities logarithmically
        :param norm_probability:  normalize the probabilities
        :param min_probability:   minimum probability for the densities
        :param max_probability:   maximum probability for the densities

        """
        # load the model (i.e. the fitted GMMs)
        if model is not None:
            import cPickle
            with open(model, 'r') as f:
                self.gmms = cPickle.load(f)
        else:
            raise ValueError('model must be given')
        # convert the given observations to an contiguous array
        self.observations = np.ascontiguousarray(observations, dtype=np.float)
        # normalise the observations
        if norm_observations:
            self.observations /= np.max(self.observations)
        # save the given parameters
        self.norm_observations = norm_observations
        # generate the observation model
        self.observation_model(self.observations, num_states, num_bar_states,
                               norm_observations, log_probability,
                               norm_probability, min_probability,
                               max_probability)

    def observation_model(self, observations, num_states, num_bar_states,
                          norm_observations=NORM_OBSERVATIONS,
                          log_probability=LOG_PROBABILITY,
                          norm_probability=NORM_PROBABILITY,
                          min_probability=MIN_PROBABILITY,
                          max_probability=MAX_PROBABILITY):
        """
        Compute the observation probability densities using (a) GMM(s).

        :param observations:      time sequence of observed feature values
        :param num_states:        number of states
        :param num_bar_states:    number of bar states
        :param norm_observations: normalise the observations
        :param log_probability:   scale the probabilities logarithmically
        :param norm_probability:  normalize the probabilities
        :param min_probability:   minimum probability for the densities
        :param max_probability:   maximum probability for the densities

        """
        # counter, etc.
        num_observations = len(observations)
        num_gmms = len(self.gmms)
        # init observation densities
        densities = np.zeros((num_observations, num_gmms), dtype=np.float)
        # define the observation states
        for i, gmm in enumerate(self.gmms):
            # get the predictions of each GMM for the activations
            # Note: the GMMs return weird probabilities, at least exp(p) is
            #       somehow predictable, since it is > 0
            densities[:, i] = np.exp(gmm.score(observations))
        # scale the densities
        if log_probability:
            densities = np.log(densities + 1)
        if norm_probability:
            densities /= np.max(densities)
        densities = np.minimum(densities, max_probability)
        densities = np.maximum(densities, min_probability)

        # init the observation pointers
        pointers = np.zeros(num_states, dtype=np.uint32)
        # distribute the observation densities defined by the GMMs uniformly
        # across the entire state space
        for i in range(num_states):
            pointers[i] = ((i + num_bar_states) % num_bar_states) // num_gmms
        # save everything
        self.densities = densities
        self.pointers = pointers
        # return the arrays
        return densities, pointers


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
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline void _best_prev_state(self, int state, int frame,
                                      double [::1] current_viterbi,
                                      double [::1] prev_viterbi,
                                      double [:, ::1] om_densities,
                                      unsigned int [::1] om_pointers,
                                      unsigned int [::1] tm_states,
                                      unsigned int [::1] tm_pointers,
                                      double [::1] tm_probabilities,
                                      unsigned int [:, ::1] pointers) nogil:
        """
        Inline function to determine the best previous state.

        :param state:            current state
        :param frame:            current frame
        :param current_viterbi:  current viterbi variables
        :param prev_viterbi:     previous viterbi variables
        :param om_densities:     observation model densities
        :param om_pointers:      observation model pointers
        :param tm_states:        transition model states
        :param tm_pointers:      transition model pointers
        :param tm_probabilities: transition model probabilities
        :param pointers:         back tracking pointers

        """
        # define variables
        cdef unsigned int prev_state, pointer
        cdef double density, transition_prob
        # reset the current viterbi variable
        current_viterbi[state] = 0.0
        # get the observation model probability density value
        # the om_pointers array holds pointers to the correct observation
        # probability density value for the actual state (i.e. column in the
        # om_densities array)
        # Note: defining density here gives a 5% speed-up!?
        density = om_densities[frame, om_pointers[state]]
        # iterate over all possible previous states
        # the tm_pointers array holds pointers to the states which are
        # stored in the tm_states array
        for pointer in range(tm_pointers[state], tm_pointers[state + 1]):
            prev_state = tm_states[pointer]
            # weight the previous state with the transition
            # probability and the observation probability density
            transition_prob = prev_viterbi[prev_state] * \
                              tm_probabilities[pointer] * density
            # if this transition probability is greater than the
            # current, overwrite it and save the previous state
            # in the current pointers
            if transition_prob > current_viterbi[state]:
                current_viterbi[state] = transition_prob
                pointers[frame, state] = prev_state
            # # forward pass only:
            # current_viterbi[state] += transition_prob



    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def viterbi(self):
        """
        Determine the best path with the Viterbi algorithm.

        :return: best state-space path sequence and its log probability

        """
        # transition model stuff
        cdef TransitionModel tm = self.transition_model
        cdef unsigned int [::1] tm_states = tm.states
        cdef unsigned int [::1] tm_pointers = tm.pointers
        cdef double [::1] tm_probabilities = tm.probabilities
        cdef unsigned int num_states = tm.num_states

        # observation model stuff
        cdef ObservationModel om = self.observation_model
        cdef double [:, ::1] om_densities = om.densities
        cdef unsigned int [::1] om_pointers = om.pointers
        cdef unsigned int num_observations = len(om.densities)

        # current viterbi variables
        current_viterbi_np = np.empty(num_states, dtype=np.float)
        cdef double [::1] current_viterbi = current_viterbi_np

        # previous viterbi variables, init with the initial state distribution
        cdef double [::1] previous_viterbi = self.initial_states

        # back-tracking pointers
        cdef unsigned int [:, ::1] bt_pointers = np.empty((num_observations,
                                                           num_states),
                                                          dtype=np.uint32)
        # back tracked path, a.k.a. path sequence
        path = np.empty(num_observations, dtype=np.uint32)

        # define counters etc.
        cdef int state, frame
        cdef unsigned int prev_state, pointer, num_threads = self.num_threads
        cdef double obs, transition_prob, viterbi_sum, path_probability = 0.0

        # iterate over all observations
        for frame in range(num_observations):
            # range() is faster than prange() for 1 thread
            if num_threads == 1:
                # search for best transitions sequentially
                for state in range(num_states):
                    self._best_prev_state(state, frame, current_viterbi,
                                          previous_viterbi, om_densities,
                                          om_pointers, tm_states, tm_pointers,
                                          tm_probabilities, bt_pointers)
            else:
                # search for best transitions in parallel
                for state in prange(num_states, nogil=True, schedule='static',
                                    num_threads=num_threads):
                    self._best_prev_state(state, frame, current_viterbi,
                                          previous_viterbi, om_densities,
                                          om_pointers, tm_states, tm_pointers,
                                          tm_probabilities, bt_pointers)

            # overwrite the old states with the normalized current ones
            # Note: this is faster than unrolling the loop! But it is a bit
            #       tricky: we need to do the summing and normalisation on the
            #       numpy array but do the assignment on the memoryview
            viterbi_sum = current_viterbi_np.sum()
            previous_viterbi = current_viterbi_np / viterbi_sum
            # add the log sum of all viterbi variables to the overall sum
            path_probability += log(viterbi_sum)

        # fetch the final best state
        state = current_viterbi_np.argmax()
        # add its log probability to the sum
        path_probability += log(current_viterbi_np.max())
        # track the path backwards, start with the last frame and do not
        # include the pointer for frame 0, since it includes the transitions
        # to the prior distribution states
        for frame in range(num_observations -1, -1, -1):
            # save the state in the path
            path[frame] = state
            # fetch the next previous one
            state = bt_pointers[frame, state]
        # save the tracked path and log sum and return them
        self._path = path
        self.path_probability = path_probability
        return path, path_probability

    @property
    def path(self):
        """Best path sequence."""
        if self._path is None:
            self.viterbi()
        return self._path

    @property
    def bar_states_path(self):
        """Bar states path."""
        return self.path % self.transition_model.num_bar_states

    @property
    def tempo_states_path(self):
        """Tempo states path."""
        states = self.path / self.transition_model.num_bar_states
        return self.transition_model.tempo_states[states]
