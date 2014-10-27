#!/usr/bin/env python
# encoding: utf-8
"""
Copyright (c) Sebastian Böck <sebastian.boeck@jku.at>

Quick and dirty hack done for conduction tracking of motion features loaded
from a file using the bar pointer model [1] and a DBN with fitted GMMs as in
[2].

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

from dbn import DBN, TransitionModel, ObservationModel


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
        # transition model
        tm = TransitionModel(num_bar_states=num_bar_states,
                             tempo_states=tempo_states,
                             tempo_change_probability=tempo_change_probability)
        # observation model
        om = ObservationModel(gmm_model, self.features,
                              num_states=tm.num_states,
                              num_bar_states=tm.num_bar_states,
                              norm_observations=norm_observations,
                              log_probability=True, norm_probability=True,
                              min_probability=0.3, max_probability=0.7)
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
    print t.densities.max()
    print t.densities.min()
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
