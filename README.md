HAMR2014
========

Quick and dirty hack done during HAMR@ISMIR 2014 for conduction tracking of
motion features loaded from a file using the bar pointer model [1] and a DBN
with fitted GMMs as in [2].

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

Run python `setup.py build_ext --inplace` to build the `dbn` module.

Required packages:
- python 2.7
- cython
- numpy
- scipy
- gcc

Run `export CC=gcc` to switch to gcc if you get OpenMP related errors.
