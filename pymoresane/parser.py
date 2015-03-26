import argparse


def handle_parser():
    """
    This function parses in values from command line, allowing for user control from the system terminal.
    """
    parser = argparse.ArgumentParser(description="Runs the pymoresane deconvolution algorithm with the specified "
                                                 "arguments. In the event that non-critical parameters are missing, "
                                                 "the defaults will be used.",
                                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("dirty", help="File name and location of the input dirty map .fits file.")

    parser.add_argument("psf", help="File name and location input psf .fits file.")

    parser.add_argument("outputname", help="File name and location of the output model and residual .fits files.")

    parser.add_argument("-sr", "--singlerun", help="Specify whether pymoresane is to be run in scale-by-scale mode or "
                                                   "in single-run mode. Scale-by-scale is usually the better choice."
                                                   , action="store_true")

    parser.add_argument("-sbr", "--subregion", help="Specify pixel width of the central region of the dirty .fits "
                                                   "which is to be deconvolved.", default=None, type=int)

    parser.add_argument("-sc", "--scalecount", help="Specify the maximum number of wavelet scales which the algorithm "
                                                    "is to consider. Only applies in single-run mode. For "
                                                    "scale-by-scale mode, use --startscale and --stopscale"
                                                    , default=None,type=int)

    parser.add_argument("-sts", "--startscale", help="Specify first scale to consider in the scale-by-scale case. "
                                                     "Should be 1 in almost all cases.", default=1, type=int)

    parser.add_argument("-sps", "--stopscale", help="Specify last scale to consider in the scale-by-scale case. "
                                                    "Can be safely omitted if all scales are to be considered."
                                                    , default=20, type=int)

    parser.add_argument("-sl", "--sigmalevel", help="Specify the sigma level at which the thresholding of the wavelet "
                                                    "coefficients is to be performed. May be used to reduce false "
                                                    "detections.", default=4, type=float)

    parser.add_argument("-lg", "--loopgain", help="Specify the loop gain for the deconvolution step."
                                                  , default=0.2, type=float)

    parser.add_argument("-tol", "--tolerance", help="Specify the percentage of the maximum wavelet coefficient which "
                                                    "is to be used as a threshold for significance at each iteration."
                                                    , default=0.75, type=float)

    parser.add_argument("-ac", "--accuracy", help="Specify the relative change in the STD of the residual that is "
                                                  "considered sufficient as a stopping criteria."
                                                  , default=1e-6, type=float)

    parser.add_argument("-malm", "--majorloopmiter", help="Specify the maximum number of major loop iterations."
                                                          , default=100, type=int)

    parser.add_argument("-milm", "--minorloopmiter", help="Specify the maximum number of minor loop iterations."
                                                          , default=50, type=int)

    parser.add_argument("-aog", "--allongpu", help="Specify whether as much code as possible is to be executed on the "
                                                   "gpu. Overrides the behaviour of all other gpu options"
                                                   , action='store_true')

    parser.add_argument("-dm", "--decommode", help="Specify whether wavelet decompositions are to performed using a "
                                                   "single CPU core, multiple CPU cores or the GPU."
                                                   , default="ser", choices=["ser","mp","gpu"])

    parser.add_argument("-cc", "--corecount", help="Specify the number of CPU cores to be used in the event that "
                                                   "multiprocessing is enabled. This might not improve performance."
                                                   , default=1, type=int)

    parser.add_argument("-cd", "--convdevice", help="Specify whether convolutions are to performed using the CPU or "
                                                    "the GPU.", default="cpu", choices=["cpu","gpu"])

    parser.add_argument("-cm", "--convmode", help="Specify convolution is to be circular or linear."
                                                  , default="circular", choices=["circular","linear"])

    parser.add_argument("-em", "--extractionmode", help="Specify whether source extraction is to be performed "
                                                        "using the CPU or the GPU."
                                                        , default="cpu", choices=["cpu","gpu"])

    parser.add_argument("-ep", "--enforcepositivity", help="Specify whether or not the model is constrained to be "
                                                           "strictly positive. Will massively improve the model in "
                                                           "simulated cases, but may cause problems with real data."
                                                           , action="store_true")

    parser.add_argument("-es", "--edgesuppression", help="Specify whether or not the edge-corrupted wavelets are to "
                                                         "be suppressed. Will decrease sensitivity along data edges."
                                                         , action="store_true")

    parser.add_argument("-eo", "--edgeoffset", help="Specify an additional offset along the edges. May reduce false "
                                                    "detections caused by convolution artifacts."
                                                    , default=0, type=int)

    parser.add_argument("-ll", "--loglevel", help="Specify logging level.", default="INFO"
                                                  , choices=["DEBUG","INFO", "WARNING", "ERROR","CRITICAL"])

    parser.add_argument("-m", "--mask", help="File name and location of the input .fits mask.", default=None)

    parser.add_argument("-ft", "--fluxthreshold", help="Flux threshold level for shallow deconvolution.", default=0,
                        type=float)

    parser.add_argument("-rn", "--residualname", help="Specific residual image name.", default=None)

    parser.add_argument("-mn", "--modelname", help="Specific model image name.", default=None)

    parser.add_argument("-rsn", "--restoredname", help="Specific restored image name.", default=None)


    return parser.parse_args()
