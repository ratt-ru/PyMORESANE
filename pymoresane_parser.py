import argparse
import re

def parser1(args=None):
    """
    This function reparses values into separate namespaces.
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
                                        "detections.", default=4, type=int)

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

    parser.add_argument("-adv", "--advancedoptions", help="Additional help for advanced functions.", action="store_true")

    parser.add_argument("-vis", "--usevis", help="Specify whether model subtraction must be done on the visibilities"
                                                 ".", action="store_true")

    if args is None:
            return parser.parse_known_args()
    else:
            return parser.parse_known_args(args)

def parser2(args=None):
    """
    This function parses the arguments for lwimager.
    """
    parser = argparse.ArgumentParser(description="Provides arguments for lwimager.",
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--ms", help="File name and location of the input MS folder.")

    parser.add_argument("--image", help="File name and location of the output .img file.")

    parser.add_argument("--operation", help="Operation which lwimager is expected to perform.", default="image")

    parser.add_argument("--data", help="Column in the MS to be used.", default="CORRECTED_DATA")

    parser.add_argument("--cachesize", help="??", default="4096")

    parser.add_argument("--mode", help="Mode of operation.", default="channel")

    parser.add_argument("--weight", help="Weighting scheme for visibilities.", default="briggs")

    parser.add_argument("--robust", help="Compromise level between natural and uniform weighting.", default="0")

    parser.add_argument("--padding", help="Padding mode.", default="2")

    parser.add_argument("--wprojplanes", help="Number of planes for w-projection.", default="128")

    parser.add_argument("--stokes", help="Stokes axis.", default="I")

    parser.add_argument("--field", help="??", default="0")

    parser.add_argument("--spwid", help="Spectral windows to image.", default="0")

    parser.add_argument("--chanstart", help="Starting channel.", default="0")

    parser.add_argument("--nchan", help="Number of channels.", default="8")

    parser.add_argument("--chanstep", help="Step between channels.", default="1")

    parser.add_argument("--img_chanstart", help="Start channel for image.", default="0")

    parser.add_argument("--img_nchan", help="Number of channels for image.", default="1")

    parser.add_argument("--img_chanstep", help="Step between channels for image.", default=8)

    parser.add_argument("--npix", help="Pixel dimension.", default="2048")

    parser.add_argument("--cellsize", help="Angular size per pixel.", default="1arcsec")

    if args is None:
            return parser.parse_known_args()
    else:
            return parser.parse_known_args(args)

def make_lwcommand(instr):
    """
    This function takes the additional arguments required by lwimager and reformats them into strings that
    can be run directly from the system terminal.
    """
    dirty_command = str(instr).split("(")[1]
    dirty_command = dirty_command.strip(")")
    dirty_command = dirty_command.replace("'","")
    dirty_command = dirty_command.replace(",","")

    dirty_command = 'lwimager ' + dirty_command

    psf_command = dirty_command
    piece = re.compile('data=[a-zA-Z_]*')
    chunk = piece.search(psf_command).span()
    psf_command = psf_command.replace(psf_command[chunk[0]:chunk[1]], 'data=psf')
    piece = re.compile('wprojplanes=[0-9]*.*')
    chunk = piece.search(psf_command).span()
    psf_command = psf_command.replace(psf_command[chunk[0]:chunk[1]], '')

    residual_command = dirty_command
    piece = re.compile('data=[a-zA-Z_]*')
    chunk = piece.search(residual_command).span()
    residual_command = residual_command.replace(residual_command[chunk[0]:chunk[1]], 'data=residual')

    tovis_command = dirty_command
    unnecessary_args = ['weight','robust','npix','cellsize','data','image']
    for i in unnecessary_args:
            piece = re.compile(i+'=[0-9a-zA-Z_]* ')
            chunk = piece.search(tovis_command).span()
            tovis_command = tovis_command.replace(tovis_command[chunk[0]:chunk[1]], '')

    piece = re.compile('operation=[a-zA-Z_]*')
    chunk = piece.search(tovis_command).span()
    tovis_command = tovis_command.replace(tovis_command[chunk[0]:chunk[1]], 'operation=csclean')
    tovis_command += ' niter=0'
    tovis_command += ' fixed=0'
    tovis_command += ' model=model.img'

    return dirty_command, residual_command, psf_command, tovis_command

def handle_parser():
    args1, remaining_args = parser1()
    if args1.advancedoptions:
            remaining_args.append('-h')
    args2, remaining_args = parser2(remaining_args)

    return args1, args2
