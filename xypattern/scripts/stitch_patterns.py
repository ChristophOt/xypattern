import argparse
from .. import Pattern
from ..combine import stitch_patterns, scale_patterns


def main():
    parser = argparse.ArgumentParser(description='Stitch patterns together',
                                     prog='stitch_patterns.py')
    parser.add_argument('patterns', metavar='pattern', type=str, nargs='+',
                        help='patterns to be stitched together')
    parser.add_argument('-o', '--output', metavar='output', type=str, nargs=1,
                        help='output file name')
    parser.add_argument('-b', '--binning', metavar='binning', type=float, nargs=1,
                        help='binning to be applied to the stitched pattern, if None, the binning of the first pattern '
                             'will be used')
    parser.add_argument('-s', '--scale', action='store_true',
                        help='scale patterns to the first pattern in respect to x')
    args = parser.parse_args()

    patterns = [Pattern.from_file(pattern) for pattern in args.patterns]
    if args.scale:
        scale_patterns(patterns)

    pattern = stitch_patterns(patterns, binning=args.binning[0] if args.binning else None)
    pattern.save(args.output[0] if args.output else 'stitched.xy')
