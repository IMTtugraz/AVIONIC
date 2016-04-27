#!/usr/bin/env python
import sys
import os
import argparse
import utils as ut

#================================================================================
# Define parsing arguments
#================================================================================
parser = argparse.ArgumentParser()

parser.add_argument("-f", dest="filename", required=True,
                    help="bin input file", metavar="FILE")
parser.add_argument("-nx", dest="nx", type=int, required=True,
                    help="nx dim")
parser.add_argument("-ny", dest="ny", type=int, required=True,
                    help="ny dim")
parser.add_argument("-nframes", dest="nframes", type=int, required=True,
                    help="nframes dim")

args = parser.parse_args()

# Load file
if not os.path.isfile(args.filename):
    print("%s is not a valid file" % filename)
    raise SystemExit

print("processing ",args.filename)

recon=ut.readbin_vector(args.filename, args.nx,args.ny,args.nframes);
ut.show_recon(recon)
 
