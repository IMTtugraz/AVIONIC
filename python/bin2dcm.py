#!/usr/bin/env python
import sys
import os
import argparse
import utils as ut
import numpy as np
import datetime, time
import glob

import dicom, dicom.UID
from dicom.dataset import Dataset, FileDataset




def write_dicom(pixel_array,filename):
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = 'Secondary Capture Image Storage'
    file_meta.MediaStorageSOPInstanceUID = '1.3.6.1.4.1.9590.100.1.1.111165684411017669021768385720736873780'
    file_meta.ImplementationClassUID = '1.3.6.1.4.1.9590.100.1.0.100.4.0'
    ds = FileDataset(filename, {},file_meta = file_meta,preamble="\0"*128)
    ds.Modality = 'WSD'
    ds.ContentDate = str(datetime.date.today()).replace('-','')
    ds.ContentTime = str(time.time()) #milliseconds since the epoch
    ds.StudyInstanceUID =  '1.3.6.1.4.1.9590.100.1.1.124313977412360175234271287472804872093'
    ds.SeriesInstanceUID = '1.3.6.1.4.1.9590.100.1.1.369231118011061003403421859172643143649'
    ds.SOPInstanceUID =    '1.3.6.1.4.1.9590.100.1.1.111165684411017669021768385720736873780'
    ds.SOPClassUID = 'Secondary Capture Image Storage'
    ds.SecondaryCaptureDeviceManufctur = 'Python 2.7.3'
    
    ## These are the necessary imaging components of the FileDataset object.
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.HighBit = 15
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.SmallestImagePixelValue = '\\x00\\x00'
    ds.LargestImagePixelValue = '\\xff\\xff'
    ds.Columns = pixel_array.shape[0]
    ds.Rows = pixel_array.shape[1]
    if pixel_array.dtype != np.uint16:
        pixel_array = pixel_array.astype(np.uint16)
    ds.PixelData = pixel_array.tostring()
    
    ds.save_as(filename)        
    
#================================================================================
# Define parsing arguments
#================================================================================
parser = argparse.ArgumentParser()

parser.add_argument("-d", dest="directory", required=True,
                    help="bin input file", metavar="FILE")
#parser.add_argument("-nx", dest="nx", type=int, required=True,
#                    help="nx dim")
#parser.add_argument("-ny", dest="ny", type=int, required=True,
#                    help="ny dim")
#parser.add_argument("-nframes", dest="nframes", type=int, required=True,
#                    help="nframes dim")

args = parser.parse_args()

print args.directory
#directory = "/home2/ictgv/RAVE/data/dce_angio_head/meas_MID00142_FID03779_RAVE_SPF8_SLIDWIN0_ICTGV2/"
# Get information
metafile = glob.glob(args.directory + "*meta*");

vars = dict()

with open(metafile[0]) as f:
    for line in f:
        eq_index = line.find('=')
        if eq_index > 0:
          var_name = line[:eq_index].strip()
          number = float(line[eq_index + 1:].strip())
          vars[var_name] = number

print(vars)

eNx = int(vars["eNx"]);
eNy = int(vars["eNy"]);
nframes = int(vars["nframes"]);
nparts = int(vars["nparts"]);


files = glob.glob(args.directory + "*ICTGV*");
filestem = files[0];
filestem = filestem[0:-6];

print filestem

for slice in range(nparts-1):
    filetoread = filestem + str(slice+1) + ".bin"
    if os.path.isfile(filetoread):   
      print "processing " + filetoread  
      recon = ut.readbin_vector(filetoread, eNx,eNy,nframes)
      
      # normalize 
      recon = np.uint16(np.abs(recon)*(2**12)/np.abs(recon.max()));
      
      # write to dicom
      for frame in range(nframes-1):
        write_dicom(recon[:,:,frame],'series'+str(frame+1)+'.slice'+str(slice+1)+'.dcm')

# Load file
#if not os.path.isfile(args.filename):
#    print("%s is not a valid file" % filename)
#    raise SystemExit


# read reconstruction


#
