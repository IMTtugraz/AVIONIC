from numpy import linalg as LA
import numpy as np, h5py 
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import scipy.io as sio
import struct
from PIL import Image # Notice the 'from PIL' at the start of the line
import os

def list_files(path):
    # returns a list of names (with extension, without full path) of all files 
    # in folder path
    files = []
    for name in os.listdir(path):
        if os.path.isfile(os.path.join(path, name)):
            files.append(name)
    return files 



def readbin_vector(filename,nx,ny,nframes):
  "READBIN_VECTOR reads a (column) vector from a binary file"
  print("loading " + filename)
  with open(filename) as f:
    v=f.read()
    f.close()
  
  iscomplex=struct.unpack("B",v[0]);
  size=struct.unpack("I",v[1:5]);
  dl=size[0]*(iscomplex[0]+1);
  data=struct.unpack('d'*dl , v[5:len(v)]) 
  dt=np.array(data,dtype=np.float32)
  if iscomplex[0]:
     dt=np.vectorize(complex)(dt[0:size[0]],dt[size[0]:])  
  ncoils=dt.size/(nx*ny*nframes);
  #print "dim1=",str((ncoils*nx*ny*nframes))," | dim2=",str(size[0])
  if ((ncoils*nx*ny*nframes)==size[0]):
    if ncoils > 1:
      dt = np.transpose(dt.reshape((ncoils,nx,ny,1)),(1,2,0,3));
      dt = np.squeeze(dt);
      #dt = np.flipud(dt);
    else:
      dt = np.transpose(dt.reshape((nframes,nx,ny,ncoils)),(1,2,0,3));
      dt = np.squeeze(dt);
      #dt = np.flipud(dt);
  return dt


def show_all(Filename,nx,ny,nframes):
    "SHOW_ALL displays all results recon/components/b1/pdgap" 
 
    fig = plt.figure()
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)

    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)
 
    # Reconstruction
    recon = readbin_vector(Filename, nx,ny,nframes); 
    ims = []
    imsp = []
    imgDim = list(recon.shape)
    for i in range(0,imgDim[2]):
      im = ax1.imshow(abs(recon[:,:,i]),cmap='Greys_r')
      ims.append([im])
    for i in range(0,imgDim[2]):
      im = ax4.imshow(np.angle(recon[:,:,i]),cmap='jet')
      imsp.append([im]) 
    ani1 = animation.ArtistAnimation(fig, ims, interval=100, blit=True,repeat_delay=1000)
    ax1.set_title("Reconstruction (mag.)") 
    ani1p = animation.ArtistAnimation(fig, imsp, interval=100, blit=True,repeat_delay=1000)
    ax4.set_title("Reconstruction (phs.)") 
 

    # Components
    if os.path.isfile(Filename[:-4]+"_comp.bin"):
      comp1 = readbin_vector(Filename[:-4]+"_comp.bin", nx,ny,nframes);
      comp2 = recon-comp1;
      ims2 = []
      ims3 = []
      for i in range(0,imgDim[2]):
        im2 = ax6.imshow(abs(comp1[:,:,i]),cmap='Greys_r')
        ims2.append([im2])
      for i in range(0,imgDim[2]):
        im3 = ax5.imshow(abs(comp2[:,:,i]),cmap='Greys_r')
        ims3.append([im3])
      ani2 = animation.ArtistAnimation(fig, ims2, interval=100, blit=True,repeat_delay=1000)
      ani3 = animation.ArtistAnimation(fig, ims3, interval=100, blit=True,repeat_delay=1000)
      ax5.set_title("1st Component") 
      ax6.set_title("2nd Component") 
 
    # Coils
    if os.path.isfile(Filename[:-4]+"_b1.bin"):
      b1 = readbin_vector(Filename[:-4]+"_b1.bin", nx,ny,1);
      coilDim = list(b1.shape)
      ims4 = []
      for i in range(0,coilDim[2]):
        im4 = ax2.imshow(abs(b1[:,:,i]),cmap='Greys_r')
        ims4.append([im4])
      ani4 = animation.ArtistAnimation(fig, ims4, interval=100, blit=True,repeat_delay=1000)
      ax2.set_title("estimated B1") 
 
    # PDgap
    if os.path.isfile(Filename[:-4]+"_pdgap.bin"):
      pdgap = readbin_vector(Filename[:-4]+"_pdgap.bin",nx,ny,nframes );
      ax3.plot()
      imgplot = ax3.semilogy(abs(pdgap),'-ro')
      ax3.grid(b=True, which='minor', color='r', linestyle='--')
      ax3.set_title("PD Gap") 
    plt.show()

def show_recon(dataFile):
    "SHOW_RECON displays video/image" 
    img=dataFile;
    imgDim = list(img.shape)
    if len(imgDim) > 2:
        fig = plt.figure()
        ims = []
        for i in range(0,imgDim[2]):
            im = plt.imshow(abs(img[:,:,i]),cmap='Greys_r')
            ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
        repeat_delay=1000)

    elif len(imgDim) == 2:
        imgplot = plt.imshow(abs(img));#, cmap=cm.gray)

    elif len(imgDim) == 1:
         imgplot = plt.semilogy(abs(img),'-ro')
         plt.grid(b=True, which='minor', color='r', linestyle='--')
    plt.show()

def exportpng(filename,dataFile):
    img = dataFile;
    imgDim = list(img.shape)
    for i in range(0,imgDim[2]):
      data = img[:,:,i];
      #print "abs.data=", np.abs(data).max()
      rescaled = (255.0 /  np.abs(data).max() * (np.abs(data) - np.abs(data).min()) + 0.0001 ).astype(np.uint8)
      im = Image.fromarray(rescaled)
      im.save(filename+"_frame"+str(i)+".png"); #,format=PNG)

def write_dicom(pixel_array,filename):
    """
    INPUTS:
    pixel_array: 2D numpy ndarray.  If pixel_array is larger than 2D, errors.
    filename: string name for the output file.
    """

    ## This code block was taken from the output of a MATLAB secondary
    ## capture.  I do not know what the long dotted UIDs mean, but
    ## this code works.
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
    return


