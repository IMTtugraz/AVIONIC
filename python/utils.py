from numpy import linalg as LA
import numpy as np, h5py 
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import scipy.io as sio
import struct

def readbin_vector(filename,nx,ny,nframes):
  "READBIN_VECTOR reads a (column) vector from a binary file"
  with open(filename) as f:
    v=f.read()
    f.close()
  
  iscomplex=struct.unpack("B",v[0]);
  size=struct.unpack("I",v[1:5]);
  print("iscomplex" + str(iscomplex[0]) + " / size " + str(size[0]))
  dl=size[0]*(iscomplex[0]+1);
  data=struct.unpack('d'*dl , v[5:len(v)]) 
  dt=np.array(data,dtype=np.float32)
  if iscomplex[0]:
     dt=np.vectorize(complex)(dt[0:size[0]],dt[size[0]:])
  
  ncoils=dt.size/(nx*ny*nframes);
  dt=np.transpose(dt.reshape((nframes,nx,ny,ncoils)),(1,2,0,3));
  dt=np.squeeze(dt);
  return dt

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
         imgplot = plot(abs(img))

    plt.show()


