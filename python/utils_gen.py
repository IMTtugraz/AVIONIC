from numpy import linalg as LA
import numpy as np, h5py 
import matplotlib
print "Default backend is: %s" % (matplotlib.get_backend(),)
matplotlib.use('TkAgg')
print "Backend is now: %s" % (matplotlib.get_backend(),)
from matplotlib import pyplot as plt
#import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import scipy.io as sio
import ismrmrd
import ismrmrd.xsd
from ismrmrdtools import show, transform, coils, grappa, sense
import struct
import time

#from ismrmrdtools import show, transform, coils, grappa, sense
#import hdf5storage
def refrecon(data):
  im = transform.fftn(data)
  
  #im = transform.transform_kspace_to_image(all_data[set_, avg, phase, rep, contrast, slice, :, :, :, :], [1,2,3])
  # Sum of squares
  im = np.sqrt(np.sum(np.abs(im) ** 2,axis=0))
  return im 

def calc_ser(ref,recon):
  dim=ref.shape
  ref=ref.reshape((dim[0]*dim[1],dim[2])); 
  recon=recon.reshape((dim[0]*dim[1],dim[2]));
  ser=-10*np.log10( (LA.norm(ref-recon,'fro'))**2/LA.norm(ref,'fro')**2)
  return ser;

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
  print "nx=" + str(nx) + " / ny=" + str(ny) + " / ncoils=" + str(ncoils) + " / nframes=" + str(nframes)
  # dt=np.transpose(dt.reshape((nframes,nx,ny)),(1,2,0));
  dt=np.transpose(dt.reshape((nframes,nx,ny,ncoils)),(1,2,0,3));
  dt=np.squeeze(dt);
  return dt

def showMat(dataFile):
    img=dataFile;
    print img.size

    imgDim = list(img.shape)
    print imgDim 

    if len(imgDim) > 2:
        fig = plt.figure()
        ims = []
        for i in range(0,imgDim[2]):
            im = plt.imshow(abs(img[:,:,i]))#, cmap=cm.gray)
            ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
        repeat_delay=1000)
        #ani.save('dynamic_images.mp4')

    elif len(imgDim) == 2:
        imgplot = plt.imshow(abs(img));#, cmap=cm.gray)

    elif len(imgDim) == 1:
         imgplot = plot(abs(img))

    plt.show()


