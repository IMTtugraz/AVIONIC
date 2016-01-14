## AVIONIC
Accelerated Variational dynamic MRI reconstruction

# Authors
* Andreas Schwarzl (andreas.schwarzl@student.tugraz.at)
* Martin Holler (martin.holler@uni-graz.at)
* Matthias Schloegl (matthias.schloegl@tugraz.at)

# License
This software is published under GNU GPLv3. In particular, all source code is provided "as is" without warranty of any kind, either expressed or implied. For details, see the attached LICENSE.

# Reference
[1]  Matthias Schloegl, Martin Holler, Kristian Bredies, Karl Kunisch, and Rudolf Stollberger. ICTGV Regularization for Highly Accelerated Dynamic MRI. Proc. Intl. Soc. Mag. Reson. Med. 23, Toronto, Canada; 

[2] Matthias Schloegl, Martin Holler, Kristian Bredies, and Rudolf Stollberger. A Variational Approach for Coil-Sensitivity Estimation for Undersampled Phase-Sensitive Dynamic MRI Reconstruction Proc. Intl. Soc. Mag. Reson. Med. 23, Toronto, Canada; 

This work is funded and supported by the Austrian Science Fund (FWF) in the context of project ”SFB F3209-19” (Mathematical Optimization and Applications in Biomedical Sciences)
http://math.uni-graz.at/mobis/
## Dependencies

* CUDA 4.0
* CMAKE 2.8
* GCC
* [AGILE](http://www.tugraz.at/fileadmin/user_upload/Institute/IMT/files/misc/agile-20160116.tar.gz)
* [gpuNUFFT](https://github.com/andyschwarzl/gpuNUFFT)
* [ISMRMRD](https://github.com/ismrmrd/ismrmrd)
* [DCMTK](http://dicom.offis.de/dcmtk.php.de)
* [Doxygen](http://www.stack.nl/~dimitri/doxygen/) (for code docs)

## Setup

0 Preparations 
* Make sure that the CUDA environment is set up correctly 
* Ensure that the DCMDICTPATH environment variable is set correctly, namely with <br>
  `export DCMDICTPATH=$DCMDICTPATH:/usr/local/share/dcmtk/dicom.dic:/usr/local/share/dcmtk/private.dic`

1 Install AGILE lib 
```
wget http://www.tugraz.at/fileadmin/user_upload/Institute/IMT/files/misc/agile-20160116.tar.gz
tar -xf agile-20160116.tar.gz
cd AGILE
mkdir build
cd build
cmake ..
make -j 
sudo make install
``` 

2 Install gpuNUFFT 
```
git clone https://github.com/andyschwarzl/gpuNUFFT.git
cd gpuNUFFT/CUDA
mkdir build
cd build
cmake ..
make
``` 

3 Install ISMRMRD 
```
git clone https://github.com/ismrmrd/ismrmrd
cd ismrmrd/
mkdir build
cd build
cmake ../
make
sudo make install
``` 

4 Install AVIONIC recon lib
```
git clone https://github.com/IMTtugraz/AVIONIC.git
cd AVIONIC/CUDA
mkdir build
cd build
cmake .. -DGPUNUFFT_ROOT_DIR=/path/to/gpuNUFFT
make -j 
```

5 Add binary to PATH (bash)
```
in ~/.bashrc add:
export PATH=/path/to/AVIONIC/bin/:${PATH} 
```

## Doc
In order to generate the code documentation, run

```
make doc
```
and open the file `doc/html/index.html`. 


## Display help
```
avionic --help
```

## Example reconstruction

a) Cartesian
```
wget ftp://ftp.tugraz.at/outgoing/avionic_testfiles/cine_tpat8_sedona.h5
mkdir ./recon_tpat/
avionic -r cine_tpat_8_sedona.h5 -a ./recon_tpat/recon.dcm
```

a) Radial
```
wget ftp://ftp.tugraz.at/outgoing/avionic_testfiles/cine_rad_24_sedona.h5
mkdir ./recon_rad/
avionic -r cine_rad_24_sedona.h5 -a -n ./recon_rad/recon.dcm


