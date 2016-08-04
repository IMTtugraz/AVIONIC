# AVIONIC
Accelerated Variational Dynamic MRI Reconstruction

## Contributors
* [Andreas Schwarzl](https://github.com/andyschwarzl) (Graz University of Technology)
* [Martin Holler](http://imsc.uni-graz.at/hollerm) (University of Graz) 
* [Matthias Schloegl](http://www.tugraz.at/institute/imt/people/schloegl/) (Graz University of Technology)
* [Kristian Bredies](http://imsc.uni-graz.at/bredies) (University of Graz) 

## License
This software is published under GNU GPLv3. In particular, all source code is provided "as is" without warranty of any kind, either expressed or implied. For details, see the attached LICENSE.

## General Information
AVIONIC (Accelerated Variational Dynamic MRI Reconstruction) is an open source software for GPU accelerated reconstruction of hightly undersampled
dynamic Magnetic Resonance data, such as functional cardiac imaging oder dynamic contrast enhanced (DCE) MRI applications. It also includes a variational
approach for the estimation of receiver-coil sensitivity profiles from the undersampled data. The current version is able to handle Cartesian and Non-Cartesian
data if the trajectory information is provided.

If you use this software please cite:
* Schloegl, Matthias and Holler, Martin and Schwarzl, Andreas and Bredies, Kristian and Stollberger, Rudolf.
  __Infimal convolution of total generalized variation functionals for dynamic MRI.__
  _Magnetic Resonance in Medicine_, 2016 (in press); doi: [10.1002/mrm.26352](http://onlinelibrary.wiley.com/doi/10.1002/mrm.26352/full) 
```
@article {MRM:MRM26352,
author = {Schloegl, Matthias and Holler, Martin and Schwarzl, Andreas and Bredies, Kristian and Stollberger, Rudolf},
title = {Infimal convolution of total generalized variation functionals for dynamic MRI},
journal = {Magnetic Resonance in Medicine},
issn = {1522-2594},
url = {http://dx.doi.org/10.1002/mrm.26352},
doi = {10.1002/mrm.26352},
pages = {n/a--n/a},
keywords = {dynamic magnetic resonance imaging, CMR, perfusion imaging, total generalized variation, infimal convolution, variational models},
year = {2016},
}
```

* Matthias Schloegl, Martin Holler, Kristian Bredies, and Rudolf Stollberger. __A Variational Approach for Coil-Sensitivity Estimation for Undersampled Phase-Sensitive Dynamic MRI Reconstruction.__  _Proc. Intl. Soc. Mag. Reson. Med. 23_, Toronto, Canada; 

This work is funded and supported by the Austrian Science Fund (FWF) in the context of project 'SFB F3209-19' [Mathematical Optimization and Applications in Biomedical Sciences](http://imsc.uni-graz.at/mobis/)


For questions and comments on the project please contact [Matthias Schloegl](mailto:matthias.schloegl@tugraz.at)
## Dependencies
* CUDA 4.0
* CMAKE 2.8
* GCC
* [AGILE](https://github.com/IMTtugraz/AGILE.git)
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
git clone https://github.com/IMTtugraz/AGILE.git
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
## DEMO Reconstruction

Example for ICTGV reconstruction with VISTA pattern and subsampling factor 12

1 CINE cardiac imaging
```
./demo_avionic_cine.sh --functype=ICTGV2 --pattern=vista --red=12
```

2 Cardiac perfusion imaging
```
./demo_avionic_perf.sh --functype=ICTGV2 --pattern=vista --red=12
```

## Example reconstruction for ISMRMRD data
1 Cartesian
```
wget ftp://ftp.tugraz.at/outgoing/AVIONIC/avionic_testdata/cine_tpat8_sedona.h5
mkdir ./recon_tpat/
avionic -r cine_tpat_8_sedona.h5 -a ./recon_tpat/recon.dcm
```

2 Radial
```
wget ftp://ftp.tugraz.at/outgoing/AVIONIC/avionic_testdata/cine_rad_24_sedona.h5
mkdir ./recon_rad/
avionic -r cine_rad_24_sedona.h5 -a -n ./recon_rad/recon.dcm


