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
* Schloegl, Matthias and Holler, Martin and Schwarzl, Andreas and Bredies, Kristian and Stollberger, Rudolf. <br>
  __Infimal convolution of total generalized variation functionals for dynamic MRI.__<br>
  _Magnetic Resonance in Medicine_, 2017; 78(1):142-155<br>
  doi: [10.1002/mrm.26352](http://onlinelibrary.wiley.com/doi/10.1002/mrm.26352/full) <br>
  OA: [EuroPubMed](http://europepmc.org/articles/PMC5553112)

```
@article {MRM:MRM26352,
author = {Schloegl, Matthias and Holler, Martin and Schwarzl, Andreas and Bredies, Kristian and Stollberger, Rudolf},
title = {Infimal convolution of total generalized variation functionals for dynamic MRI},
journal = {Magnetic Resonance in Medicine},
volume = {78},
number = {1},
issn = {1522-2594},
url = {http://dx.doi.org/10.1002/mrm.26352},
doi = {10.1002/mrm.26352},
pages = {142--155},
keywords = {dynamic magnetic resonance imaging, CMR, perfusion imaging, total generalized variation, infimal convolution, variational models},
year = {2017},
}
```

* Matthias Schloegl, Martin Holler, Kristian Bredies, and Rudolf Stollberger.<br>
  __A Variational Approach for Coil-Sensitivity Estimation for Undersampled Phase-Sensitive Dynamic MRI Reconstruction.__ <br>
  _Proc. Intl. Soc. Mag. Reson. Med. 23_, Toronto, Canada 

## Acknowledgement
This work is funded and supported by the [Austrian Science Fund (FWF)](http://fwf.ac.at) in the context of project 'SFB F3209-19' [Mathematical Optimization and Applications in Biomedical Sciences](http://imsc.uni-graz.at/mobis/).
 We also gratefully acknowledge the support of [NVIDIA Corporation](http://nvidia.com) with the donation of the Tesla K40c GPU used for this research.

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
* build dcmtk from source (shared libs on) <br>
 `wget https://distfiles.macports.org/dcmtk/dcmtk-3.6.1_20160630.tar.gz`
* build hdf5 from source (shared libs on) <br>
 `wget https://www.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8.10/src/hdf5-1.8.10.tar.bz2`
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
git clone --branch deapo-scaling https://github.com/andyschwarzl/gpuNUFFT.git
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
## DEMO 1: Reconstruction from BINARY data for retrospectively accelerated cine cardiac and cardiac perfusion data (shell script)


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.807196.svg)](https://doi.org/10.5281/zenodo.807196)
 
Available examples for sampling patterns: "vista", "vd" (variable density), "uni" uniformly sampled
Available acceleration factors: 4,8,12,16
Available functional types: ICTGV2, TGV2, TV 

Results are exported again to binary format

Example for ICTGV reconstruction with VISTA pattern and subsampling factor 12

1 CINE cardiac imaging
```
./demo_avionic_cine.sh --functype=ICTGV2 --pattern=vista --red=12
```

2 Cardiac perfusion imaging
```
./demo_avionic_perf.sh --functype=ICTGV2 --pattern=vista --red=12
```

## DEMO 2: Reconstruction from ISMRMRD data for real accelerated cine cardiac data (T-Pat and radial acquisition)


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.807635.svg)](https://doi.org/10.5281/zenodo.807635)

Results are exported to dicom format

1 Cartesian
```
wget https://zenodo.org/record/807635/files/cine_tpat_8_sedona.h5 --no-check-certificate
mkdir ./recon_tpat/
avionic  -o -p ./CUDA/config/default_cine.cfg  -r cine_tpat_8_sedona.h5 -a ./recon_tpat/recon.dcm
```

2 Radial
```
wget https://zenodo.org/record/807635/files/cine_rad_24_sedona.h5 --no-check-certificate
mkdir ./recon_rad/
avionic -o -p ./CUDA/config/default_cine.cfg -r cine_rad_24_sedona.h5 -a -n ./recon_rad/recon.dcm
```

## Demo 3: Reconstruction of MATLAB data with demo script 


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.815385.svg)](https://doi.org/10.5281/zenodo.815385)

Please look into demo_avionic.m

1 Cartesian testdata for dynamic MRI reconstruction with ICTGV, spatio-temp. TGV2 or spatio-temp. TV

testdata_cinecardiac_avionic.mat: Cartesian acquired fully-sampled retrospectively gated CINE cardiac data (bSSFP) in short-axis view

 
2 Cartesian volumetric testdata for static TGV reconstruction, example for retrospective acceleration with CAIPIRINHA

testdata_cart_avionic_tgv3d.mat: Cartesian VIBE data of the left hand with partial-fourier acquisition

 
3 Non-Cartesian volumetric testdata_for static TGV2 reconstruction, retrospective acceleration possible by selection a lower number of spokes

noncart_avionic_tgv3d.mat: radial VIBE data of the human brain with golden-angle stack-of-stars sampling 

