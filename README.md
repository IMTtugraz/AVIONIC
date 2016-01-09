## AVIONIC
Accelerated Variational dynamic MRI reconstruction

## Dependencies

* CUDA 4.0
* CMAKE 2.8
* GCC
* [AGILE](https://github.com/IMTtugraz/AGILE)
* [gpuNUFFT](https://github.com/andyschwarzl/gpuNUFFT)
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
3 Install AVIONIC recon lib
```
git clone https://github.com/IMTtugraz/AVIONIC.git
cd AVIONIC/CUDA
mkdir build
cd build
cmake .. -DGPUNUFFT_ROOT_DIR=/path/to/gpuNUFFT
make -j 
```
4 Run reconstruction
```
bin/avionic 
```

## Doc

In order to generate the code documentation, run

```
make doc
```

and open the file `doc/html/index.html`. 
