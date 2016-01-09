## AVIONIC
Accelerated Variational dynamic MRI reconstruction

## Dependencies

* CUDA 4.0
* CMAKE 2.8
* GCC
* [gpuNUFFT](https://github.com/andyschwarzl/gpuNUFFT)
* [GoogleTest](https://code.google.com/p/googletest/downloads/detail?name=gtest-1.6.0.zip&can=2&q=)
* [Doxygen](http://www.stack.nl/~dimitri/doxygen/) (for code docs)
* [DCMTK](http://dicom.offis.de/dcmtk.php.de)

## Setup

0 Preparations 
* Make sure that the CUDA environment is set up correctly 
* Ensure that the DCMDICTPATH environment variable is set correctly, namely with <br>
  `export DCMDICTPATH=$DCMDICTPATH:/usr/local/share/dcmtk/dicom.dic:/usr/local/share/dcmtk/private.dic`

1 Install AGILE lib 
```
cd CUDA/AGILE
mkdir build
cd build
cmake ..
make -j 
sudo make install
``` 
2 Install ICTGV recon lib
```
cd CUDA/ictgv
mkdir build
cd build
cmake .. -DGTEST_DIR=/path/to/google/test/framework -DGPUNUFFT_ROOT_DIR=/path/to/gpuNUFFT
make -j 
```
3 Run tests
```
make tests
```
4 Run reconstruction
```
bin/fredy_mri 
```

## Doc

In order to generate the code documentation, run

```
make doc
```

and open the file `doc/html/index.html`. 
