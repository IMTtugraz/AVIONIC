#include "../include/noncartesian_operator3d.h"

NoncartesianOperator3D::NoncartesianOperator3D( unsigned width, unsigned height,
                                                unsigned depth, unsigned coils,
                                                unsigned nSpokes, unsigned nFE,
                                                unsigned spokesPerFrame,
                                                RVector &kTraj, RVector &dens,
                                                CVector &sens, DType kernelWidth,
                                                DType sectorWidth, DType osf)
  : BaseOperator(width, height, depth, coils, 0), kTraj(kTraj), dens(dens),
    sens(sens), nSpokes(nSpokes), nFE(nFE), spokesPerFrame(spokesPerFrame),
    kernelWidth(kernelWidth), sectorWidth(sectorWidth), osf(osf)
{
  Init();
}

CVector EmptySens3d(0);

NoncartesianOperator3D::NoncartesianOperator3D( unsigned width, unsigned height,
                                                unsigned depth, unsigned coils,
                                                unsigned nSpokes, unsigned nFE,
                                                unsigned spokesPerFrame,
                                                RVector &kTraj, RVector &dens,
                                                DType kernelWidth, DType sectorWidth, DType osf)
  : BaseOperator(width, height, 0, coils, frames), kTraj(kTraj), dens(dens),
    sens(EmptySens3d), nSpokes(nSpokes), nFE(nFE), spokesPerFrame(spokesPerFrame),
    kernelWidth(kernelWidth), sectorWidth(sectorWidth), osf(osf)
{
  Init();
}

NoncartesianOperator3D::~NoncartesianOperator3D()
{
}

void NoncartesianOperator3D::Init()
{

  nSamples  = nFE * nSpokes;
  std::cout << "nSpokes: " << nSpokes << " nFE: " << nFE
     << " nSamples: " << nSamples << std::endl;


  // In order to initialize the gpuNUFFT Operator factory
  // correctly, the trajectory, sensitivity and density
  // data has to reside on CPU memory
  kTrajHost = std::vector<RType>(3 * nSamples);
  kTraj.copyToHost(kTrajHost);
  kTrajData.data = &(kTrajHost[0]);
  kTrajData.dim.length = nSamples;

  densHost = std::vector<RType>(nSamples);
  dens.copyToHost(densHost);
  //std::fill(densHost.begin(),densHost.end(),1.0);

  densData.data = &(densHost[0]);
  densData.dim.length = nSamples;

  imgDims.width  = width;
  imgDims.height = height;
  imgDims.depth  = depth;

  bool hasCoilData = sens.size() > 0;
  if (hasCoilData)
  {
    sensHost = std::vector<DType2>(sens.size());
    sens.copyToHost(sensHost);
    sensData.data = (float2 *)&(sensHost[0]);
    sensData.dim = imgDims;
    sensData.dim.channels = coils;
  }
  
  gpuNUFFT::GpuNUFFTOperatorFactory factory;
 //gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp = NULL;
  //kTrajData.data = &(kTrajHost[0]);
  //densData.data = &(densHost[0]);
  if (hasCoilData)
  {
    gpuNUFFTOp = factory.createGpuNUFFTOperator(
        kTrajData, densData, sensData, kernelWidth, sectorWidth, osf,
        imgDims);
    std::cout << "Creating gpuNUFFT Operator of Type " << gpuNUFFTOp->getType()<<std::endl;
  }
  else
  {
    gpuNUFFTOp  = factory.createGpuNUFFTOperator(
        kTrajData, densData, kernelWidth, sectorWidth, osf,
        imgDims);
  }
}

RType NoncartesianOperator3D::AdaptLambda(RType k, RType d)
{
  //TODO: check
  RType lambda = 0;
  RType subfac = width*depth / (nSpokes   * M_PI / 2.0);
  lambda = subfac * k + d;

  return lambda;
}


//======================================================================================================
// image to kdata
//======================================================================================================
void NoncartesianOperator3D::BackwardOperation(CVector &x_gpu, CVector &z_gpu,
                                             CVector &b1_gpu)
{

  // Input Image Array¬
  gpuNUFFT::GpuArray<CufftType> imgArray;
  imgArray.data = (CufftType *)x_gpu.data();
  imgArray.dim = imgDims;

  // Output KSpace Array¬
  gpuNUFFT::GpuArray<DType2> dataArray;
  dataArray.data = (float2 *)z_gpu.data();
  dataArray.dim.length = nSamples;
  dataArray.dim.channels = 
      coils;  // < essential to ensure gpuNUFFT multi-coil performance
 // Perform FT Operation
  imgArray.data = (float2 *)(x_gpu.data());
  dataArray.data =  (float2 *)(z_gpu.data());
 
  gpuNUFFTOp->performForwardGpuNUFFT(imgArray, dataArray);

  /*
  // Multiply with sqrt of densitiy compensation (square-root of dens. was applied in main)
  for (unsigned coil = 0; coil < coils ; coil++)
  {
  agile::lowlevel:: multiplyElementwise(z_gpu.data() + coil * nSamples,
                                        dens.data(), z_gpu.data() + coil*nSamples,
                                        nSamples);
  }
  */
}
void NoncartesianOperator3D::BackwardOperation(CVector &x_gpu, CVector &z_gpu,
                                 CVector &b1_gpu, CVector &x_hat_gpu)
{
  this->BackwardOperation(x_gpu, z_gpu, b1_gpu);
}



CVector NoncartesianOperator3D::BackwardOperation(CVector &x_gpu, CVector &b1_gpu)
{
  unsigned int N = nSpokes * nFE;

  CVector z_gpu(N * coils);
  this->BackwardOperation(x_gpu, z_gpu, b1_gpu);

  return z_gpu;
}

//======================================================================================================
// kdata to image
//======================================================================================================
void NoncartesianOperator3D::ForwardOperation(CVector &x_gpu, CVector &sum,
                                            CVector &b1_gpu)
{
  
 /* 
  // Multiply with sqrt of densitiy compensation  (square-root of dens. was applied in main)
  for (unsigned coil = 0; coil < coils ; coil++)
  {
    agile::lowlevel:: multiplyElementwise(x_gpu.data() + coil * nSamples,
                                        dens.data(), x_gpu.data() + coil * nSamples,
                                        nSamples);
  }
*/

  // Input kspace Data
  gpuNUFFT::GpuArray<DType2> dataArray;
  dataArray.data = (float2 *)x_gpu.data();
  dataArray.dim.length = nSamples;
  dataArray.dim.channels =
      coils;  // < essential to ensure gpuNUFFT multi-coil performance

  // Output Image Array¬
  gpuNUFFT::GpuArray<CufftType> imgArray;
  imgArray.data = (CufftType *)sum.data();
  imgArray.dim = imgDims;
  imgArray.dim.channels = coils;

  imgArray.data = (CufftType *)(sum.data());
  dataArray.data =
      (float2 *)(x_gpu.data() );
  gpuNUFFTOp->performGpuNUFFTAdj(dataArray, imgArray);

}

void NoncartesianOperator3D::ForwardOperation(CVector &x_gpu, CVector &sum,
                              CVector &b1_gpu, CVector &z_gpu)
{

  ForwardOperation(x_gpu, sum, b1_gpu);
}

CVector NoncartesianOperator3D::ForwardOperation(CVector &x_gpu, CVector &b1_gpu)
{
  unsigned N = width * height * depth;
  CVector sum_gpu(N);
  ForwardOperation(x_gpu, sum_gpu, b1_gpu);
  return sum_gpu;
}


