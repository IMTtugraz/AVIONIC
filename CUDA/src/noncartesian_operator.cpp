#include "../include/noncartesian_operator.h"

NoncartesianOperator::NoncartesianOperator(unsigned width, unsigned height,
                                           unsigned coils, unsigned frames,
                                           unsigned nSpokes, unsigned nFE,
                                           unsigned spokesPerFrame,
                                           RVector &kTraj, RVector &dens,
                                           CVector &sens, DType kernelWidth,
                                           DType sectorWidth, DType osf)
  : BaseOperator(width, height, 0, coils, frames), kTraj(kTraj), dens(dens),
    sens(sens), nSpokes(nSpokes), nFE(nFE), spokesPerFrame(spokesPerFrame),
    kernelWidth(kernelWidth), sectorWidth(sectorWidth), osf(osf)
{
  Init();
}

CVector EmptySens(0);

NoncartesianOperator::NoncartesianOperator(
    unsigned width, unsigned height, unsigned coils, unsigned frames,
    unsigned nSpokes, unsigned nFE, unsigned spokesPerFrame, RVector &kTraj,
    RVector &dens, DType kernelWidth, DType sectorWidth, DType osf)
  : BaseOperator(width, height, 0, coils, frames), kTraj(kTraj), dens(dens),
    sens(EmptySens), nSpokes(nSpokes), nFE(nFE), spokesPerFrame(spokesPerFrame),
    kernelWidth(kernelWidth), sectorWidth(sectorWidth), osf(osf)
{
  Init();
}

NoncartesianOperator::~NoncartesianOperator()
{
}

void NoncartesianOperator::Init()
{
  nSamplesPerFrame = nFE * spokesPerFrame;

  std::cout << "nSpokes: " << nSpokes << " nFE: " << nFE
            << " spokesPerFrame: " << spokesPerFrame << std::endl;
  std::cout << "nSamplesPerFrame: " << nSamplesPerFrame << std::endl;


  // In order to initialize the gpuNUFFT Operator factory
  // correctly, the trajectory, sensitivity and density
  // data has to reside on CPU memory
  kTrajHost = std::vector<RType>(2 * nSamplesPerFrame);
  kTraj.copyToHost(kTrajHost);
  kTrajData.data = &(kTrajHost[0]);
  kTrajData.dim.length = nSamplesPerFrame;

  std::cout<< "noncart op init norm(dens)" << agile::norm2(dens)  << std::endl;
  densHost = std::vector<RType>(nSamplesPerFrame*frames);
  //dens.copyToHost(densHost);
  std::fill(densHost.begin(),densHost.end(),1.0);

  densData.data = &(densHost[0]);
  densData.dim.length = nSamplesPerFrame;

  imgDims.width = width;
  imgDims.height = height;

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
  gpuNUFFTOps = std::vector<gpuNUFFT::GpuNUFFTOperator *>(frames);
  for (unsigned frame = 0; frame < frames; frame++)
  {
    unsigned fOff = frame * nSamplesPerFrame;
    kTrajData.data = &(kTrajHost[2 * fOff]);
    densData.data = &(densHost[fOff]);
    if (hasCoilData)
    {
      gpuNUFFTOps[frame] = factory.createGpuNUFFTOperator(
          kTrajData, densData, sensData, kernelWidth, sectorWidth, osf,
          imgDims);
    }
    else
    {
      gpuNUFFTOps[frame] = factory.createGpuNUFFTOperator(
          kTrajData, densData, kernelWidth, sectorWidth, osf,
          imgDims);
    }
  }
}

RType NoncartesianOperator::AdaptLambda(RType k, RType d)
{
  RType lambda = 0;
  RType subfac = width / (spokesPerFrame * M_PI / 2.0);
  lambda = subfac * k + d;

  return lambda;
}


//======================================================================================================
// image to kdata
//======================================================================================================
void NoncartesianOperator::BackwardOperation(CVector &x_gpu, CVector &z_gpu,
                                             CVector &b1_gpu)
{
  // Input Image Array¬
  gpuNUFFT::GpuArray<CufftType> imgArray;
  imgArray.data = (CufftType *)x_gpu.data();
  imgArray.dim = imgDims;

  // Output KSpace Array¬
  gpuNUFFT::GpuArray<DType2> dataArray;
  dataArray.data = (float2 *)z_gpu.data();
  dataArray.dim.length = nSamplesPerFrame;
  dataArray.dim.channels =
      coils;  // < essential to ensure gpuNUFFT multi-coil performance

  // Perform FT Operation¬ per frame
  for (unsigned frame = 0; frame < frames; frame++)
  {
    imgArray.data = (float2 *)(x_gpu.data() + frame * width * height);
    dataArray.data =
        (float2 *)(z_gpu.data() + frame * coils * nSamplesPerFrame);
    gpuNUFFTOps[frame]->performForwardGpuNUFFT(imgArray, dataArray);

  }

   // Multiply with sqrt of densitiy compensation (square-root of dens. was applied in main)
  for (unsigned coil = 0; coil < coils ; coil++)
  {
  agile::lowlevel:: multiplyElementwise(z_gpu.data() + coil * spokesPerFrame * nFE *frames,
                                        dens.data(), z_gpu.data() + coil* spokesPerFrame * nFE *frames,
                                        spokesPerFrame * nFE *frames);
  }

}



CVector NoncartesianOperator::BackwardOperation(CVector &x_gpu, CVector &b1_gpu)
{
  unsigned int N = nSpokes * nFE;

  CVector z_gpu(N * coils);
  this->BackwardOperation(x_gpu, z_gpu, b1_gpu);

  return z_gpu;
}

//======================================================================================================
// kdata to image
//======================================================================================================
void NoncartesianOperator::ForwardOperation(CVector &x_gpu, CVector &sum,
                                            CVector &b1_gpu)
{
  // Multiply with sqrt of densitiy compensation  (square-root of dens. was applied in main)
  for (unsigned coil = 0; coil < coils ; coil++)
  {
    agile::lowlevel:: multiplyElementwise(x_gpu.data() + coil * spokesPerFrame * nFE *frames,
                                        dens.data(), x_gpu.data() + coil * spokesPerFrame * nFE *frames,
                                        spokesPerFrame * nFE *frames);
  }


  // Input kspace Data
  gpuNUFFT::GpuArray<DType2> dataArray;
  dataArray.data = (float2 *)x_gpu.data();
  dataArray.dim.length = nSamplesPerFrame;
  dataArray.dim.channels =
      coils;  // < essential to ensure gpuNUFFT multi-coil performance

  // Output Image Array¬
  gpuNUFFT::GpuArray<CufftType> imgArray;
  imgArray.data = (CufftType *)sum.data();
  imgArray.dim = imgDims;
  imgArray.dim.channels = coils;
  for (unsigned frame = 0; frame < frames; frame++)
  {
    imgArray.data = (CufftType *)(sum.data() + frame * width * height);
    dataArray.data =
        (float2 *)(x_gpu.data() + frame * coils * nSamplesPerFrame);
    gpuNUFFTOps[frame]->performGpuNUFFTAdj(dataArray, imgArray);
  }

  //agile::scale((CType) (1.0/std::sqrt(4.0*width*height)),x_gpu,x_gpu);
}

CVector NoncartesianOperator::ForwardOperation(CVector &x_gpu, CVector &b1_gpu)
{
  unsigned N = width * height * frames;
  CVector sum_gpu(N);
  ForwardOperation(x_gpu, sum_gpu, b1_gpu);
  return sum_gpu;
}


