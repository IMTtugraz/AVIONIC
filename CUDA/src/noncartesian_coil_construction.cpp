#include "../include/noncartesian_coil_construction.h"

NoncartesianCoilConstruction::NoncartesianCoilConstruction(
    unsigned width, unsigned height, unsigned coils, unsigned frames,
    NoncartesianOperator *mrOp)
  : CoilConstruction(width, height, coils, frames), mrOp(mrOp)
{
}

NoncartesianCoilConstruction::NoncartesianCoilConstruction(
    unsigned width, unsigned height, unsigned coils, unsigned frames,
    CoilConstructionParams &params, NoncartesianOperator *mrOp)
  : CoilConstruction(width, height, coils, frames, params), mrOp(mrOp)
{
}

NoncartesianCoilConstruction::~NoncartesianCoilConstruction()
{
}

void NoncartesianCoilConstruction::TimeAveragedReconstruction(CVector &kdata,
                                                              CVector &u,
                                                              CVector &crec,
                                                              bool applyPhase)
{
  unsigned N = mrOp->nFE * mrOp->spokesPerFrame * frames;

  // create gpuNUFFT operator with the full trajectory
  // needs to rearrange (x,y) trajectory tuples and data values
  std::vector<DType> kTrajHost = std::vector<RType>(2 * N);
  mrOp->kTraj.copyToHost(kTrajHost);
  std::vector<DType> kTrajRearranged(2 * N);

  std::vector<CType> kDataHost;
  std::vector<DType2> kDataRearranged(N * coils);
  kdata.copyToHost(kDataHost);

  unsigned nSamplesPerFrame = mrOp->nFE * mrOp->spokesPerFrame;
  for (unsigned frame = 0; frame < frames; frame++)
  {
    for (unsigned cnt = 0; cnt < nSamplesPerFrame; cnt++)
    {
      unsigned off = frame * 2 * nSamplesPerFrame;
      unsigned ind = frame * nSamplesPerFrame + cnt;
      // x
      kTrajRearranged[ind] = kTrajHost[cnt + off];
      // y
      kTrajRearranged[ind + frames * nSamplesPerFrame] =
          kTrajHost[cnt + off + nSamplesPerFrame];

      // rearrange data
      for (unsigned coil = 0; coil < coils; coil++)
      {
        unsigned off = coil * frames * nSamplesPerFrame;
        kDataRearranged[cnt + frame * nSamplesPerFrame + off].x =
            kDataHost[cnt + coil * nSamplesPerFrame +
                      frame * coils * nSamplesPerFrame].real();
        kDataRearranged[cnt + frame * nSamplesPerFrame + off].y =
            kDataHost[cnt + coil * nSamplesPerFrame +
                      frame * coils * nSamplesPerFrame].imag();
      }
    }
  }

  gpuNUFFT::Array<DType> kTrajData;
  kTrajData.data = &(kTrajRearranged[0]);
  kTrajData.dim.length = N;

  std::vector<DType> densHost = std::vector<RType>(N);
  mrOp->dens.copyToHost(densHost);
  gpuNUFFT::Array<DType> densData;
  densData.data = &(densHost[0]);
  densData.dim.length = N;

  gpuNUFFT::Dimensions imgDims(width, height);

  CVector kDataTemp(N * coils);
  kDataTemp.assignFromHost(kDataRearranged.begin(), kDataRearranged.end());
  gpuNUFFT::GpuArray<DType2> dataArray;
  dataArray.data = (float2 *)kDataTemp.data();
  dataArray.dim.length = N;
  dataArray.dim.channels =
      coils;  // < essential to ensure gpuNUFFT multi-coil performance

  gpuNUFFT::GpuNUFFTOperatorFactory factory;
  gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp = factory.createGpuNUFFTOperator(
      kTrajData, densData, 3.0, 8.0, 2.0, imgDims);

  CVector temp(width * height);
  CVector img(width * height * coils);
  CVector angle(width * height);

  // initialize vectors
  angle.assign(angle.size(), 0);
  crec.assign(crec.size(), 0);
  RVector angleReal(width * height);

  // init img output array
  // containing multicoil images
  gpuNUFFT::GpuArray<CufftType> imgArray;
  imgArray.data = (CufftType *)img.data();
  imgArray.dim = imgDims;
  imgArray.dim.channels = coils;

  // perform adjoint operation
  gpuNUFFTOp->performGpuNUFFTAdj(dataArray, imgArray);

  for (unsigned cnt = 0; cnt < coils; cnt++)
  {
    utils::GetSubVector(img, temp, cnt, width * height);
    utils::SetSubVector(temp, crec, cnt, width * height);
    agile::addVector(angle, temp, angle);
    agile::multiplyConjElementwise(temp, temp, temp);
    agile::addVector(u, temp, u);
  }
  agile::sqrt(u, u);

  if (applyPhase)
  {
    agile::phaseVector(angle, angleReal);
    agile::scale(CType(0, 1.0), angleReal, angle);
    agile::expVector(angle, angle);

    agile::multiplyElementwise(u, angle, u);
  }
  delete gpuNUFFTOp;
}

