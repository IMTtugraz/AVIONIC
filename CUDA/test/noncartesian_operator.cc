#include <gtest/gtest.h>

#include "agile/gpu_environment.hpp"
#include "agile/gpu_vector.hpp"
#include "agile/gpu_matrix.hpp"
#include "agile/calc/fft.hpp"
#include "agile/io/file.hpp"

#include "./test_utils.h"
#include "../include/types.h"
#include "../include/tv.h"
#include "../include/utils.h"
#include "../include/noncartesian_operator.h"

#include "gpuNUFFT_operator_factory.hpp"

class Test_NonCartesianOperator : public ::testing::Test
{
 public:
  static const unsigned int width = 6;
  static const unsigned int height = 6;
  static const unsigned int coils = 3;
  static const unsigned int frames = 2;
  static const unsigned int N = width * height * frames;

  static const unsigned int nSpokes = 4;
  static const unsigned int nFE = width;
  static const unsigned int nTraj = nSpokes * nFE;
  static const unsigned int spokesPerFrame = 2;
  static const unsigned int nSamplesPerFrame = nTraj / spokesPerFrame;

  virtual void SetUp()
  {
    agile::GPUEnvironment::allocateGPU(0);

    // row major
    for (unsigned frame = 0; frame < frames; frame++)
      for (unsigned coil = 0; coil < coils; coil++)
        for (unsigned row = 0; row < height; ++row)
          for (unsigned column = 0; column < width; ++column)
          {
            unsigned offset = frame * width * height * coils;
            matrix_data.push_back(
                CType(offset + XYZ2Lin(column, row, coil, width, height), 0));
          }

    trajectory_data = std::vector<RType>(2 * nTraj);

    // trajectory in correct order (frame-by-frame)
    //
    // x11, x12, x13, ..., x1N, y11, y12, y13,
    // ... y1N, x21, x22, x23, ..., x2N, y21, y22, y23, ... y2N
    //
    // N .... nFE*spokesPerFrame
    // x_ij ... x component of i-th frame, j-th data sample
    //
    for (unsigned frame = 0; frame < frames; frame++)
      for (unsigned spoke = 0; spoke < spokesPerFrame; spoke++)
        for (unsigned cnt = 0; cnt < nFE; cnt++)
        {
          unsigned ind = frame * 2 * spokesPerFrame * nFE + spoke * nFE + cnt;
          // x component
          trajectory_data[ind] = ((DType)cnt / nFE) - 0.5;

          // y component
          trajectory_data[nSamplesPerFrame + ind] =
              ((DType)(frame * spokesPerFrame + spoke) / nSpokes) - 0.5;
        }

    // data
    img = CVector(N);
    img.assign(N, 1.0);

    kdata = CVector(nTraj * coils);
    kdata.assign(nTraj * coils, 0.0);

    // ktraj
    ktraj = RVector(2 * nTraj);
    ktraj.assignFromHost(trajectory_data.begin(), trajectory_data.end());

    // density comp
    w = RVector(nTraj);
    w.assign(nTraj, 1.0);

    // b1 map
    b1 = CVector(width * height * coils);
    b1.assign(width * height * coils, 1.0);
    agile::lowlevel::scale((CType)2.0, b1.data() + width * height,
                           b1.data() + width * height, width * height);
    agile::lowlevel::scale((CType)3.0, b1.data() + 2 * width * height,
                           b1.data() + 2 * width * height, width * height);
  }

  virtual void ShutDown()
  {
  }

  std::vector<CType> matrix_data;
  std::vector<RType> trajectory_data;
  CVector img;
  CVector kdata;
  RVector ktraj;
  RVector w;
  CVector b1;
};

TEST_F(Test_NonCartesianOperator, BackwardOperation)
{
  print("Traj: ", trajectory_data);

  NoncartesianOperator nonCartOp(width, height, coils, frames, nSpokes, nFE,
                                 spokesPerFrame, ktraj, w, b1);
  kdata.assign(nTraj * coils, 0.0);
  nonCartOp.BackwardOperation(img, kdata, b1);

  std::vector<CType> dataHost(nTraj * coils);
  kdata.copyToHost(dataHost);

  print("kData:", nFE, spokesPerFrame, coils, frames, dataHost, true);
  EXPECT_NEAR(2.949,
              std::abs(dataHost[XYZ2Lin(9, 0, 0, nSamplesPerFrame, coils)]),
              EPS);
  EXPECT_NEAR(5.897,
              std::abs(dataHost[XYZ2Lin(9, 1, 0, nSamplesPerFrame, coils)]),
              EPS);
  EXPECT_NEAR(8.846,
              std::abs(dataHost[XYZ2Lin(9, 2, 0, nSamplesPerFrame, coils)]),
              EPS);

  EXPECT_NEAR(12.5416,
              std::abs(dataHost[XYZ2Lin(3, 0, 1, nSamplesPerFrame, coils)]),
              EPS);
  EXPECT_NEAR(25.0832,
              std::abs(dataHost[XYZ2Lin(3, 1, 1, nSamplesPerFrame, coils)]),
              EPS);
  EXPECT_NEAR(37.6248,
              std::abs(dataHost[XYZ2Lin(3, 2, 1, nSamplesPerFrame, coils)]),
              EPS);
}

TEST_F(Test_NonCartesianOperator, ForwardOperation)
{
  print("Traj: ", trajectory_data);

  NoncartesianOperator nonCartOp(width, height, coils, frames, nSpokes, nFE,
                                 spokesPerFrame, ktraj, w, b1);

  // Input KSpace Array¬
  kdata.assign(nTraj * coils, 1.0);
  nonCartOp.ForwardOperation(kdata, img, b1);

  std::vector<CType> imgHost(width * height * frames);
  img.copyToHost(imgHost);

  print("img:", width, height, frames, imgHost, true);
  EXPECT_NEAR(17.842, std::abs(imgHost[XYZ2Lin(3, 0, 0, width, height)]), EPS);
  EXPECT_NEAR(17.78, std::abs(imgHost[XYZ2Lin(3, 2, 0, width, height)]), EPS);
  EXPECT_NEAR(25.198, std::abs(imgHost[XYZ2Lin(3, 3, 0, width, height)]), EPS);
}

TEST_F(Test_NonCartesianOperator, Adjointness)
{
  // Generate random values¬
  std::vector<CType> x;
  std::vector<CType> y;

  // init random number generator
  srand(time(NULL));
  for (unsigned cnt = 0; cnt < nTraj * coils; cnt++)
    x.push_back(randomValue());

  CVector x_gpu(nTraj * coils);
  x_gpu.assignFromHost(x.begin(), x.end());

  for (unsigned cnt = 0; cnt < width * height * frames; cnt++)
    y.push_back(randomValue());

  CVector y_gpu(width * height * frames);
  y_gpu.assignFromHost(y.begin(), y.end());

  CVector Kx(width * height * frames);
  Kx.assign(Kx.size(), 0);

  CVector KHy(nTraj * coils);
  KHy.assign(KHy.size(), 0);

  NoncartesianOperator nonCartOp(width, height, coils, frames, nSpokes, nFE,
                                 spokesPerFrame, ktraj, w, b1);

  nonCartOp.BackwardOperation(y_gpu, KHy, b1);
  nonCartOp.ForwardOperation(x_gpu, Kx, b1);

  CType Kxy = agile::getScalarProduct(Kx, y_gpu);
  CType xKHy = agile::getScalarProduct(x_gpu, KHy);

  std::cout << Kxy << std::endl;
  std::cout << xKHy << std::endl;

  EXPECT_NEAR(0.0, std::real(Kxy - xKHy), EPS);
  EXPECT_NEAR(0.0, std::imag(Kxy - xKHy), EPS);
}

TEST_F(Test_NonCartesianOperator, AdaptLambda)
{
  NoncartesianOperator nonCartOp(width, height, coils, frames, nSpokes, nFE,
                                 spokesPerFrame, ktraj, w, b1);

  RType k = 1.0 / 3.0;
  RType d = 6;
  EXPECT_NEAR(6.636, nonCartOp.AdaptLambda(k, d), EPS);
}

TEST(Test_CUDAMemcopy, TestCopyToDeviceFromDevicePointer)
{
  std::vector<DType> host(3);
  std::vector<DType> host2(3);
  DType *dev;
  DType *dev2;

  host[0] = 1;
  host[1] = 2;
  host[2] = 4;

  cudaMalloc(&dev, 3 * sizeof(DType));
  cudaMalloc(&dev2, 3 * sizeof(DType));
  cudaMemcpy(dev, &host[0], 3 * sizeof(DType), cudaMemcpyHostToDevice);

  agile::GPUEnvironment::allocateGPU(0);
  agile::lowlevel::scale((float)2.0, dev, dev, 3);
  cudaMemcpy(&host[0], dev, 3 * sizeof(DType), cudaMemcpyDeviceToHost);

  // HostToDevice seems to work although memory of dev is alread on GPU!
  cudaMemcpy(dev2, dev, 3 * sizeof(DType), cudaMemcpyHostToDevice);
  agile::lowlevel::scale((float)3.0, dev2, dev2, 3);

  cudaMemcpy(&host2[0], dev2, 3 * sizeof(DType), cudaMemcpyDeviceToHost);

  std::cout << "host: " << host[0] << ", " << host[1] << ", " << host[2]
            << std::endl;
  std::cout << "host2: " << host2[0] << ", " << host2[1] << ", " << host2[2]
            << std::endl;
}

TEST(Test_NonCartesianOperatorReal, DISABLED_Recon)
{
  agile::GPUEnvironment::allocateGPU(0);
  agile::GPUEnvironment::printInformation(std::cout);
  unsigned int height = 384;
  unsigned int width = 384;
  unsigned int coils = 12;
  unsigned int frames = 14;

  unsigned int nFE = 384;
  unsigned int nSpokesPerFrame = 21;

  unsigned int N = width * height * frames;

  const char *output = "../test/data/output/noncart/test_recon_384_384_14.bin";

  // data
  std::vector<CType> dataHost;
  agile::readVectorFile("../test/data/noncart/test_data_384_21_12_14.bin",
                        dataHost);
  print("Data: ", 20, 1, 1, 1, dataHost, false);

  unsigned nTraj = nFE * nSpokesPerFrame * frames;
  CVector data(nTraj * coils);
  data.assignFromHost(dataHost.begin(), dataHost.end());

  // b1 map
  std::vector<CType> b1Host;
  agile::readVectorFile("../test/data/noncart/test_b1_384_384_12.bin", b1Host);
  print("B1: ", 20, 1, 1, b1Host, false);

  CVector b1(width * height * coils);
  b1.assignFromHost(b1Host.begin(), b1Host.end());

  // kspace trajectory
  std::vector<RType> traj;
  agile::readVectorFile("../test/data/noncart/test_k_384_21_14.bin", traj);
  print("Trajectory: ", 20, 1, 1, traj);
  std::cout << "Trajectory size: " << traj.size() << std::endl;
  RVector ktraj(2 * nTraj);
  ktraj.assignFromHost(traj.begin(), traj.end());

  // weight trajectory
  std::vector<RType> wHost;
  agile::readVectorFile("../test/data/noncart/test_w_384_21_14.bin", wHost);
  print("Density: ", 20, 1, 1, wHost);

  RVector w(nTraj);
  w.assignFromHost(wHost.begin(), wHost.end());

  std::cout << "debug first frame first tuple (x,y,v,w) : (" << traj[0] << ","
            << traj[nFE * nSpokesPerFrame] << "," << dataHost[0] << ","
            << wHost[0] << ")" << std::endl;

  std::cout << "debug second tuple (x,y,v,w) : (" << traj[1] << ","
            << traj[1 + nFE * nSpokesPerFrame] << "," << dataHost[1] << ","
            << wHost[1] << ")" << std::endl;

  std::cout << "debug second frame first tuple (x,y,v,w) : ("
            << traj[2 * nFE * nSpokesPerFrame] << ","
            << traj[2 * nFE * nSpokesPerFrame + nFE * nSpokesPerFrame] << ","
            << dataHost[nFE * nSpokesPerFrame * coils] << ","
            << wHost[2 * nFE * nSpokesPerFrame] << ")" << std::endl;

  std::cout << "debug second tuple (x,y,v,w) : ("
            << traj[2 * nFE * nSpokesPerFrame + 1] << ","
            << traj[2 * nFE * nSpokesPerFrame + 1 + nFE * nSpokesPerFrame]
            << "," << dataHost[nFE * nSpokesPerFrame * coils + 1] << ","
            << wHost[2 * nFE * nSpokesPerFrame + 1] << ")" << std::endl;

  NoncartesianOperator nonCartOp(width, height, coils, frames,
                                 nSpokesPerFrame * frames, nFE, nSpokesPerFrame,
                                 ktraj, w, b1);

  CVector img(width * height * frames);
  nonCartOp.ForwardOperation(data, img, b1);

  // get result
  std::vector<CType> result(N);
  img.copyToHost(result);
  print("result:", 10, 10, 1, result, false);
  agile::writeVectorFile(output, result);
}

