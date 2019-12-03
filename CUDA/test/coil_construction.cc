#include <gtest/gtest.h>

#include "../include/types.h"
#include "./test_utils.h"
#include "../include/ictgv2.h"
#include "../include/utils.h"
#include "../include/cartesian_coil_construction.h"
#include "../include/noncartesian_coil_construction.h"

#include "agile/io/file.hpp"
#include "agile/operator/cg.hpp"

class Test_CoilConstruction : public ::testing::Test
{
 public:
  static const unsigned int width = 5;
  static const unsigned int height = 5;
  static const unsigned int coils = 3;
  static const unsigned int frames = 2;
  static const unsigned int N = width * height * frames;

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

    // data
    kdata = CVector(N * coils);
    kdata.assignFromHost(matrix_data.begin(), matrix_data.begin() + N * coils);

    // b1 map
    b1 = CVector(width * height * coils);
    b1.assign(width * height * coils, 0.0);

    // mask 1.0
    mask = RVector(N);
    mask.assign(N, 1.0);

    cartOp = new CartesianOperator(width, height, coils, frames, mask, false);
    coilConstruction =
        new CartesianCoilConstruction(width, height, coils, frames, cartOp);
  }

  virtual void ShutDown()
  {
    delete cartOp;
    delete coilConstruction;
  }

  std::vector<CType> matrix_data;
  CVector kdata;
  CVector b1;
  RVector mask;
  CoilConstruction *coilConstruction;
  CartesianOperator *cartOp;
};

TEST_F(Test_CoilConstruction, StandardReconstruction)
{
  print("Input: ", width, height, coils, frames, matrix_data);

  CVector u(width * height);
  CVector crec(width * height * coils);
  u.assign(u.size(), 0);
  crec.assign(crec.size(), 0);

  coilConstruction->TimeAveragedReconstruction(kdata, u, crec);

  std::vector<CType> uHost(width * height);
  u.copyToHost(uHost);
  print("u:", width, height, uHost, false);
  EXPECT_NEAR(668.968, std::real(uHost[XY2Lin(0, 0, width)]), EPS);
  EXPECT_NEAR(-4.33, std::real(uHost[XY2Lin(1, 0, width)]), EPS);
  EXPECT_NEAR(-21.65, std::real(uHost[XY2Lin(0, 1, width)]), EPS);

  std::vector<CType> cresHost(width * height * coils);
  crec.copyToHost(cresHost);
  print("Cres:", width, height, coils, cresHost, false);
  EXPECT_NEAR(247.50, std::real(cresHost[XYZ2Lin(0, 0, 0, width, height)]),
              EPS);
  EXPECT_NEAR(-2.50, std::real(cresHost[XYZ2Lin(1, 0, 1, width, height)]), EPS);
  EXPECT_NEAR(-12.50, std::real(cresHost[XYZ2Lin(0, 1, 2, width, height)]),
              EPS);
}

TEST_F(Test_CoilConstruction, StandardNonCartReconstruction)
{
  unsigned nSpokes = 4;
  unsigned nFE = width;
  unsigned nTraj = nSpokes * nFE;
  unsigned spokesPerFrame = 2;
  unsigned nSamplesPerFrame = nTraj / spokesPerFrame;
  RVector ktraj;
  RVector w(nTraj);

  // init k-space trajectory
  std::vector<RType> trajectory_data(2 * nTraj);
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
  ktraj.assignFromHost(trajectory_data.begin(), trajectory_data.end());
  w.assign(nTraj, 1.0);

  NoncartesianOperator *noncartOp = new NoncartesianOperator(
      width, height, coils, frames, nSpokes, nFE, spokesPerFrame, ktraj, w);
  NoncartesianCoilConstruction *nonCartCoilConstruction =
      new NoncartesianCoilConstruction(width, height, coils, frames, noncartOp);

  CVector u(width * height);
  CVector crec(width * height * coils);
  u.assign(u.size(), 0);
  crec.assign(crec.size(), 0);

  kdata = CVector(nTraj * coils);
  kdata.assign(nTraj * coils, 0.0);
  kdata.assignFromHost(matrix_data.begin(),
                       matrix_data.begin() + nTraj * coils);
  print("Input: ", width, height, coils, frames, matrix_data);

  nonCartCoilConstruction->TimeAveragedReconstruction(kdata, u, crec);

  std::vector<CType> uHost(width * height);
  u.copyToHost(uHost);
  print("u:", width, height, uHost, true);
  EXPECT_NEAR(2.734E-6, std::abs(uHost[XY2Lin(0, 0, width)]), EPS);
  EXPECT_NEAR(5.104E-6, std::abs(uHost[XY2Lin(1, 0, width)]), EPS);
  EXPECT_NEAR(0.0607, std::abs(uHost[XY2Lin(0, 1, width)]), EPS);

  std::vector<CType> cresHost(width * height * coils);
  crec.copyToHost(cresHost);
  print("Cres:", width, height, coils, cresHost, false);
  EXPECT_NEAR(1.2227E-6, std::real(cresHost[XYZ2Lin(0, 0, 0, width, height)]),
              EPS);
  EXPECT_NEAR(1.209E-6, std::real(cresHost[XYZ2Lin(1, 0, 1, width, height)]),
              EPS);
  EXPECT_NEAR(-0.0333, std::real(cresHost[XYZ2Lin(0, 1, 2, width, height)]),
              EPS);
  delete nonCartCoilConstruction;
  delete noncartOp;
}

TEST_F(Test_CoilConstruction, LaplaceOperator)
{
  unsigned N = width * height;
  print("Input: ", width, height, 1, matrix_data);

  CVector divergence(N);
  CVector temp(N);
  std::vector<CVector> gradient;
  gradient.push_back(CVector(N));
  gradient.push_back(CVector(N));

  agile::lowlevel::diff3(1, width, height, kdata.data(), gradient[0].data(), N,
                         true);
  agile::lowlevel::diff3(2, width, height, kdata.data(), gradient[1].data(), N,
                         true);

  agile::lowlevel::bdiff3(1, width, height, gradient[0].data(),
                          divergence.data(), N, true);

  agile::lowlevel::bdiff3(2, width, height, gradient[1].data(), temp.data(), N,
                          true);

  agile::addVector(temp, divergence, divergence);
  agile::scale((DType)-1.0, divergence, divergence);

  std::vector<CType> res(N);
  divergence.copyToHost(res);
  print("div(grad(x)): ", width, height, 1, res, false);

  EXPECT_NEAR(-30.0, std::real(res[XY2Lin(0, 0, width)]), EPS);
  EXPECT_NEAR(-25.0, std::real(res[XY2Lin(1, 0, width)]), EPS);
  EXPECT_NEAR(-5.0, std::real(res[XY2Lin(0, 1, width)]), EPS);
  EXPECT_NEAR(5.0, std::real(res[XY2Lin(4, 1, width)]), EPS);
  EXPECT_NEAR(20.0, std::real(res[XY2Lin(0, 4, width)]), EPS);
  EXPECT_NEAR(30.0, std::real(res[XY2Lin(4, 4, width)]), EPS);
}

TEST_F(Test_CoilConstruction, H1Regularization)
{
  communicator_type com;
  com.allocateGPU();

  agile::GPUEnvironment::printInformation(std::cout);
  std::cout << std::endl;

  CVector u(width * height);
  CVector crec(width * height * coils);
  u.assign(u.size(), 0);
  crec.assign(crec.size(), 0);

  coilConstruction->TimeAveragedReconstruction(kdata, u, crec);

  // TODO check if really abs(u) shall be passed
  agile::multiplyConjElementwise(u, u, u);
  agile::sqrt(u, u);

  CVector b1(width * height * coils);
  coilConstruction->B1FromUH1(u, crec, com, b1);

  std::vector<CType> xHost(N * coils);
  b1.copyToHost(xHost);
  print("x:", width, height, coils, xHost);

  EXPECT_NEAR(0.3718, std::abs(xHost[XYZ2Lin(0, 0, 0, width, height)]), EPS);
  EXPECT_NEAR(0.3747, std::abs(xHost[XYZ2Lin(0, 3, 0, width, height)]), EPS);
  EXPECT_NEAR(0.5567, std::abs(xHost[XYZ2Lin(2, 2, 1, width, height)]), EPS);
  EXPECT_NEAR(0.5567, std::abs(xHost[XYZ2Lin(4, 2, 1, width, height)]), EPS);
  EXPECT_NEAR(0.7410, std::abs(xHost[XYZ2Lin(4, 4, 2, width, height)]), EPS);
}

TEST_F(Test_CoilConstruction, InitialIndex)
{
  communicator_type com;
  com.allocateGPU();

  agile::GPUEnvironment::printInformation(std::cout);
  std::cout << std::endl;

  CVector u(width * height);
  CVector crec(width * height * coils);
  u.assign(u.size(), 0);
  crec.assign(crec.size(), 0);

  coilConstruction->TimeAveragedReconstruction(kdata, u, crec);

  // TODO check if really abs(u) shall be passed
  agile::multiplyConjElementwise(u, u, u);
  agile::sqrt(u, u);

  CVector b1(width * height * coils);
  coilConstruction->B1FromUH1(u, crec, com, b1);

  // find absolute sum of each coil and maximum index
  unsigned maxInd = coilConstruction->FindMaximumSum(crec);

  EXPECT_EQ(2u, maxInd);
}

TEST_F(Test_CoilConstruction, UReconWithTGV2Regularization)
{
  communicator_type com;
  com.allocateGPU();

  agile::GPUEnvironment::printInformation(std::cout);
  std::cout << std::endl;

  CVector u(width * height);
  CVector crec(width * height * coils);
  u.assign(u.size(), 0);
  crec.assign(crec.size(), 0);

  coilConstruction->TimeAveragedReconstruction(kdata, u, crec);

  // TODO check if really abs(u) shall be passed
  agile::multiplyConjElementwise(u, u, u);
  agile::sqrt(u, u);

  CVector absb1(width * height * coils);
  coilConstruction->B1FromUH1(u, crec, com, absb1);

  // find absolute sum of each coil and maximum index
  unsigned maxInd = coilConstruction->FindMaximumSum(crec);

  unsigned N = width * height;

  CVector uMax(N);
  uMax.assign(N, 0.0);

  CVector b1Max(width * height);
  utils::GetSubVector(absb1, b1Max, maxInd, N);

  CVector crecMax(width * height);
  utils::GetSubVector(crec, crecMax, maxInd, N);

  std::vector<unsigned> inds;
  inds.push_back(0);
  coilConstruction->UTGV2Recon(b1Max, crecMax, uMax, inds);

  std::vector<CType> uHost(N);
  uMax.copyToHost(uHost);
  print("u", width, height, 1, uHost, false);

  EXPECT_NEAR(669.533, std::real(uHost[XY2Lin(0, 0, width)]), EPS);
  EXPECT_NEAR(-2.952, std::real(uHost[XY2Lin(1, 0, width)]), EPS);
  EXPECT_NEAR(-16.433, std::real(uHost[XY2Lin(0, 1, width)]), EPS);
  EXPECT_NEAR(-0.115, std::real(uHost[XY2Lin(4, 4, width)]), EPS);
}

TEST_F(Test_CoilConstruction, B1ReconWithQuadraticRegularization)
{
  communicator_type com;
  com.allocateGPU();

  agile::GPUEnvironment::printInformation(std::cout);
  std::cout << std::endl;

  CVector u(width * height);
  CVector crec(width * height * coils);
  u.assign(u.size(), 0);
  crec.assign(crec.size(), 0);

  coilConstruction->TimeAveragedReconstruction(kdata, u, crec);

  // TODO check if really abs(u) shall be passed
  agile::multiplyConjElementwise(u, u, u);
  agile::sqrt(u, u);

  CVector absb1(width * height * coils);
  coilConstruction->B1FromUH1(u, crec, com, absb1);

  // find absolute sum of each coil and maximum index
  unsigned maxInd = coilConstruction->FindMaximumSum(crec);

  unsigned N = width * height;

  CVector uMax(N);
  uMax.assign(N, 0.0);

  CVector b1Max(width * height);
  utils::GetSubVector(absb1, b1Max, maxInd, N);

  CVector crecMax(width * height);
  utils::GetSubVector(crec, crecMax, maxInd, N);

  std::vector<unsigned> inds;
  inds.push_back(0);
  coilConstruction->UTGV2Recon(b1Max, crecMax, uMax, inds);

  // use coil 2 for now
  utils::GetSubVector(absb1, b1Max, 1, N);
  utils::GetSubVector(crec, crecMax, 1, N);
  agile::multiplyElementwise(b1Max, uMax, uMax);

  CVector x1(N);
  x1.assign(N, 0.0);

  coilConstruction->B1Recon(uMax, crecMax, x1, 1);

  // Get Phase
  RVector angleReal(N);
  CVector angle(N);
  agile::phaseVector(x1, angleReal);
  agile::scale(CType(0, 1.0), angleReal, angle);
  agile::expVector(angle, angle);
  agile::multiplyElementwise(b1Max, angle, x1);

  std::vector<CType> b1Host(N);
  x1.copyToHost(b1Host);
  print("b1", width, height, 1, b1Host, false);

  EXPECT_NEAR(0.5572, std::real(b1Host[XY2Lin(0, 0, width)]), EPS);
  EXPECT_NEAR(0.5575, std::real(b1Host[XY2Lin(1, 0, width)]), EPS);
  EXPECT_NEAR(0.5576, std::real(b1Host[XY2Lin(0, 1, width)]), EPS);
  EXPECT_NEAR(0.5576, std::real(b1Host[XY2Lin(4, 4, width)]), EPS);
}

TEST_F(Test_CoilConstruction, U1B1ComputationLoop)
{
  communicator_type com;
  com.allocateGPU();

  agile::GPUEnvironment::printInformation(std::cout);
  std::cout << std::endl;

  CVector u(N * coils);
  u.assign(N * coils, 0.0);
  CVector b1(N * coils);
  b1.assign(N * coils, 0.0);

  coilConstruction->PerformCoilConstruction(kdata, u, b1, com);

  std::vector<CType> b1Host(N * coils);
  b1.copyToHost(b1Host);
  print("b1", width, height, coils, b1Host, false);

  EXPECT_NEAR(0.3714, std::real(b1Host[XYZ2Lin(0, 0, 0, width, height)]), EPS);
  EXPECT_NEAR(0.5594, std::real(b1Host[XYZ2Lin(2, 2, 1, width, height)]), EPS);
  EXPECT_NEAR(0.7336, std::real(b1Host[XYZ2Lin(4, 4, 2, width, height)]), EPS);

  std::vector<CType> uHost(N * coils);
  u.copyToHost(uHost);
  print("u", width, height, coils, uHost, false);
  EXPECT_NEAR(669.533, std::real(uHost[XYZ2Lin(0, 0, 0, width, height)]), EPS);
  EXPECT_NEAR(-0.07361, std::real(uHost[XYZ2Lin(2, 2, 1, width, height)]), EPS);
  EXPECT_NEAR(-0.06341, std::real(uHost[XYZ2Lin(4, 4, 2, width, height)]), EPS);
}

TEST(Test_CoilConstruction_Real, DISABLED_H1Regularization)
{
  communicator_type com;
  com.allocateGPU();

  agile::GPUEnvironment::printInformation(std::cout);
  std::cout << std::endl;

  unsigned int height = 416;
  unsigned int width = 168;
  unsigned int coils = 30;
  unsigned int frames = 25;

  unsigned int N = width * height;

  const char *uOutput = "../test/data/output/u_recon_416_168_30.bin";
  const char *b1Output = "../test/data/output/b1_recon_416_168_30.bin";

  // kdata
  std::vector<CType> kdataHost;
  agile::readVectorFile("../test/data/test_data_416_168_30_25.bin", kdataHost);
  print("KData: ", 20, 1, 1, 1, kdataHost, false);
  CVector kdata;
  kdata.assignFromHost(kdataHost.begin(), kdataHost.end());

  // kspace mask
  std::vector<float> maskHost;
  agile::readVectorFile("../test/data/test_mask_416_168_25.bin", maskHost);
  print("Mask: ", 20, 1, 1, maskHost);
  RVector mask;
  mask.assignFromHost(maskHost.begin(), maskHost.end());

  CVector u(N * coils);
  u.assign(N * coils, 0.0);
  CVector b1(N * coils);
  b1.assign(N * coils, 0.0);

  CartesianOperator *cartOp =
      new CartesianOperator(width, height, coils, frames, mask, true);

  CartesianCoilConstruction coilConstruction(width, height, coils, frames,
                                             cartOp);
  coilConstruction.PerformCoilConstruction(kdata, u, b1, com);

  std::vector<CType> b1Host(N * coils);
  b1.copyToHost(b1Host);
  print("b1:", 20, 1, 1, b1Host);
  agile::writeVectorFile(b1Output, b1Host);

  std::vector<CType> uHost(N * coils);
  u.copyToHost(uHost);
  print("u:", 20, 1, 1, uHost);
  agile::writeVectorFile(uOutput, uHost);
}

