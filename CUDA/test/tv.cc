#include <gtest/gtest.h>

#include <stdlib.h>  //< srand, rand
#include <time.h>    //< time

#include "agile/gpu_environment.hpp"
#include "agile/gpu_vector.hpp"
#include "agile/gpu_matrix.hpp"
#include "agile/calc/fft.hpp"
#include "agile/io/file.hpp"

#include "./test_utils.h"
#include "../include/types.h"
#include "../include/tv.h"
#include "../include/utils.h"
#include "../include/cartesian_operator.h"
#include "../include/noncartesian_operator.h"

class Test_TV : public ::testing::Test
{
 public:
  static void SetUpTestCase()
  {
    unsigned int max_cols = 10;
    unsigned int max_rows = 10;
    unsigned int max_slices = 5;

    // row major
    for (unsigned slice = 0; slice < max_slices; slice++)
      for (unsigned row = 0; row < max_rows; ++row)
        for (unsigned column = 0; column < max_cols; ++column)
          matrix_data.push_back(
              CType(XYZ2Lin(column, row, slice, max_cols, max_rows), 0));
  }
  virtual void SetUp()
  {
    agile::GPUEnvironment::allocateGPU(0);
  }
  static std::vector<CType> matrix_data;
};

std::vector<CType> Test_TV::matrix_data;

TEST_F(Test_TV, AgileSetup)
{
  agile::GPUEnvironment::allocateGPU(0);
  agile::GPUEnvironment::printInformation(std::cout);
  std::cout << std::endl;
}

TEST_F(Test_TV, ScalarProcuct)
{
  agile::GPUEnvironment::allocateGPU(0);

  std::vector<float> x_host, y_host;
  for (unsigned counter = 0; counter < 10; ++counter)
  {
    x_host.push_back(1.0f);
    y_host.push_back(2.0f);
  }

  print("x:", x_host);
  print("y:", y_host);

  agile::GPUVector<float> x, y;
  x.assignFromHost(x_host.begin(), x_host.end());
  y.assignFromHost(y_host.begin(), y_host.end());

  float sp = agile::getScalarProduct(x, y);

  EXPECT_NEAR(20.0f, sp, EPS);
}

TEST_F(Test_TV, GradientNorm)
{
  unsigned int width = 5;
  unsigned int height = 5;
  unsigned int frames = 2;
  unsigned int N = width * height * frames;

  print("Input: ", width, height, frames, matrix_data);

  agile::GPUVector<CType> data_gpu;
  data_gpu.assignFromHost(matrix_data.begin(), matrix_data.begin() + N);

  std::vector<CVector> gradient = utils::Gradient(data_gpu, width, height);

  CVector norm_gpu;
  norm_gpu = utils::GradientNorm(gradient);

  std::vector<CType> norm(N);
  norm_gpu.copyToHost(norm);

  print("gradient norm: ", width, height, frames, norm);
  EXPECT_NEAR(25.514, std::abs(norm[XYZ2Lin(0, 0, 0, width, height)]), EPS);
  EXPECT_NEAR(25.514, std::abs(norm[XYZ2Lin(1, 0, 0, width, height)]), EPS);
  EXPECT_NEAR(25.514, std::abs(norm[XYZ2Lin(1, 1, 0, width, height)]), EPS);
  EXPECT_NEAR(5.099, std::abs(norm[XYZ2Lin(1, 1, 1, width, height)]), EPS);
  EXPECT_NEAR(1.0, std::abs(norm[XYZ2Lin(1, 4, 1, width, height)]), EPS);
  EXPECT_NEAR(0.0, std::abs(norm[XYZ2Lin(4, 4, 1, width, height)]), EPS);
}

TEST_F(Test_TV, ScaledGradientNorm)
{
  unsigned int width = 5;
  unsigned int height = 5;
  unsigned int frames = 2;
  DType dx = 1.0;
  DType dy = 1.0;
  DType dz = 0.5;
  unsigned int N = width * height * frames;

  print("Input: ", width, height, frames, matrix_data);

  agile::GPUVector<CType> data_gpu;
  data_gpu.assignFromHost(matrix_data.begin(), matrix_data.begin() + N);

  std::vector<CVector> gradient =
      utils::Gradient(data_gpu, width, height, dx, dy, dz);

  CVector norm_gpu;
  norm_gpu = utils::GradientNorm(gradient);

  std::vector<CType> norm(N);
  norm_gpu.copyToHost(norm);

  print("gradient norm: ", width, height, frames, norm);

  EXPECT_NEAR(50.2593, std::abs(norm[XYZ2Lin(0, 0, 0, width, height)]), EPS);
  EXPECT_NEAR(50.2593, std::abs(norm[XYZ2Lin(1, 0, 0, width, height)]), EPS);
  EXPECT_NEAR(50.2593, std::abs(norm[XYZ2Lin(1, 1, 0, width, height)]), EPS);
  EXPECT_NEAR(50.0100, std::abs(norm[XYZ2Lin(1, 4, 0, width, height)]), EPS);
  EXPECT_NEAR(5.099, std::abs(norm[XYZ2Lin(1, 1, 1, width, height)]), EPS);
  EXPECT_NEAR(1.0, std::abs(norm[XYZ2Lin(1, 4, 1, width, height)]), EPS);
  EXPECT_NEAR(0.0, std::abs(norm[XYZ2Lin(4, 4, 1, width, height)]), EPS);
}

TEST_F(Test_TV, MAX)
{
  unsigned int width = 5;
  unsigned int height = 5;
  unsigned int coils = 1;
  unsigned int frames = 2;

  unsigned int N = width * height * frames;

  agile::GPUVector<CType> data_gpu;
  std::vector<CType> data;

  print("A", width, height, frames, coils, matrix_data);

  data_gpu.assignFromHost(matrix_data.begin(), matrix_data.begin() + N);

  agile::max(data_gpu, CType(20.0, 0.0), data_gpu);

  data_gpu.copyToHost(data);

  print("max(A,20)", width, height, frames, coils, data);

  for (unsigned i = 0; i < N; i++)
    EXPECT_EQ(std::max(i, 20u), std::abs(data[i]));
}

TEST_F(Test_TV, Proximal_Mapping)
{
  unsigned int width = 5;
  unsigned int height = 5;
  unsigned int coils = 1;
  unsigned int frames = 2;

  unsigned int N = width * height * frames;

  agile::GPUVector<CType> data_gpu(N);
  agile::GPUVector<CType> nominator_gpu(N);
  std::vector<CType> data;

  print("A", width, height, frames, coils, matrix_data);

  data_gpu.assignFromHost(matrix_data.begin(), matrix_data.begin() + N);

  agile::max(data_gpu, CType(5.0, 0.0), nominator_gpu);
  agile::divideElementwise(data_gpu, nominator_gpu, nominator_gpu);

  nominator_gpu.copyToHost(data);

  print("A./max(A,5)", width, height, frames, coils, data);
  for (unsigned i = 0; i < N; i++)
    EXPECT_EQ((float)i / std::max(i, 5u), std::abs(data[i]));
}

TEST_F(Test_TV, Divergence)
{
  unsigned int width = 5;
  unsigned int height = 5;
  unsigned int coils = 1;
  unsigned int frames = 2;

  unsigned int N = width * height * frames;

  print("A", width, height, frames, coils, matrix_data);

  CVector data_gpu(N);
  data_gpu.assignFromHost(matrix_data.begin(), matrix_data.begin() + N);

  std::vector<CVector> gradient = utils::Gradient(data_gpu, width, height);

  std::vector<CType> temp(N);

  CVector norm = utils::GradientNorm(gradient);

  agile::max(norm, CType(1.0f), norm);

  norm.copyToHost(temp);
  print("max(norm(grad(A)),1.0)", width, height, frames, coils, temp, false);

  agile::divideElementwise(gradient[0], norm, gradient[0]);
  agile::divideElementwise(gradient[1], norm, gradient[1]);
  agile::divideElementwise(gradient[2], norm, gradient[2]);

  CVector divergence = utils::Divergence(gradient, width, height, frames);

  std::vector<CType> data;
  divergence.copyToHost(data);

  print("div(grad(A)/max(1,|grad(A)|))", width, height, frames, coils, data,
        false);
  EXPECT_NEAR(1.215, data[XYZ2Lin(0, 0, 0, width, height)].real(), EPS);
  EXPECT_NEAR(1.019, data[XYZ2Lin(0, 1, 0, width, height)].real(), EPS);
  EXPECT_NEAR(1.019, data[XYZ2Lin(0, 2, 0, width, height)].real(), EPS);
  EXPECT_NEAR(0.843, data[XYZ2Lin(0, 4, 0, width, height)].real(), EPS);
  EXPECT_NEAR(0.1969, data[XYZ2Lin(0, 0, 1, width, height)].real(), EPS);
  EXPECT_NEAR(-0.7837, data[XYZ2Lin(0, 1, 1, width, height)].real(), EPS);
  EXPECT_NEAR(-3.000, data[XYZ2Lin(4, 4, 1, width, height)].real(), EPS);
}

TEST_F(Test_TV, AdaptLambda)
{
  unsigned int width = 6;
  unsigned int height = 6;
  unsigned int coils = 5;
  unsigned int frames = 2;

  unsigned N = width * height * frames;

  RVector mask(N);
  mask.assign(mask.size(), 1.0);

  BaseOperator *cartOp =
      new CartesianOperator(width, height, coils, frames, mask);
  TV tvSolver(width, height, coils, frames, cartOp);

  RType k = 0.4 * 0.2991;
  RType d = 10.0 * 0.2991;
  EXPECT_NEAR(3.1101, tvSolver.AdaptLambda(k, d), EPS);
  delete cartOp;
}

TEST_F(Test_TV, AdaptStepSize)
{
  unsigned int width = 6;
  unsigned int height = 5;
  unsigned int coils = 3;
  unsigned int frames = 2;

  unsigned N = width * height * frames;

  print("Input: ", width, height, frames, matrix_data);

  RVector mask(N);
  mask.assign(mask.size(), 1.0);

  // b1 map
  CVector b1_gpu(width * height * coils);
  b1_gpu.assign(width * height * coils, 1.0);
  agile::lowlevel::scale(CType(0.0, 2.0), b1_gpu.data() + width * height,
                         b1_gpu.data() + width * height, width * height);

  agile::GPUVector<CType> data_gpu;
  data_gpu.assignFromHost(matrix_data.begin(), matrix_data.begin() + N);
  agile::lowlevel::scale(CType(0.0, 1.0), data_gpu.data(), data_gpu.data(), N);

  BaseOperator *cartOp =
      new CartesianOperator(width, height, coils, frames, mask);
  TV tvSolver(width, height, coils, frames, cartOp);
  tvSolver.GetParams().sigma = 1.2;
  tvSolver.AdaptStepSize(data_gpu, b1_gpu);

  EXPECT_NEAR(0.3950, tvSolver.GetParams().sigma, EPS);
  EXPECT_NEAR(0.3950, tvSolver.GetParams().tau, EPS);
  delete cartOp;
}

TEST_F(Test_TV, ComputeTVNorm)
{
  unsigned int width = 6;
  unsigned int height = 5;
  unsigned int frames = 2;

  unsigned N = width * height * frames;
  print("Input: ", width, height, frames, matrix_data);

  agile::GPUVector<CType> data_gpu;
  data_gpu.assignFromHost(matrix_data.begin(), matrix_data.begin() + N);

  RType ds = 1;
  RType dt = 1;
  RType norm = utils::TVNorm(data_gpu, width, height, ds, ds, dt);
  EXPECT_NEAR(1065.324, norm, EPS);

  ds = 1.2;
  dt = 2;
  norm = utils::TVNorm(data_gpu, width, height, ds, ds, dt);
  EXPECT_NEAR(595.5739, norm, EPS);

  // Compute Gradient Norm
  agile::addScaledVector(data_gpu, CType(0, 2), data_gpu, data_gpu);
  norm = utils::TVNorm(data_gpu, width, height, ds, ds, dt);
  EXPECT_NEAR(1331.743, norm, EPS);
}

TEST_F(Test_TV, ComputeGStar)
{
  unsigned int width = 6;
  unsigned int height = 5;
  unsigned int coils = 3;
  unsigned int frames = 2;

  unsigned N = width * height * frames;
  for (unsigned frame = 0; frame < frames; frame++)
    for (unsigned coil = 0; coil < coils; coil++)
      for (unsigned row = 0; row < height; ++row)
        for (unsigned column = 0; column < width; ++column)
        {
          unsigned offset = frame * width * height * coils;
          matrix_data.push_back(
              CType(offset + XYZ2Lin(column, row, coil, width, height), 0));
        }

  RVector mask(N);
  mask.assign(mask.size(), 1.0);

  BaseOperator *cartOp =
      new CartesianOperator(width, height, coils, frames, mask, false);
  TV tvSolver(width, height, coils, frames, cartOp);
  tvSolver.GetParams().lambda = 1.5;

  // b1 map
  CVector b1_gpu(width * height * coils);
  b1_gpu.assign(width * height * coils, 1.0);
  agile::lowlevel::scale(CType(0.0, 2.0), b1_gpu.data() + width * height,
                         b1_gpu.data() + width * height, width * height);

  // data
  agile::GPUVector<CType> data_gpu;
  data_gpu.assignFromHost(matrix_data.begin(), matrix_data.begin() + N * coils);
  print("Input: ", width, height, frames, coils, matrix_data);

  // primal var
  agile::GPUVector<CType> x;
  x.assignFromHost(matrix_data.begin(), matrix_data.begin() + N);

  // dual vars
  agile::GPUVector<CType> z(N * coils), zTemp(N * coils);
  z.assign(N * coils, 1);
  zTemp.assign(N * coils, 0);

  std::vector<CVector> y;
  y.push_back(CVector(N));
  y.push_back(CVector(N));
  y.push_back(CVector(N));
  y[0].assign(N, 0);
  y[1].assign(N, 0);
  y[2].assign(N, 0);

  RType gstar = tvSolver.ComputeGStar(x, y, z, data_gpu, b1_gpu);
  EXPECT_NEAR(1727794, gstar, EPS);
  delete cartOp;
}

TEST_F(Test_TV, PDGap)
{
  unsigned int width = 6;
  unsigned int height = 5;
  unsigned int coils = 3;
  unsigned int frames = 2;

  unsigned N = width * height * frames;
  for (unsigned frame = 0; frame < frames; frame++)
    for (unsigned coil = 0; coil < coils; coil++)
      for (unsigned row = 0; row < height; ++row)
        for (unsigned column = 0; column < width; ++column)
        {
          unsigned offset = frame * width * height * coils;
          matrix_data.push_back(
              CType(offset + XYZ2Lin(column, row, coil, width, height), 0));
        }

  RVector mask(N);
  mask.assign(mask.size(), 1.0);

  BaseOperator *cartOp =
      new CartesianOperator(width, height, coils, frames, mask, false);
  TV tvSolver(width, height, coils, frames, cartOp);
  tvSolver.GetParams().lambda = 1.5;

  // b1 map
  CVector b1_gpu(width * height * coils);
  b1_gpu.assign(width * height * coils, 1.0);
  agile::lowlevel::scale(CType(0.0, 2.0), b1_gpu.data() + width * height,
                         b1_gpu.data() + width * height, width * height);

  // data
  agile::GPUVector<CType> data_gpu;
  data_gpu.assignFromHost(matrix_data.begin(), matrix_data.begin() + N * coils);
  print("Input: ", width, height, frames, coils, matrix_data);

  // primal var
  agile::GPUVector<CType> x;
  x.assignFromHost(matrix_data.begin(), matrix_data.begin() + N);

  // dual vars
  agile::GPUVector<CType> z(N * coils), zTemp(N * coils);
  z.assign(N * coils, 1);
  zTemp.assign(N * coils, 0);

  std::vector<CVector> y;
  y.push_back(CVector(N));
  y.push_back(CVector(N));
  y.push_back(CVector(N));
  y[0].assign(N, 0);
  y[1].assign(N, 0);
  y[2].assign(N, 0);

  RType PDGap = tvSolver.ComputePDGap(x, y, z, data_gpu, b1_gpu);
  EXPECT_NEAR(1728859.375, PDGap, EPS);
  delete cartOp;
}

TEST_F(Test_TV, Iteration)
{
  unsigned int width = 6;
  unsigned int height = 6;
  unsigned int coils = 2;
  unsigned int frames = 2;

  BaseOperator *cartOp = new CartesianOperator(width, height, coils, frames);
  TV tvSolver(width, height, coils, frames, cartOp);

  unsigned int N = width * height * frames;

  for (unsigned frame = 0; frame < frames; frame++)
    for (unsigned coil = 0; coil < coils; coil++)
      for (unsigned row = 0; row < height; ++row)
        for (unsigned column = 0; column < width; ++column)
        {
          unsigned offset = frame * width * height * coils;
          matrix_data.push_back(
              CType(offset + XYZ2Lin(column, row, coil, width, height), 0));
        }

  print("Input: ", width, height, coils, frames, matrix_data);

  // data
  CVector data_gpu;
  data_gpu.assignFromHost(matrix_data.begin(), matrix_data.end());

  // b1 map
  CVector b1_gpu(width * height * coils);
  b1_gpu.assign(width * height * coils, 1.0);
  agile::lowlevel::scale(CType(0.0, 2.0), b1_gpu.data() + width * height,
                         b1_gpu.data() + width * height, width * height);

  CVector x(N);
  x.assign(x.size(), 0.0);

  tvSolver.IterativeReconstruction(data_gpu, x, b1_gpu);

  // get result
  std::vector<CType> result(width * height * frames);
  x.copyToHost(result);
  print("result:", width, height, frames, result, true);
  EXPECT_NEAR(130.1532, std::abs(result[XYZ2Lin(3, 3, 0, width, height)]), EPS);
  EXPECT_NEAR(16.0557, std::abs(result[XYZ2Lin(3, 4, 0, width, height)]), EPS);
  EXPECT_NEAR(319.6157, std::abs(result[XYZ2Lin(3, 3, 1, width, height)]), EPS);
  EXPECT_NEAR(16.0616, std::abs(result[XYZ2Lin(3, 4, 1, width, height)]), EPS);
  delete cartOp;
}

TEST(Test_TV_Real, DISABLED_Iteration)
{
  agile::GPUEnvironment::allocateGPU(0);
  agile::GPUEnvironment::printInformation(std::cout);
  unsigned int height = 416;
  unsigned int width = 168;
  unsigned int coils = 30;
  unsigned int frames = 25;

  unsigned int N = width * height * frames;

  const char *output = "../test/data/output/test_vector_data_416_168_25.bin";

  // data
  std::vector<CType> data;
  agile::readVectorFile("../test/data/test_data_416_168_30_25.bin", data);
  print("Data: ", 20, 1, 1, 1, data, false);

  CVector data_gpu(N * coils);
  data_gpu.assignFromHost(data.begin(), data.end());

  // b1 map
  std::vector<CType> b1;
  agile::readVectorFile("../test/data/test_b1_416_168_30.bin", b1);
  print("B1: ", 20, 1, 1, b1, false);

  CVector b1_gpu(width * height * coils);
  b1_gpu.assignFromHost(b1.begin(), b1.end());

  // kspace mask
  std::vector<float> mask;
  agile::readVectorFile("../test/data/test_mask_416_168_25.bin", mask);
  print("Mask: ", 20, 1, 1, mask);

  agile::GPUVector<float> mask_gpu(N);
  mask_gpu.assignFromHost(mask.begin(), mask.end());

  // x0
  std::vector<float> x0;
  agile::readVectorFile("../test/data/test_x0_416_168_30.bin", x0);
  CVector x0_gpu(width * height);
  x0_gpu.assignFromHost(x0.end() - width * height, x0.end());

  CVector x_gpu(N);
  for (unsigned frame = 0; frame < frames; frame++)
  {
    //    x_gpu.assign(x_gpu.size(), 0.0f);
    agile::lowlevel::scale(1.0f, x0_gpu.data(),
                           x_gpu.data() + frame * width * height,
                           width * height);
  }

  BaseOperator *cartOp =
      new CartesianOperator(width, height, coils, frames, mask_gpu, false);
  TV tvSolver(width, height, coils, frames, cartOp);

  tvSolver.IterativeReconstruction(data_gpu, x_gpu, b1_gpu);

  // get result
  std::vector<CType> result(N);
  x_gpu.copyToHost(result);
  print("result:", 10, 10, 1, result, false);
  agile::writeVectorFile(output, result);
  delete cartOp;
}

TEST(Test_TV_NonCart_Real, DISABLED_Iteration)
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

  const char *output =
      "../test/data/output/noncart/test_tv_recon_384_384_14.bin";

  // data
  std::vector<CType> data;
  agile::readVectorFile("../test/data/noncart/test_data_384_21_12_14.bin",
                        data);
  print("Data: ", 20, 1, 1, 1, data, false);

  CVector data_gpu(N * coils);
  data_gpu.assignFromHost(data.begin(), data.end());

  // b1 map
  std::vector<CType> b1;
  agile::readVectorFile("../test/data/noncart/test_b1_384_384_12.bin", b1);
  print("B1: ", 20, 1, 1, b1, false);

  CVector b1_gpu(width * height * coils);
  b1_gpu.assignFromHost(b1.begin(), b1.end());

  // kspace mask
  std::vector<RType> traj;
  agile::readVectorFile("../test/data/noncart/test_k_384_21_14.bin", traj);
  print("Traj: ", 20, 1, 1, traj);

  unsigned int nTraj = frames * nSpokesPerFrame * nFE;
  RVector ktraj(2 * nTraj);
  ktraj.assignFromHost(traj.begin(), traj.end());

  // density data
  std::vector<RType> wHost;
  agile::readVectorFile("../test/data/noncart/test_w_384_21_14.bin", wHost);
  print("Density: ", 20, 1, 1, wHost);

  RVector w(nTraj);
  w.assignFromHost(wHost.begin(), wHost.end());

  // x0
  std::vector<CType> x0;
  agile::readVectorFile("../test/data/noncart/test_u0_384_384.bin", x0);
  CVector x0_gpu(width * height);
  x0_gpu.assignFromHost(x0.end() - width * height, x0.end());

  CVector x_gpu(N);
  for (unsigned frame = 0; frame < frames; frame++)
  {
    //    x_gpu.assign(x_gpu.size(), 0.0f);
    agile::lowlevel::scale(1.0f, x0_gpu.data(),
                           x_gpu.data() + frame * width * height,
                           width * height);
  }

  BaseOperator *nonCartOp = new NoncartesianOperator(
      width, height, coils, frames, nSpokesPerFrame * frames, nFE,
      nSpokesPerFrame, ktraj, w, b1_gpu);

  TV tvSolver(width, height, coils, frames, nonCartOp);

  tvSolver.GetParams().ds = 1.3;
  tvSolver.GetParams().dt = 0.2;

  tvSolver.IterativeReconstruction(data_gpu, x_gpu, b1_gpu);

  // get result
  std::vector<CType> result(N);
  x_gpu.copyToHost(result);
  print("result:", 10, 10, 1, result, false);
  agile::writeVectorFile(output, result);
  delete nonCartOp;
}

