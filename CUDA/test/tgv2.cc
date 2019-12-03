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
#include "../include/tgv2.h"
#include "../include/utils.h"
#include "../include/cartesian_operator.h"
#include "../include/noncartesian_operator.h"

class Test_TGV : public ::testing::Test
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

std::vector<CType> Test_TGV::matrix_data;

TEST_F(Test_TGV, GradientNorm)
{
  unsigned int width = 5;
  unsigned int height = 5;
  unsigned int frames = 3;
  unsigned int N = width * height * frames;

  print("Input: ", width, height, 4 * frames, matrix_data);

  CVector data1_gpu;
  data1_gpu.assignFromHost(matrix_data.begin(), matrix_data.begin() + N);

  std::vector<CVector> y1 = utils::Gradient(data1_gpu, width, height);

  std::vector<CVector> data2_gpu(3);

  data2_gpu[0].assignFromHost(matrix_data.begin() + N,
                              matrix_data.begin() + 2 * N);
  data2_gpu[1].assignFromHost(matrix_data.begin() + 2 * N,
                              matrix_data.begin() + 3 * N);
  data2_gpu[2].assignFromHost(matrix_data.begin() + 3 * N,
                              matrix_data.begin() + 4 * N);

  agile::subVector(y1[0], data2_gpu[0], y1[0]);
  agile::subVector(y1[1], data2_gpu[1], y1[1]);
  agile::subVector(y1[2], data2_gpu[2], y1[2]);

  std::vector<CVector> y2 = utils::SymmetricGradient(data2_gpu, width, height);

  CVector norm_gpu;
  norm_gpu = utils::GradientNorm(y1);

  std::vector<CType> norm(N);
  norm_gpu.copyToHost(norm);
  print("gradient norm: ", width, height, frames, norm);

  EXPECT_NEAR(266.018, std::abs(norm[XYZ2Lin(0, 1, 0, width, height)]), EPS);
  EXPECT_NEAR(277.468, std::abs(norm[XYZ2Lin(2, 2, 0, width, height)]), EPS);
  EXPECT_NEAR(300.421, std::abs(norm[XYZ2Lin(4, 4, 0, width, height)]), EPS);

  EXPECT_NEAR(307.149, std::abs(norm[XYZ2Lin(0, 1, 1, width, height)]), EPS);
  EXPECT_NEAR(318.769, std::abs(norm[XYZ2Lin(2, 2, 1, width, height)]), EPS);
  EXPECT_NEAR(342.020, std::abs(norm[XYZ2Lin(4, 4, 1, width, height)]), EPS);

  EXPECT_NEAR(367.479, std::abs(norm[XYZ2Lin(0, 1, 2, width, height)]), EPS);
  EXPECT_NEAR(379.096, std::abs(norm[XYZ2Lin(2, 2, 2, width, height)]), EPS);
  EXPECT_NEAR(402.216, std::abs(norm[XYZ2Lin(4, 4, 2, width, height)]), EPS);

  // Compute Norm of Symmetric Gradient
  norm_gpu = utils::SymmetricGradientNorm(y2);

  norm_gpu.copyToHost(norm);
  print("symmetric gradient norm: ", width, height, frames, norm);

  EXPECT_NEAR(364.657, std::abs(norm[XYZ2Lin(0, 1, 0, width, height)]), EPS);
  EXPECT_NEAR(272.083, std::abs(norm[XYZ2Lin(2, 2, 0, width, height)]), EPS);
  EXPECT_NEAR(386.472, std::abs(norm[XYZ2Lin(4, 4, 0, width, height)]), EPS);

  EXPECT_NEAR(261.605, std::abs(norm[XYZ2Lin(0, 1, 1, width, height)]), EPS);
  EXPECT_NEAR(38.170, std::abs(norm[XYZ2Lin(2, 2, 1, width, height)]), EPS);
  EXPECT_NEAR(405.159, std::abs(norm[XYZ2Lin(4, 4, 1, width, height)]), EPS);

  EXPECT_NEAR(366.912, std::abs(norm[XYZ2Lin(0, 1, 2, width, height)]), EPS);
  EXPECT_NEAR(302.341, std::abs(norm[XYZ2Lin(2, 2, 2, width, height)]), EPS);
  EXPECT_NEAR(650.270, std::abs(norm[XYZ2Lin(4, 4, 2, width, height)]), EPS);
}

TEST_F(Test_TGV, ScaledGradientNorm)
{
  unsigned int width = 5;
  unsigned int height = 5;
  unsigned int frames = 3;
  DType dx = 1.5;
  DType dy = 1.5;
  DType dz = 0.5;
  unsigned int N = width * height * frames;

  print("Input: ", width, height, 4 * frames, matrix_data);

  CVector data1_gpu;
  data1_gpu.assignFromHost(matrix_data.begin(), matrix_data.begin() + N);

  std::vector<CVector> y1 =
      utils::Gradient(data1_gpu, width, height, dx, dy, dz);

  std::vector<CVector> data2_gpu(3);

  data2_gpu[0].assignFromHost(matrix_data.begin() + N,
                              matrix_data.begin() + 2 * N);
  data2_gpu[1].assignFromHost(matrix_data.begin() + 2 * N,
                              matrix_data.begin() + 3 * N);
  data2_gpu[2].assignFromHost(matrix_data.begin() + 3 * N,
                              matrix_data.begin() + 4 * N);

  agile::subVector(y1[0], data2_gpu[0], y1[0]);
  agile::subVector(y1[1], data2_gpu[1], y1[1]);
  agile::subVector(y1[2], data2_gpu[2], y1[2]);

  std::vector<CVector> y2 =
      utils::SymmetricGradient(data2_gpu, width, height, dx, dy, dz);

  CVector norm_gpu;
  norm_gpu = utils::GradientNorm(y1);

  std::vector<CType> norm(N);
  norm_gpu.copyToHost(norm);
  print("gradient norm: ", width, height, frames, norm);

  EXPECT_NEAR(248.387, std::abs(norm[XYZ2Lin(0, 1, 0, width, height)]), EPS);
  EXPECT_NEAR(259.995, std::abs(norm[XYZ2Lin(2, 2, 0, width, height)]), EPS);
  EXPECT_NEAR(282.272, std::abs(norm[XYZ2Lin(4, 4, 0, width, height)]), EPS);

  EXPECT_NEAR(290.037, std::abs(norm[XYZ2Lin(0, 1, 1, width, height)]), EPS);
  EXPECT_NEAR(301.782, std::abs(norm[XYZ2Lin(2, 2, 1, width, height)]), EPS);
  EXPECT_NEAR(324.273, std::abs(norm[XYZ2Lin(4, 4, 1, width, height)]), EPS);

  EXPECT_NEAR(368.505, std::abs(norm[XYZ2Lin(0, 1, 2, width, height)]), EPS);
  EXPECT_NEAR(380.128, std::abs(norm[XYZ2Lin(2, 2, 2, width, height)]), EPS);
  EXPECT_NEAR(402.216, std::abs(norm[XYZ2Lin(4, 4, 2, width, height)]), EPS);

  // Compute Norm of Symmetric Gradient
  norm_gpu = utils::SymmetricGradientNorm(y2);

  norm_gpu.copyToHost(norm);
  print("symmetric gradient norm: ", width, height, frames, norm);

  EXPECT_NEAR(564.200, std::abs(norm[XYZ2Lin(0, 1, 0, width, height)]), EPS);
  EXPECT_NEAR(541.777, std::abs(norm[XYZ2Lin(2, 2, 0, width, height)]), EPS);
  EXPECT_NEAR(546.366, std::abs(norm[XYZ2Lin(4, 4, 0, width, height)]), EPS);

  EXPECT_NEAR(201.590, std::abs(norm[XYZ2Lin(0, 1, 1, width, height)]), EPS);
  EXPECT_NEAR(72.286, std::abs(norm[XYZ2Lin(2, 2, 1, width, height)]), EPS);
  EXPECT_NEAR(255.651, std::abs(norm[XYZ2Lin(4, 4, 1, width, height)]), EPS);

  EXPECT_NEAR(584.208, std::abs(norm[XYZ2Lin(0, 1, 2, width, height)]), EPS);
  EXPECT_NEAR(606.820, std::abs(norm[XYZ2Lin(2, 2, 2, width, height)]), EPS);
  EXPECT_NEAR(798.410, std::abs(norm[XYZ2Lin(4, 4, 2, width, height)]), EPS);
}

TEST_F(Test_TGV, Divergence)
{
  unsigned int width = 5;
  unsigned int height = 5;
  unsigned int frames = 3;
  float alpha0 = 1.0;
  float alpha1 = std::sqrt(2.0);
  DType dx = 1.5;
  DType dy = 1.5;
  DType dz = 0.5;
  unsigned int N = width * height * frames;

  print("Input: ", width, height, 4 * frames, matrix_data);

  CVector data1_gpu;
  data1_gpu.assignFromHost(matrix_data.begin(), matrix_data.begin() + N);

  std::vector<CVector> y1 =
      utils::Gradient(data1_gpu, width, height, dx, dy, dz);

  std::vector<CVector> data2_gpu(3);

  data2_gpu[0].assignFromHost(matrix_data.begin() + N,
                              matrix_data.begin() + 2 * N);
  data2_gpu[1].assignFromHost(matrix_data.begin() + 2 * N,
                              matrix_data.begin() + 3 * N);
  data2_gpu[2].assignFromHost(matrix_data.begin() + 3 * N,
                              matrix_data.begin() + 4 * N);

  agile::subVector(y1[0], data2_gpu[0], y1[0]);
  agile::subVector(y1[1], data2_gpu[1], y1[1]);
  agile::subVector(y1[2], data2_gpu[2], y1[2]);

  // Gradient of p and prox operator
  CVector norm_gpu = utils::GradientNorm(y1);
  agile::scale((DType)1.0 / alpha1, norm_gpu, norm_gpu);
  agile::max(norm_gpu, CType(1.0f), norm_gpu);
  for (int cnt = 0; cnt < 3; cnt++)
    agile::divideElementwise(y1[cnt], norm_gpu, y1[cnt]);

  // Gradient of q and prox operator
  std::vector<CVector> y2 =
      utils::SymmetricGradient(data2_gpu, width, height, dx, dy, dz);
  CVector norm2_gpu = utils::SymmetricGradientNorm(y2);
  agile::scale((DType)1.0 / alpha0, norm2_gpu, norm2_gpu);
  agile::max(norm2_gpu, CType(1.0f), norm2_gpu);
  for (int cnt = 0; cnt < 6; cnt++)
    agile::divideElementwise(y2[cnt], norm2_gpu, y2[cnt]);

  CVector divergence = utils::Divergence(y1, width, height, frames, dx, dy, dz);
  std::vector<CType> data;
  divergence.copyToHost(data);
  print("div_3_3(grad(A)/max(1,|grad(A)|))", width, height, frames, data,
        false);

  std::vector<CVector> symDiv =
      utils::SymmetricDivergence(y2, width, height, frames, dx, dy, dz);

  std::vector<CType> symData[3];
  for (int cnt = 0; cnt < 3; cnt++)
  {
    symDiv[cnt].copyToHost(symData[cnt]);
    print("div_3_6(grad(A)/max(1,|grad(A)|))", width, height, frames,
          symData[cnt], false);
  }

  EXPECT_NEAR(0.1819, std::abs(symData[0][XYZ2Lin(0, 0, 0, width, height)]),
              EPS);
  EXPECT_NEAR(0.3152,
              std::abs(symData[0][XYZ2Lin(width - 1, 0, 1, width, height)]),
              EPS);
  EXPECT_NEAR(0.091,
              std::abs(symData[0][XYZ2Lin(0, height - 1, 2, width, height)]),
              EPS);
  EXPECT_NEAR(0, std::abs(symData[0][XYZ2Lin(width - 1, height - 1, frames - 1,
                                             width, height)]),
              EPS);

  EXPECT_NEAR(0.1485, std::abs(symData[1][XYZ2Lin(0, 0, 0, width, height)]),
              EPS);
  EXPECT_NEAR(1.554,
              std::abs(symData[1][XYZ2Lin(width - 1, 0, 1, width, height)]),
              EPS);
  EXPECT_NEAR(0.0685,
              std::abs(symData[1][XYZ2Lin(0, height - 1, 2, width, height)]),
              EPS);
  EXPECT_NEAR(0, std::abs(symData[1][XYZ2Lin(width - 1, height - 1, frames - 1,
                                             width, height)]),
              EPS);

  EXPECT_NEAR(1.2661, std::abs(symData[2][XYZ2Lin(0, 0, 0, width, height)]),
              EPS);
  EXPECT_NEAR(2.2851,
              std::abs(symData[2][XYZ2Lin(width - 1, 0, 1, width, height)]),
              EPS);
  EXPECT_NEAR(0.0910,
              std::abs(symData[2][XYZ2Lin(0, height - 1, 2, width, height)]),
              EPS);
  EXPECT_NEAR(0, std::abs(symData[2][XYZ2Lin(width - 1, height - 1, frames - 1,
                                             width, height)]),
              EPS);
}

TEST_F(Test_TGV, AdaptStepSize)
{
  unsigned int width = 5;
  unsigned int height = 5;
  unsigned int coils = 3;
  unsigned int frames = 2;

  BaseOperator *cartOp =
      new CartesianOperator(width, height, coils, frames, false);
  TGV2 tgvSolver(width, height, coils, frames, cartOp);

  PDParams &params = tgvSolver.GetParams();
  params.sigma = 1.2;

  unsigned N = width * height * frames;

  print("Input: ", width, height, frames, matrix_data);

  RVector mask(N);
  mask.assign(mask.size(), 1.0);

  // b1 map
  CVector b1(width * height * coils);
  b1.assign(width * height * coils, 1.0);
  agile::lowlevel::scale(CType(0.0, 2.0), b1.data() + width * height,
                         b1.data() + width * height, width * height);

  CVector ext1(N);
  ext1.assignFromHost(matrix_data.begin(), matrix_data.begin() + N);

  std::vector<CVector> ext2(3);

  ext2[0].assignFromHost(matrix_data.begin(), matrix_data.begin() + N);
  ext2[1].assignFromHost(matrix_data.begin(), matrix_data.begin() + N);
  ext2[2].assignFromHost(matrix_data.begin(), matrix_data.begin() + N);

  // adapt stepsize
  tgvSolver.AdaptStepSize(ext1, ext2, b1);

  EXPECT_NEAR(1.1400, tgvSolver.GetParams().sigma, EPS);
  EXPECT_NEAR(0.3164, tgvSolver.GetParams().tau, EPS);
  delete cartOp;
}

TEST_F(Test_TGV, ComputeTGV2Norm)
{
  unsigned int width = 5;
  unsigned int height = 5;
  unsigned int frames = 2;
  unsigned N = width * height * frames;
  RType ds = 1;
  RType dt = 1;
  float alpha0 = std::sqrt(2.0);
  float alpha1 = 1.0;

  print("Input: ", width, height, 4 * frames, matrix_data);

  // test data
  CVector data1_gpu;
  data1_gpu.assignFromHost(matrix_data.begin(), matrix_data.begin() + N);

  std::vector<CVector> data2_gpu(3);
  data2_gpu[0].assignFromHost(matrix_data.begin(), matrix_data.begin() + N);
  data2_gpu[1].assignFromHost(matrix_data.begin(), matrix_data.begin() + N);
  data2_gpu[2].assignFromHost(matrix_data.begin(), matrix_data.begin() + N);

  // TGV2 Norm tests
  RType norm = utils::TGV2Norm(data1_gpu, data2_gpu, alpha0, alpha1, width,
                               height, ds, ds, dt);
  EXPECT_NEAR(4467.988, norm, EPS);

  ds = 1.2;
  dt = 2;
  norm = utils::TGV2Norm(data1_gpu, data2_gpu, alpha0, alpha1, width, height,
                         ds, ds, dt);
  EXPECT_NEAR(3703.5632, norm, EPS);

  agile::addScaledVector(data1_gpu, CType(0, 2), data1_gpu, data1_gpu);
  norm = utils::TGV2Norm(data1_gpu, data2_gpu, alpha0, alpha1, width, height,
                         ds, ds, dt);
  EXPECT_NEAR(4085.057, norm, EPS);
}

TEST_F(Test_TGV, ComputeGStar)
{
  unsigned int width = 5;
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
  TGV2 solver(width, height, coils, frames, cartOp);
  solver.GetParams().lambda = 1.5;

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
  agile::GPUVector<CType> z(N * coils);
  z.assign(N * coils, 1);

  std::vector<CVector> y1, y2;
  for (unsigned cnt = 0; cnt < 3; cnt++)
  {
    y1.push_back(CVector(N));
    y1[cnt].assign(N, 0);
  }
  for (unsigned cnt = 0; cnt < 6; cnt++)
  {
    y2.push_back(CVector(N));
    y2[cnt].assign(N, 0);
  }

  RType gstar = solver.ComputeGStar(x, y1, y2, z, data_gpu, b1_gpu);
  EXPECT_NEAR(996634.1875, gstar, EPS);
  delete cartOp;
}

TEST_F(Test_TGV, PDGap)
{
  unsigned int width = 5;
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
  TGV2 solver(width, height, coils, frames, cartOp);
  solver.GetParams().lambda = 1.5;

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
  agile::GPUVector<CType> x1;
  x1.assignFromHost(matrix_data.begin(), matrix_data.begin() + N);

  std::vector<CVector> x2(3);
  x2[0].assignFromHost(matrix_data.begin(), matrix_data.begin() + N);
  x2[1].assignFromHost(matrix_data.begin(), matrix_data.begin() + N);
  x2[2].assignFromHost(matrix_data.begin(), matrix_data.begin() + N);

  // dual vars
  agile::GPUVector<CType> z(N * coils);
  z.assign(N * coils, 1);

  std::vector<CVector> y1, y2;
  for (unsigned cnt = 0; cnt < 3; cnt++)
  {
    y1.push_back(CVector(N));
    y1[cnt].assign(N, 0);
  }
  for (unsigned cnt = 0; cnt < 6; cnt++)
  {
    y2.push_back(CVector(N));
    y2[cnt].assign(N, 0);
  }

  RType PDGap = solver.ComputePDGap(x1, x2, y1, y2, z, data_gpu, b1_gpu);
  EXPECT_NEAR(1001102.187, PDGap, EPS);
  delete cartOp;
}

TEST_F(Test_TGV, Iteration)
{
  unsigned int width = 5;
  unsigned int height = 5;
  unsigned int coils = 2;
  unsigned int frames = 3;

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
  data_gpu.assignFromHost(matrix_data.begin(), matrix_data.begin() + N * coils);
  // b1 map
  CVector b1_gpu(width * height * coils);
  b1_gpu.assign(width * height * coils, 1.0);
  agile::lowlevel::scale(CType(0.0, 2.0), b1_gpu.data() + width * height,
                         b1_gpu.data() + width * height, width * height);

  CVector x1(N);
  x1.assign(x1.size(), 0.0);

  RVector mask_gpu(N);
  mask_gpu.assign(N, 1.0);

  BaseOperator *cartOp =
      new CartesianOperator(width, height, coils, frames, mask_gpu, false);
  TGV2 solver(width, height, coils, frames, cartOp);

  solver.IterativeReconstruction(data_gpu, x1, b1_gpu);

  // get result
  std::vector<CType> result(width * height * frames);
  x1.copyToHost(result);
  print("result:", width, height, frames, result, true);
  EXPECT_NEAR(75.052, std::abs(result[XYZ2Lin(0, 0, 0, width, height)]), EPS);
  EXPECT_NEAR(9.4832, std::abs(result[XYZ2Lin(0, 1, 0, width, height)]), EPS);
  EXPECT_NEAR(184.697, std::abs(result[XYZ2Lin(0, 0, 1, width, height)]), EPS);
  EXPECT_NEAR(9.481, std::abs(result[XYZ2Lin(0, 1, 1, width, height)]), EPS);
  EXPECT_NEAR(295.893, std::abs(result[XYZ2Lin(0, 0, 2, width, height)]), EPS);
  EXPECT_NEAR(9.475, std::abs(result[XYZ2Lin(0, 1, 2, width, height)]), EPS);
  delete cartOp;
}

TEST(Test_TGV_Real, DISABLED_Iteration)
{
  agile::GPUEnvironment::allocateGPU(0);
  agile::GPUEnvironment::printInformation(std::cout);
  unsigned int height = 416;
  unsigned int width = 168;
  unsigned int coils = 30;
  unsigned int frames = 25;

  unsigned int N = width * height * frames;

  const char *output =
      "../test/data/output/tgv2_test_vector_data_416_168_25.bin";

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

  RVector mask_gpu(N);
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
  TGV2 solver(width, height, coils, frames, cartOp);

  solver.IterativeReconstruction(data_gpu, x_gpu, b1_gpu);

  // get result
  std::vector<CType> result(N);
  x_gpu.copyToHost(result);
  print("result:", 10, 10, 1, result, false);
  agile::writeVectorFile(output, result);
  delete cartOp;
}

TEST(Test_TGV_NonCart_Real, DISABLED_Iteration)
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
      "../test/data/output/noncart/test_tgv_recon_384_384_14.bin";

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

  TGV2 tgv2Solver(width, height, coils, frames, nonCartOp);

  tgv2Solver.GetParams().ds = 1.3;
  tgv2Solver.GetParams().dt = 0.2;

  tgv2Solver.IterativeReconstruction(data_gpu, x_gpu, b1_gpu);

  // get result
  std::vector<CType> result(N);
  x_gpu.copyToHost(result);
  print("result:", 10, 10, 1, result, false);
  agile::writeVectorFile(output, result);
  delete nonCartOp;
}

