#include <gtest/gtest.h>

#include <stdlib.h>  //< srand, rand
#include <time.h>    //< time
#include <boost/lexical_cast.hpp>

#include "agile/gpu_environment.hpp"
#include "agile/gpu_vector.hpp"
#include "agile/gpu_matrix.hpp"
#include "agile/calc/fft.hpp"
#include "agile/io/file.hpp"

#include "./test_utils.h"
#include "../include/types.h"
#include "../include/ictgv2.h"
#include "../include/utils.h"
#include "../include/cartesian_operator.h"
#include "../include/noncartesian_operator.h"

class Test_ICTGV : public ::testing::Test
{
 public:
  static void SetUpTestCase()
  {
    unsigned int max_cols = 10;
    unsigned int max_rows = 10;
    unsigned int max_slices = 8;

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

std::vector<CType> Test_ICTGV::matrix_data;

TEST_F(Test_ICTGV, GradientNorm)
{
  unsigned int width = 5;
  unsigned int height = 5;
  unsigned int frames = 3;
  unsigned int N = width * height * frames;

  print("Input: ", width, height, 4 * frames, matrix_data);

  CVector data1_gpu, data3_gpu, temp(N);
  std::vector<CVector> data2_gpu(3), data4_gpu(3);

  data1_gpu.assignFromHost(matrix_data.begin(), matrix_data.begin() + N);
  for (unsigned cnt = 0; cnt < 3; cnt++)
  {
    data2_gpu[cnt].assignFromHost(matrix_data.begin() + (cnt + 1) * N,
                                  matrix_data.begin() + (cnt + 2) * N);
  }

  data3_gpu.assignFromHost(matrix_data.begin() + 4 * N,
                           matrix_data.begin() + 5 * N);
  for (unsigned cnt = 0; cnt < 3; cnt++)
  {
    data4_gpu[cnt].assignFromHost(matrix_data.begin() + (cnt + 1 + 4) * N,
                                  matrix_data.begin() + (cnt + 2 + 4) * N);
  }

  agile::subVector(data1_gpu, data3_gpu, temp);

  std::vector<CVector> y1 = utils::Gradient(temp, width, height);

  agile::subVector(y1[0], data2_gpu[0], y1[0]);
  agile::subVector(y1[1], data2_gpu[1], y1[1]);
  agile::subVector(y1[2], data2_gpu[2], y1[2]);

  std::vector<CVector> y2 = utils::SymmetricGradient(data2_gpu, width, height);

  CVector norm_gpu;
  norm_gpu = utils::GradientNorm(y1);

  std::vector<CType> norm(N);
  norm_gpu.copyToHost(norm);
  print("gradient norm: ", width, height, frames, norm);

  EXPECT_NEAR(288.660, std::abs(norm[XYZ2Lin(0, 1, 0, width, height)]), EPS);
  EXPECT_NEAR(299.970, std::abs(norm[XYZ2Lin(2, 2, 0, width, height)]), EPS);
  EXPECT_NEAR(319.496, std::abs(norm[XYZ2Lin(4, 4, 0, width, height)]), EPS);

  EXPECT_NEAR(329.317, std::abs(norm[XYZ2Lin(0, 1, 1, width, height)]), EPS);
  EXPECT_NEAR(340.818, std::abs(norm[XYZ2Lin(2, 2, 1, width, height)]), EPS);
  EXPECT_NEAR(360.628, std::abs(norm[XYZ2Lin(4, 4, 1, width, height)]), EPS);

  EXPECT_NEAR(370.573, std::abs(norm[XYZ2Lin(0, 1, 2, width, height)]), EPS);
  EXPECT_NEAR(382.206, std::abs(norm[XYZ2Lin(2, 2, 2, width, height)]), EPS);
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

  std::vector<CVector> y3 = utils::Gradient(data3_gpu, width, height);

  agile::subVector(y3[0], data4_gpu[0], y3[0]);
  agile::subVector(y3[1], data4_gpu[1], y3[1]);
  agile::subVector(y3[2], data4_gpu[2], y3[2]);

  std::vector<CVector> y4 = utils::SymmetricGradient(data4_gpu, width, height);

  norm_gpu = utils::GradientNorm(y3);
  norm_gpu.copyToHost(norm);
  EXPECT_NEAR(775.349, std::abs(norm[XYZ2Lin(0, 1, 0, width, height)]), EPS);
  EXPECT_NEAR(830.430, std::abs(norm[XYZ2Lin(2, 2, 1, width, height)]), EPS);
  EXPECT_NEAR(913.771, std::abs(norm[XYZ2Lin(4, 4, 2, width, height)]), EPS);

  norm_gpu = utils::SymmetricGradientNorm(y4);
  norm_gpu.copyToHost(norm);
  EXPECT_NEAR(1025.17, std::abs(norm[XYZ2Lin(0, 1, 0, width, height)]), EPS);
  EXPECT_NEAR(38.17, std::abs(norm[XYZ2Lin(2, 2, 1, width, height)]), EPS);
  EXPECT_NEAR(1544.879, std::abs(norm[XYZ2Lin(4, 4, 2, width, height)]), EPS);
}

TEST_F(Test_ICTGV, ScaledGradientNorm)
{
  unsigned int width = 5;
  unsigned int height = 5;
  unsigned int frames = 3;
  DType ds = 1.5;
  DType dt = 0.5;
  DType ds2 = 1.2;
  DType dt2 = 0.75;
  unsigned int N = width * height * frames;

  print("Input: ", width, height, 4 * frames, matrix_data);

  CVector data1_gpu, data3_gpu, temp(N);
  std::vector<CVector> data2_gpu(3), data4_gpu(3);

  data1_gpu.assignFromHost(matrix_data.begin(), matrix_data.begin() + N);
  for (unsigned cnt = 0; cnt < 3; cnt++)
  {
    data2_gpu[cnt].assignFromHost(matrix_data.begin() + (cnt + 1) * N,
                                  matrix_data.begin() + (cnt + 2) * N);
  }

  data3_gpu.assignFromHost(matrix_data.begin() + 4 * N,
                           matrix_data.begin() + 5 * N);
  for (unsigned cnt = 0; cnt < 3; cnt++)
  {
    data4_gpu[cnt].assignFromHost(matrix_data.begin() + (cnt + 1 + 4) * N,
                                  matrix_data.begin() + (cnt + 2 + 4) * N);
  }

  agile::subVector(data1_gpu, data3_gpu, temp);

  std::vector<CVector> y1 = utils::Gradient(temp, width, height, ds, ds, dt);

  agile::subVector(y1[0], data2_gpu[0], y1[0]);
  agile::subVector(y1[1], data2_gpu[1], y1[1]);
  agile::subVector(y1[2], data2_gpu[2], y1[2]);

  std::vector<CVector> y2 =
      utils::SymmetricGradient(data2_gpu, width, height, ds, ds, dt);

  CVector norm_gpu;
  norm_gpu = utils::GradientNorm(y1);

  std::vector<CType> norm(N);
  norm_gpu.copyToHost(norm);
  print("gradient norm: ", width, height, frames, norm);

  EXPECT_NEAR(288.660, std::abs(norm[XYZ2Lin(0, 1, 0, width, height)]), EPS);
  EXPECT_NEAR(299.970, std::abs(norm[XYZ2Lin(2, 2, 0, width, height)]), EPS);
  EXPECT_NEAR(319.496, std::abs(norm[XYZ2Lin(4, 4, 0, width, height)]), EPS);

  EXPECT_NEAR(329.317, std::abs(norm[XYZ2Lin(0, 1, 1, width, height)]), EPS);
  EXPECT_NEAR(340.818, std::abs(norm[XYZ2Lin(2, 2, 1, width, height)]), EPS);
  EXPECT_NEAR(360.628, std::abs(norm[XYZ2Lin(4, 4, 1, width, height)]), EPS);

  EXPECT_NEAR(370.573, std::abs(norm[XYZ2Lin(0, 1, 2, width, height)]), EPS);
  EXPECT_NEAR(382.206, std::abs(norm[XYZ2Lin(2, 2, 2, width, height)]), EPS);
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

  std::vector<CVector> y3 =
      utils::Gradient(data3_gpu, width, height, ds2, ds2, dt2);

  agile::subVector(y3[0], data4_gpu[0], y3[0]);
  agile::subVector(y3[1], data4_gpu[1], y3[1]);
  agile::subVector(y3[2], data4_gpu[2], y3[2]);

  std::vector<CVector> y4 =
      utils::SymmetricGradient(data4_gpu, width, height, ds2, ds2, dt2);

  norm_gpu = utils::GradientNorm(y3);
  norm_gpu.copyToHost(norm);
  EXPECT_NEAR(770.516, std::abs(norm[XYZ2Lin(0, 1, 0, width, height)]), EPS);
  EXPECT_NEAR(825.635, std::abs(norm[XYZ2Lin(2, 2, 1, width, height)]), EPS);
  EXPECT_NEAR(913.771, std::abs(norm[XYZ2Lin(4, 4, 2, width, height)]), EPS);

  norm_gpu = utils::SymmetricGradientNorm(y4);
  norm_gpu.copyToHost(norm);
  EXPECT_NEAR(1144.277, std::abs(norm[XYZ2Lin(0, 1, 0, width, height)]), EPS);
  EXPECT_NEAR(49.279, std::abs(norm[XYZ2Lin(2, 2, 1, width, height)]), EPS);
  EXPECT_NEAR(1573.446, std::abs(norm[XYZ2Lin(4, 4, 2, width, height)]), EPS);
}

TEST_F(Test_ICTGV, Divergence)
{
  unsigned int width = 5;
  unsigned int height = 5;
  unsigned int frames = 3;
  float alpha = 0.5;
  float alpha0 = 1.0;
  float alpha1 = std::sqrt(2.0);
  DType ds = 1.5;
  DType dt = 0.5;
  DType ds2 = 1.2;
  DType dt2 = 0.75;
  unsigned int N = width * height * frames;

  print("Input: ", width, height, 4 * frames, matrix_data);

  CVector data1_gpu, data3_gpu, temp(N);
  std::vector<CVector> data2_gpu(3), data4_gpu(3);

  data1_gpu.assignFromHost(matrix_data.begin(), matrix_data.begin() + N);
  for (unsigned cnt = 0; cnt < 3; cnt++)
  {
    data2_gpu[cnt].assignFromHost(matrix_data.begin() + (cnt + 1) * N,
                                  matrix_data.begin() + (cnt + 2) * N);
  }

  data3_gpu.assignFromHost(matrix_data.begin() + 4 * N,
                           matrix_data.begin() + 5 * N);
  for (unsigned cnt = 0; cnt < 3; cnt++)
  {
    data4_gpu[cnt].assignFromHost(matrix_data.begin() + (cnt + 1 + 4) * N,
                                  matrix_data.begin() + (cnt + 2 + 4) * N);
  }

  // compute gradients
  agile::subVector(data1_gpu, data3_gpu, temp);
  std::vector<CVector> y1 = utils::Gradient(temp, width, height, ds, ds, dt);

  agile::subVector(y1[0], data2_gpu[0], y1[0]);
  agile::subVector(y1[1], data2_gpu[1], y1[1]);
  agile::subVector(y1[2], data2_gpu[2], y1[2]);

  std::vector<CVector> y2 =
      utils::SymmetricGradient(data2_gpu, width, height, ds, ds, dt);

  std::vector<CVector> y3 =
      utils::Gradient(data3_gpu, width, height, ds2, ds2, dt2);

  agile::subVector(y3[0], data4_gpu[0], y3[0]);
  agile::subVector(y3[1], data4_gpu[1], y3[1]);
  agile::subVector(y3[2], data4_gpu[2], y3[2]);

  std::vector<CVector> y4 =
      utils::SymmetricGradient(data4_gpu, width, height, ds2, ds2, dt2);

  // prox operator y1
  RType denom = alpha1 * (alpha / std::min(alpha, (RType)1.0 - alpha));
  utils::ProximalMap3(y1, 1.0 / denom);

  // prox operator y2
  denom = alpha0 * (alpha / std::min(alpha, (RType)1.0 - alpha));
  utils::ProximalMap6(y2, 1.0 / denom);

  // prox operator y3
  denom = alpha1 * ((1.0 - alpha) / std::min(alpha, (RType)1.0 - alpha));
  utils::ProximalMap3(y3, 1.0 / denom);

  // prox operator y4
  denom = alpha0 * ((1.0 - alpha) / std::min(alpha, (RType)1.0 - alpha));
  utils::ProximalMap6(y4, 1.0 / denom);

  // Divergences of y3
  CVector div1 = utils::Divergence(y1, width, height, frames, ds, ds, dt);
  std::vector<CType> div1Host;
  div1.copyToHost(div1Host);
  print("div_3_3(y1)", width, height, frames, div1Host, false);

  EXPECT_NEAR(-3.0237, std::real(div1Host[XYZ2Lin(0, 0, 0, width, height)]),
              EPS);
  EXPECT_NEAR(0.0577, std::real(div1Host[XYZ2Lin(2, 2, 1, width, height)]),
              EPS);
  EXPECT_NEAR(3.0216, std::real(div1Host[XYZ2Lin(4, 4, 2, width, height)]),
              EPS);

  // Divergence of y2
  std::vector<CVector> symDiv2 =
      utils::SymmetricDivergence(y2, width, height, frames, ds, ds, dt);
  std::vector<CType> symDiv2Host[3];
  for (int cnt = 0; cnt < 3; cnt++)
  {
    symDiv2[cnt].copyToHost(symDiv2Host[cnt]);
    print("div_3_6(y2)", width, height, frames, symDiv2Host[cnt], false);
  }

  EXPECT_NEAR(0.1819,
              std::real(symDiv2Host[0][XYZ2Lin(0, 0, 0, width, height)]), EPS);
  EXPECT_NEAR(-1.3486,
              std::real(symDiv2Host[1][XYZ2Lin(2, 2, 1, width, height)]), EPS);
  EXPECT_NEAR(0.0, std::real(symDiv2Host[2][XYZ2Lin(4, 4, 2, width, height)]),
              EPS);

  // Divergence of y3
  CVector div3 = utils::Divergence(y3, width, height, frames, ds2, ds2, dt2);
  std::vector<CType> div3Host;
  div3.copyToHost(div3Host);
  print("div_3_3(y3)", width, height, frames, div3Host, false);

  EXPECT_NEAR(-2.4852, std::real(div3Host[XYZ2Lin(0, 0, 0, width, height)]),
              EPS);
  EXPECT_NEAR(0.0061, std::real(div3Host[XYZ2Lin(2, 2, 1, width, height)]),
              EPS);
  EXPECT_NEAR(2.4507, std::real(div3Host[XYZ2Lin(4, 4, 2, width, height)]),
              EPS);

  // Divergence of y4
  std::vector<CVector> symDiv4 =
      utils::SymmetricDivergence(y4, width, height, frames, ds2, ds2, dt2);

  std::vector<CType> symDiv4Host[3];
  for (int cnt = 0; cnt < 3; cnt++)
  {
    symDiv4[cnt].copyToHost(symDiv4Host[cnt]);
    print("div_3_6(y4)", width, height, frames, symDiv4Host[cnt], false);
  }
  EXPECT_NEAR(-0.3287,
              std::real(symDiv4Host[0][XYZ2Lin(0, 0, 0, width, height)]), EPS);
  EXPECT_NEAR(-0.9557,
              std::real(symDiv4Host[1][XYZ2Lin(2, 2, 1, width, height)]), EPS);
  EXPECT_NEAR(0.0, std::real(symDiv4Host[2][XYZ2Lin(4, 4, 2, width, height)]),
              EPS);
}

TEST_F(Test_ICTGV, AdaptStepSize)
{
  unsigned int width = 5;
  unsigned int height = 5;
  unsigned int coils = 3;
  unsigned int frames = 2;

  BaseOperator *cartOp =
      new CartesianOperator(width, height, coils, frames, false);
  ICTGV2 solver(width, height, coils, frames, cartOp);

  ICTGV2Params &params = static_cast<ICTGV2Params &>(solver.GetParams());
  params.sigma = 1.2;
  params.ds = 1.5;
  params.dt = 0.5;
  params.ds2 = 0.2;
  params.dt2 = 0.75;

  unsigned N = width * height * frames;

  print("Input: ", width, height, frames, matrix_data);

  RVector mask(N);
  mask.assign(mask.size(), 1.0);

  // b1 map
  CVector b1(width * height * coils);
  b1.assign(width * height * coils, 1.0);
  agile::lowlevel::scale(CType(0.0, 2.0), b1.data() + width * height,
                         b1.data() + width * height, width * height);

  CVector data1_gpu(N), data3_gpu(N);
  data1_gpu.assignFromHost(matrix_data.begin(), matrix_data.begin() + N);
  data3_gpu.assignFromHost(matrix_data.begin() + N,
                           matrix_data.begin() + 2 * N);

  std::vector<CVector> data2_gpu(3), data4_gpu(3);
  for (unsigned cnt = 0; cnt < 3; cnt++)
  {
    data2_gpu[cnt].assignFromHost(matrix_data.begin(), matrix_data.begin() + N);
    data4_gpu[cnt].assignFromHost(matrix_data.begin() + N,
                                  matrix_data.begin() + 2 * N);
  }

  // adapt stepsize
  solver.AdaptStepSize(data1_gpu, data2_gpu, data3_gpu, data4_gpu, b1);

  EXPECT_NEAR(0.3184, solver.GetParams().sigma, EPS);
  EXPECT_NEAR(0.3184, solver.GetParams().tau, EPS);
  delete cartOp;
}

TEST_F(Test_ICTGV, ComputeICTGV2Norm)
{
  unsigned int width = 5;
  unsigned int height = 5;
  unsigned int frames = 2;
  unsigned N = width * height * frames;

  RType alpha0 = std::sqrt(2.0);
  RType alpha1 = 1.0;
  RType alpha = 0.3;
  RType ds = 1.5;
  RType dt = 0.5;
  RType ds2 = 0.2;
  RType dt2 = 0.75;

  print("Input: ", width, height, 4 * frames, matrix_data);

  // test data
  CVector data1_gpu(N), data3_gpu(N);
  data1_gpu.assignFromHost(matrix_data.begin(), matrix_data.begin() + N);
  data3_gpu.assignFromHost(matrix_data.begin() + N,
                           matrix_data.begin() + 2 * N);
  std::vector<CVector> data2_gpu(3), data4_gpu(3);
  for (unsigned cnt = 0; cnt < 3; cnt++)
  {
    data2_gpu[cnt].assignFromHost(matrix_data.begin(), matrix_data.begin() + N);
    data4_gpu[cnt].assignFromHost(matrix_data.begin() + N,
                                  matrix_data.begin() + 2 * N);
  }

  // ICTGV2 Norm tests
  RType norm =
      utils::ICTGV2Norm(data1_gpu, data2_gpu, data3_gpu, data4_gpu, alpha0,
                        alpha1, alpha, width, height, ds, dt, ds2, dt2);
  EXPECT_NEAR(86183.710, norm, EPS);

  ds = 1.2;
  dt = 2;
  norm = utils::ICTGV2Norm(data1_gpu, data2_gpu, data3_gpu, data4_gpu, alpha0,
                           alpha1, alpha, width, height, ds, dt, ds2, dt2);
  EXPECT_NEAR(85009.867, norm, EPS);

  agile::addScaledVector(data1_gpu, CType(0, 2), data1_gpu, data1_gpu);
  norm = utils::ICTGV2Norm(data1_gpu, data2_gpu, data3_gpu, data4_gpu, alpha0,
                           alpha1, alpha, width, height, ds, dt, ds2, dt2);
  EXPECT_NEAR(85376.156, norm, EPS);
}

TEST_F(Test_ICTGV, ComputeGStar)
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
  ICTGV2 solver(width, height, coils, frames, cartOp);
  ICTGV2Params &params = static_cast<ICTGV2Params &>(solver.GetParams());
  params.lambda = 1.5;
  params.sigma = 1.2;
  params.ds = 1.5;
  params.dt = 0.5;
  params.ds2 = 0.2;
  params.dt2 = 0.75;

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

  // dual vars
  agile::GPUVector<CType> z(N * coils);
  z.assign(N * coils, 1);

  std::vector<CVector> y1, y2, y3, y4;
  for (unsigned cnt = 0; cnt < 3; cnt++)
  {
    y1.push_back(CVector(N));
    y1[cnt].assign(N, 0);
    y3.push_back(CVector(N));
    y3[cnt].assign(N, 0);
  }
  for (unsigned cnt = 0; cnt < 6; cnt++)
  {
    y2.push_back(CVector(N));
    y2[cnt].assign(N, 0);
    y4.push_back(CVector(N));
    y4[cnt].assign(N, 0);
  }

  RType gstar = solver.ComputeGStar(x1, y1, y2, y3, y4, z, data_gpu, b1_gpu);
  EXPECT_NEAR(996634.1875, gstar, EPS);
  delete cartOp;
}

TEST_F(Test_ICTGV, PDGap)
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
  ICTGV2 solver(width, height, coils, frames, cartOp);
  ICTGV2Params &params = static_cast<ICTGV2Params &>(solver.GetParams());
  params.lambda = 1.5;
  params.sigma = 1.2;
  params.ds = 1.5;
  params.dt = 0.5;
  params.ds2 = 0.2;
  params.dt2 = 0.75;
  params.alpha0 = std::sqrt(2.0);
  params.alpha1 = 1.0;
  params.alpha = 0.3;

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
  CVector x1, x3;
  x1.assignFromHost(matrix_data.begin(), matrix_data.begin() + N);
  x3.assignFromHost(matrix_data.begin() + N, matrix_data.begin() + 2 * N);
  std::vector<CVector> x2(3), x4(3);
  for (unsigned cnt = 0; cnt < 3; cnt++)
  {
    x2[cnt].assignFromHost(matrix_data.begin(), matrix_data.begin() + N);
    x4[cnt].assignFromHost(matrix_data.begin() + N,
                           matrix_data.begin() + 2 * N);
  }

  // dual vars
  agile::GPUVector<CType> z(N * coils);
  z.assign(N * coils, 1);

  std::vector<CVector> y1, y2, y3, y4;
  for (unsigned cnt = 0; cnt < 3; cnt++)
  {
    y1.push_back(CVector(N));
    y1[cnt].assign(N, 0);
    y3.push_back(CVector(N));
    y3[cnt].assign(N, 0);
  }
  for (unsigned cnt = 0; cnt < 6; cnt++)
  {
    y2.push_back(CVector(N));
    y2[cnt].assign(N, 0);
    y4.push_back(CVector(N));
    y4[cnt].assign(N, 0);
  }

  RType PDGap =
      solver.ComputePDGap(x1, x2, x3, x4, y1, y2, y3, y4, z, data_gpu, b1_gpu);
  EXPECT_NEAR(1082817.875, PDGap, EPS);
  delete cartOp;
}

TEST_F(Test_ICTGV, Iteration)
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
  ICTGV2 solver(width, height, coils, frames, cartOp);
  solver.IterativeReconstruction(data_gpu, x1, b1_gpu);

  // get result
  std::vector<CType> result(width * height * frames);
  x1.copyToHost(result);
  print("result:", width, height, frames, result, true);
  EXPECT_NEAR(75.015, std::abs(result[XYZ2Lin(0, 0, 0, width, height)]), EPS);
  EXPECT_NEAR(9.486, std::abs(result[XYZ2Lin(0, 1, 0, width, height)]), EPS);
  EXPECT_NEAR(184.705, std::abs(result[XYZ2Lin(0, 0, 1, width, height)]), EPS);
  EXPECT_NEAR(9.487, std::abs(result[XYZ2Lin(0, 1, 1, width, height)]), EPS);
  EXPECT_NEAR(295.917, std::abs(result[XYZ2Lin(0, 0, 2, width, height)]), EPS);
  EXPECT_NEAR(9.487, std::abs(result[XYZ2Lin(0, 1, 2, width, height)]), EPS);
  delete cartOp;
}

TEST(Test_ICTGV_Real, DISABLED_Iteration)
{
  agile::GPUEnvironment::allocateGPU(0);
  agile::GPUEnvironment::printInformation(std::cout);
  unsigned int height = 416;
  unsigned int width = 168;
  unsigned int coils = 30;
  unsigned int frames = 25;

  unsigned int N = width * height * frames;

  const char *output =
      "../test/data/output/ictgv2_test_vector_data_416_168_25.bin";

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
  ICTGV2 solver(width, height, coils, frames, cartOp);

  solver.IterativeReconstruction(data_gpu, x_gpu, b1_gpu);

  // get result
  std::vector<CType> result(N);
  x_gpu.copyToHost(result);
  print("result:", 10, 10, 1, result, false);
  agile::writeVectorFile(output, result);
  delete cartOp;
}

TEST(Test_ICTGV_NonCart_Real, DISABLED_Iteration)
{
  agile::GPUEnvironment::allocateGPU(0);
  agile::GPUEnvironment::printInformation(std::cout);
  agile::GPUEnvironment::printUsage(std::cout);

  unsigned int height = 384;
  unsigned int width = 384;
  unsigned int coils = 12;
  unsigned int frames = 42;

  unsigned int nFE = 384;
  unsigned int nSpokesPerFrame = 14;

  unsigned int N = width * height * frames;

  std::string output = "../test/data/output/noncart/test_ictgv_recon_384_384_" +
                       boost::lexical_cast<std::string>(frames) + "_osf20.bin";

  // data
  std::vector<CType> data;
  agile::readVectorFile(
      std::string("../test/data/noncart/test_data_384_" +
                  boost::lexical_cast<std::string>(nSpokesPerFrame) + "_" +
                  boost::lexical_cast<std::string>(coils) + "_" +
                  boost::lexical_cast<std::string>(frames) + ".bin").c_str(),
      data);

  print("Data: ", 20, 1, 1, 1, data, false);

  CVector data_gpu(nFE * nSpokesPerFrame * frames * coils);
  data_gpu.assignFromHost(data.begin(), data.end());

  // b1 map
  std::vector<CType> b1;
  agile::readVectorFile(std::string("../test/data/noncart/test_b1_384_384_" +
                                    boost::lexical_cast<std::string>(coils) +
                                    ".bin").c_str(),
                        b1);
  print("B1: ", 20, 1, 1, b1, false);

  CVector b1_gpu(width * height * coils);
  b1_gpu.assignFromHost(b1.begin(), b1.end());

  // kspace mask
  std::vector<RType> traj;
  agile::readVectorFile(
      std::string("../test/data/noncart/test_k_384_" +
                  boost::lexical_cast<std::string>(nSpokesPerFrame) + "_" +
                  boost::lexical_cast<std::string>(frames) + ".bin").c_str(),
      traj);
  print("Traj: ", 20, 1, 1, traj);

  unsigned int nTraj = frames * nSpokesPerFrame * nFE;
  RVector ktraj(2 * nTraj);
  ktraj.assignFromHost(traj.begin(), traj.end());

  // density data
  std::vector<RType> wHost;
  agile::readVectorFile(
      std::string("../test/data/noncart/test_w_384_" +
                  boost::lexical_cast<std::string>(nSpokesPerFrame) + "_" +
                  boost::lexical_cast<std::string>(frames) + ".bin").c_str(),
      wHost);
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

  ICTGV2 ictgv2Solver(width, height, coils, frames, nonCartOp);
  ICTGV2Params &params = static_cast<ICTGV2Params &>(ictgv2Solver.GetParams());

  params.ds = 1.3;
  params.dt = 0.2;

  params.ds2 = 1.0;
  params.dt2 = 0.4;

  ictgv2Solver.SetVerbose(true);
  ictgv2Solver.IterativeReconstruction(data_gpu, x_gpu, b1_gpu);

  // get result
  std::vector<CType> result(N);
  x_gpu.copyToHost(result);
  print("result:", 10, 10, 1, result, false);
  std::cout << "writing output to: " << output << std::endl;
  agile::writeVectorFile(output.c_str(), result);
  delete nonCartOp;
}

