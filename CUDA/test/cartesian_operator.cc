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
#include "../include/cartesian_operator.h"

class Test_CartesianOperator : public ::testing::Test
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

std::vector<CType> Test_CartesianOperator::matrix_data;

TEST_F(Test_CartesianOperator, ForwardOperation)
{
  unsigned int width = 6;
  unsigned int height = 6;
  unsigned int coils = 2;
  unsigned int frames = 2;
  BaseOperator *cartOp = new CartesianOperator(width, height, coils, frames);

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

  agile::GPUVector<CType> x_gpu;
  x_gpu.assignFromHost(matrix_data.begin(), matrix_data.end());

  agile::GPUVector<CType> b1_gpu(width * height * coils);

  b1_gpu.assign(width * height * coils, 1.0);
  agile::lowlevel::scale(CType(0.0, 2.0), b1_gpu.data() + width * height,
                         b1_gpu.data() + width * height, width * height);

  CVector sum_img_gpu = cartOp->ForwardOperation(x_gpu, b1_gpu);

  std::vector<CType> z(N);
  sum_img_gpu.copyToHost(z);

  print("z: ", width, height, frames, z, true);
  EXPECT_NEAR(40.2492, std::abs(z[XYZ2Lin(3, 0, 0, width, height)]), EPS);
  EXPECT_NEAR(46.4758, std::abs(z[XYZ2Lin(3, 1, 0, width, height)]), EPS);
  EXPECT_NEAR(80.4984, std::abs(z[XYZ2Lin(3, 2, 0, width, height)]), EPS);
  EXPECT_NEAR(650.5298, std::abs(z[XYZ2Lin(3, 3, 0, width, height)]), EPS);
  EXPECT_NEAR(6.7082, std::abs(z[XYZ2Lin(0, 3, 0, width, height)]), EPS);
  EXPECT_NEAR(7.7460, std::abs(z[XYZ2Lin(1, 3, 0, width, height)]), EPS);
  EXPECT_NEAR(13.4164, std::abs(z[XYZ2Lin(2, 3, 0, width, height)]), EPS);

  EXPECT_NEAR(40.2492, std::abs(z[XYZ2Lin(3, 0, 1, width, height)]), EPS);
  EXPECT_NEAR(46.4758, std::abs(z[XYZ2Lin(3, 1, 1, width, height)]), EPS);
  EXPECT_NEAR(80.4984, std::abs(z[XYZ2Lin(3, 2, 1, width, height)]), EPS);
  EXPECT_NEAR(1598.876, std::abs(z[XYZ2Lin(3, 3, 1, width, height)]), EPS);
  EXPECT_NEAR(6.7082, std::abs(z[XYZ2Lin(0, 3, 1, width, height)]), EPS);
  EXPECT_NEAR(7.7460, std::abs(z[XYZ2Lin(1, 3, 1, width, height)]), EPS);
  EXPECT_NEAR(13.4164, std::abs(z[XYZ2Lin(2, 3, 1, width, height)]), EPS);
  delete cartOp;
}

TEST_F(Test_CartesianOperator, MaskedForwardOperation)
{
  unsigned int width = 6;
  unsigned int height = 6;
  unsigned int coils = 2;
  unsigned int frames = 2;

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

  agile::GPUVector<CType> x_gpu;
  x_gpu.assignFromHost(matrix_data.begin(), matrix_data.end());
  print("Input: ", width, height, coils, frames, matrix_data);

  std::vector<float> mask;
  for (unsigned frame = 0; frame < frames; frame++)
    for (unsigned row = 0; row < height; row++)
      for (unsigned column = 0; column < width; column++)
      {
        if (row % 2 == 0)
          mask.push_back(1.0f);
        else
          mask.push_back(0.0f);
      }
  print("Mask: ", width, height, frames, mask);
  agile::GPUVector<float> mask_gpu;
  mask_gpu.assignFromHost(mask.begin(), mask.end());

  BaseOperator *cartOp =
      new CartesianOperator(width, height, coils, frames, mask_gpu);

  agile::GPUVector<CType> b1_gpu(width * height * coils);
  b1_gpu.assign(width * height * coils, 1.0);
  agile::lowlevel::scale(CType(0.0, 2.0), b1_gpu.data() + width * height,
                         b1_gpu.data() + width * height, width * height);

  CVector sum_img_gpu(width * height * frames);
  cartOp->ForwardOperation(x_gpu, sum_img_gpu, b1_gpu);

  std::vector<CType> z(N);
  sum_img_gpu.copyToHost(z);

  print("z: ", width, height, frames, z, false);
  delete cartOp;
}

TEST_F(Test_CartesianOperator, Adjointness)
{
  unsigned int width = 168;
  unsigned int height = 416;
  unsigned int coils = 30;
  unsigned int frames = 1;
  BaseOperator *cartOp = new CartesianOperator(width, height, coils, frames);

  unsigned int N = width * height * frames;

  std::vector<CType> x;
  std::vector<CType> y;
  // init random number generator
  srand(time(NULL));
  for (unsigned cnt = 0; cnt < N; cnt++)
    y.push_back(randomValue());

  for (unsigned cnt = 0; cnt < N * coils; cnt++)
    x.push_back(randomValue());

  //  print("Input: ", width, height, coils, frames, matrix_data);

  // data
  CVector x_gpu(N * coils);
  CVector y_gpu(N);
  x_gpu.assignFromHost(x.begin(), x.end());
  y_gpu.assignFromHost(y.begin(), y.end());

  CVector Kx(N);
  CVector KHy(N * coils);

  // b1 map
  CVector b1_gpu(width * height * coils);
  b1_gpu.assign(width * height * coils, 1.0);
  agile::lowlevel::scale(CType(0.0, 2.0), b1_gpu.data() + width * height,
                         b1_gpu.data() + width * height, width * height);

  cartOp->BackwardOperation(y_gpu, KHy, b1_gpu);

  cartOp->ForwardOperation(x_gpu, Kx, b1_gpu);

  CType Kxy = agile::getScalarProduct(Kx, y_gpu);
  CType xKHy = agile::getScalarProduct(x_gpu, KHy);

  std::cout << Kxy << std::endl;
  std::cout << xKHy << std::endl;

  EXPECT_NEAR(0.0, std::real(Kxy - xKHy), 1E-2);
  EXPECT_NEAR(0.0, std::imag(Kxy - xKHy), 1E-2);
  delete cartOp;
}

TEST_F(Test_CartesianOperator, BackwardOperation)
{
  unsigned int width = 6;
  unsigned int height = 6;
  unsigned int coils = 3;
  unsigned int frames = 2;
  BaseOperator *cartOp = new CartesianOperator(width, height, coils, frames);

  unsigned int N = width * height * frames;

  for (unsigned frame = 0; frame < frames; frame++)
    for (unsigned row = 0; row < height; ++row)
      for (unsigned column = 0; column < width; ++column)
      {
        matrix_data.push_back(
            CType(XYZ2Lin(column, row, frame, width, height), 0));
      }

  print("Input: ", width, height, frames, matrix_data);

  CVector x_gpu;
  x_gpu.assignFromHost(matrix_data.begin(), matrix_data.end());

  CVector b1_gpu(width * height * coils);

  b1_gpu.assign(width * height * coils, 1.0f);
  agile::lowlevel::scale(2.0f, b1_gpu.data() + width * height,
                         b1_gpu.data() + width * height, width * height);
  agile::lowlevel::scale(3.0f, b1_gpu.data() + 2 * width * height,
                         b1_gpu.data() + 2 * width * height, width * height);

  CVector z_gpu = cartOp->BackwardOperation(x_gpu, b1_gpu);
  std::vector<CType> z(N * coils);
  z_gpu.copyToHost(z);

  print("z: ", width, height, frames, coils, z, true);
  EXPECT_NEAR(105, std::abs(z[XYZ2Lin(3, 3, 0, width, height)]), EPS);
  EXPECT_NEAR(2 * 105, std::abs(z[XYZ2Lin(3, 3, 1, width, height)]), EPS);
  EXPECT_NEAR(3 * 105, std::abs(z[XYZ2Lin(3, 3, 2, width, height)]), EPS);
  EXPECT_NEAR(36, std::abs(z[XYZ2Lin(3, 2, 0, width, height)]), EPS);
  EXPECT_NEAR(2 * 36, std::abs(z[XYZ2Lin(3, 2, 1, width, height)]), EPS);
  EXPECT_NEAR(3 * 36, std::abs(z[XYZ2Lin(3, 2, 2, width, height)]), EPS);
  EXPECT_NEAR(18, std::abs(z[XYZ2Lin(3, 0, 0, width, height)]), EPS);
  EXPECT_NEAR(2 * 18, std::abs(z[XYZ2Lin(3, 0, 1, width, height)]), EPS);
  EXPECT_NEAR(3 * 18, std::abs(z[XYZ2Lin(3, 0, 2, width, height)]), EPS);
  EXPECT_NEAR(3, std::abs(z[XYZ2Lin(0, 3, 0, width, height)]), EPS);
  EXPECT_NEAR(2 * 3, std::abs(z[XYZ2Lin(0, 3, 1, width, height)]), EPS);

  for (unsigned frame = 0; frame < frames; frame++)
  {
    for (unsigned ind = 0; ind < width * height; ind++)
    {
      unsigned offset = frame * width * height * coils;
      CType ref = z[ind + offset];

      for (unsigned coil = 1; coil < coils; coil++)
      {
        EXPECT_NEAR((float)(coil + 1.0) * ref.real(),
                    z[ind + offset + coil * width * height].real(), EPS);
        EXPECT_NEAR((float)(coil + 1.0) * ref.imag(),
                    z[ind + offset + coil * width * height].imag(), EPS);
      }
    }
  }
  delete cartOp;
}

TEST_F(Test_CartesianOperator, MaskedBackwardOperation)
{
  unsigned int width = 6;
  unsigned int height = 6;
  unsigned int coils = 3;
  unsigned int frames = 2;

  unsigned int N = width * height * frames;

  for (unsigned frame = 0; frame < frames; frame++)
    for (unsigned row = 0; row < height; ++row)
      for (unsigned column = 0; column < width; ++column)
      {
        matrix_data.push_back(
            CType(XYZ2Lin(column, row, frame, width, height), 0));
      }

  agile::GPUVector<CType> x_gpu;
  x_gpu.assignFromHost(matrix_data.begin(), matrix_data.end());
  print("Input: ", width, height, coils, frames, matrix_data);

  std::vector<float> mask;
  for (unsigned frame = 0; frame < frames; frame++)
    for (unsigned row = 0; row < height; row++)
      for (unsigned column = 0; column < width; column++)
      {
        if (row % 2 == 0)
          mask.push_back(1.0f);
        else
          mask.push_back(0.0f);
      }
  print("Mask: ", width, height, frames, mask);
  agile::GPUVector<float> mask_gpu;
  mask_gpu.assignFromHost(mask.begin(), mask.end());

  BaseOperator *cartOp =
      new CartesianOperator(width, height, coils, frames, mask_gpu);

  CVector b1_gpu(width * height * coils);

  b1_gpu.assign(width * height * coils, 1.0f);
  agile::lowlevel::scale(2.0f, b1_gpu.data() + width * height,
                         b1_gpu.data() + width * height, width * height);
  agile::lowlevel::scale(3.0f, b1_gpu.data() + 2 * width * height,
                         b1_gpu.data() + 2 * width * height, width * height);

  CVector z_gpu(width * height * frames * coils);
  cartOp->BackwardOperation(x_gpu, z_gpu, b1_gpu);

  std::vector<CType> z(N);
  z_gpu.copyToHost(z);

  print("z: ", width, height, coils, frames, z, true);
  delete cartOp;
}

