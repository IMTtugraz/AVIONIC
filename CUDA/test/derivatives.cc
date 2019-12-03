#include <gtest/gtest.h>

#include "agile/gpu_environment.hpp"
#include "agile/gpu_vector.hpp"
#include "agile/gpu_matrix.hpp"
#include "agile/gpu_vector_base.hpp"

#include "./test_utils.h"

// Test fixtures
// Reuse test resources
//
typedef struct ParamSet
{
  int cols;
  int rows;
  int slices;
  bool borderWrap;

  ParamSet(int cols, int rows, int slices, bool borderWrap)
    : cols(cols), rows(rows), slices(slices), borderWrap(borderWrap){};
} ParamSet;

class Derivative_3D : public ::testing::Test,
                      public ::testing::WithParamInterface<ParamSet>
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
              XYZ2Lin(column, row, slice, max_cols, max_rows));
  }
  virtual void SetUp()
  {
    agile::GPUEnvironment::allocateGPU(0);
  }
  static std::vector<float> matrix_data;
  static const int slices = 3;
};

class Derivative_2D : public ::testing::Test
{
 protected:
  static void SetUpTestCase()
  {
    // row major
    for (unsigned row = 0; row < rows; ++row)
      for (unsigned column = 0; column < cols; ++column)
        matrix_data.push_back(XY2Lin(column, row, cols));
  }
  virtual void SetUp()
  {
    agile::GPUEnvironment::allocateGPU(0);
  }
  static std::vector<float> matrix_data;
  static const unsigned int cols = 5;
  static const unsigned int rows = 5;
};

// Static definition
std::vector<float> Derivative_3D::matrix_data;
std::vector<float> Derivative_2D::matrix_data;

// Initialize Test sets for 3D derivatives
INSTANTIATE_TEST_CASE_P(WrapBorderSet, Derivative_3D,
                        ::testing::Values(ParamSet(5, 5, 3, false),
                                          ParamSet(5, 5, 3, true)));

INSTANTIATE_TEST_CASE_P(WrapBorderSet2, Derivative_3D,
                        ::testing::Values(ParamSet(4, 4, 3, false),
                                          ParamSet(4, 4, 3, true)));

INSTANTIATE_TEST_CASE_P(WrapBorderSetAniso, Derivative_3D,
                        ::testing::Values(ParamSet(7, 5, 3, false),
                                          ParamSet(7, 5, 3, true)));

INSTANTIATE_TEST_CASE_P(WrapBorderSetAniso2, Derivative_3D,
                        ::testing::Values(ParamSet(5, 7, 3, false),
                                          ParamSet(5, 7, 3, true)));

TEST_F(Derivative_2D, Nabla_x)
{
  print("A: ", cols, rows, matrix_data);

  agile::GPUMatrix<float> data_gpu(rows, cols, &matrix_data[0]);
  agile::GPUMatrix<float> dx_gpu(rows, cols, NULL);

  std::vector<float> dx(matrix_data.size());

  agile::diff(1, data_gpu, dx_gpu);

  dx_gpu.copyToHost(dx);

  EXPECT_EQ(dx[XY2Lin(0, 0, cols)], 1);
  EXPECT_EQ(dx[XY2Lin(1, 0, cols)], 1);
  EXPECT_EQ(dx[XY2Lin(cols - 1, 0, cols)], -(int)rows + 1);

  print("dx: ", cols, rows, dx);
}

TEST_F(Derivative_2D, Nabla_transpose_x)
{
  print("A: ", cols, rows, matrix_data);

  agile::GPUMatrix<float> data_gpu(rows, cols, &matrix_data[0]);
  agile::GPUMatrix<float> dx_gpu(rows, cols, NULL);

  std::vector<float> dx(matrix_data.size());

  agile::difftrans(1, data_gpu, dx_gpu);

  dx_gpu.copyToHost(dx);

  print("dx: ", cols, rows, dx);

  EXPECT_EQ(dx[XY2Lin(0, 0, cols)], rows - 1);
  EXPECT_EQ(dx[XY2Lin(1, 0, cols)], -1);
  EXPECT_EQ(dx[XY2Lin(cols - 1, 0, cols)], -1);
}

TEST_F(Derivative_2D, Nabla_y)
{
  print("A: ", cols, rows, matrix_data);

  agile::GPUMatrix<float> data_gpu(rows, cols, &matrix_data[0]);
  agile::GPUMatrix<float> dy_gpu(rows, cols, NULL);

  std::vector<float> dy(matrix_data.size());

  agile::diff(2, data_gpu, dy_gpu);

  dy_gpu.copyToHost(dy);

  EXPECT_EQ(dy[XY2Lin(0, 0, cols)], 5);
  EXPECT_EQ(dy[XY2Lin(1, 0, cols)], 5);
  EXPECT_EQ(dy[XY2Lin(1, rows - 1, cols)], -1 * (int)((rows - 1) * cols));

  print("dy: ", cols, rows, dy);
}

TEST_F(Derivative_2D, Nabla_transpose_y)
{
  print("A: ", cols, rows, matrix_data);

  agile::GPUMatrix<float> data_gpu(rows, cols, &matrix_data[0]);
  agile::GPUMatrix<float> dy_gpu(rows, cols, NULL);

  std::vector<float> dy(matrix_data.size());

  agile::difftrans(2, data_gpu, dy_gpu);

  dy_gpu.copyToHost(dy);

  EXPECT_EQ(dy[XY2Lin(0, 0, cols)], (rows - 1) * cols);
  EXPECT_EQ(dy[XY2Lin(1, 0, cols)], (rows - 1) * cols);
  EXPECT_EQ(dy[XY2Lin(1, rows - 1, cols)], -(int)cols);

  print("dy: ", cols, rows, dy);
}

TEST_P(Derivative_3D, Nabla_x)
{
  int cols = GetParam().cols;
  int rows = GetParam().rows;
  bool borderWrap = GetParam().borderWrap;

  print("A: ", cols, rows, slices, matrix_data);

  agile::GPUVector<float> data_gpu;
  data_gpu.assignFromHost(matrix_data.begin(), matrix_data.end());
  agile::GPUVector<float> dx_gpu(rows * cols * slices);
  dx_gpu.assign(rows * cols * slices, 0.0f);

  std::vector<float> dx(cols * rows * slices);

  agile::lowlevel::diff3(1, cols, rows, data_gpu.data(), dx_gpu.data(),
                         rows * cols * slices, borderWrap);

  dx_gpu.copyToHost(dx);

  print("dx: ", cols, rows, slices, dx);
  EXPECT_EQ(dx[XYZ2Lin(0, 0, 0, cols, rows)], 1);
  EXPECT_EQ(dx[XYZ2Lin(2, 0, 1, cols, rows)], 1);

  // true if border is wrapped
  if (borderWrap)
  {
    EXPECT_EQ(dx[XYZ2Lin(cols - 1, 0, 0, cols, rows)], -(int)cols + 1);
    EXPECT_EQ(dx[XYZ2Lin(cols - 1, 0, slices - 1, cols, rows)], -(int)cols + 1);
  }
  else
  {
    EXPECT_EQ(dx[XYZ2Lin(cols - 1, 0, 0, cols, rows)], 0);
    EXPECT_EQ(dx[XYZ2Lin(cols - 1, 0, slices - 1, cols, rows)], 0);
  }
}

TEST_P(Derivative_3D, Nabla_transpose_x)
{
  int cols = GetParam().cols;
  int rows = GetParam().rows;
  bool borderWrap = GetParam().borderWrap;

  print("A: ", cols, rows, slices, matrix_data);

  agile::GPUVector<float> data_gpu;
  data_gpu.assignFromHost(matrix_data.begin(), matrix_data.end());
  agile::GPUVector<float> dx_gpu(rows * cols * slices);
  dx_gpu.assign(rows * cols * slices, 0.0f);

  std::vector<float> dx(cols * rows * slices);

  agile::lowlevel::diff3trans(1, cols, rows, data_gpu.data(), dx_gpu.data(),
                              rows * cols * slices, borderWrap);

  dx_gpu.copyToHost(dx);

  print("dx: ", cols, rows, slices, dx);
  EXPECT_EQ(dx[XYZ2Lin(2, 0, 1, cols, rows)], -1);

  // true if border is wrapped
  if (borderWrap)
  {
    EXPECT_EQ(dx[XYZ2Lin(cols - 1, 0, slices - 1, cols, rows)], -1);
    EXPECT_EQ(dx[XYZ2Lin(0, 0, 0, cols, rows)], cols - 1);
    EXPECT_EQ(dx[XYZ2Lin(cols - 1, 0, 0, cols, rows)], -1);
  }
  else
  {
    EXPECT_EQ(dx[XYZ2Lin(cols - 1, 0, slices - 1, cols, rows)],
              XYZ2Lin(cols - 1, 0, slices - 1, cols, rows) - 1);
    EXPECT_EQ(dx[XYZ2Lin(0, 0, 0, cols, rows)], 0);
    EXPECT_EQ(dx[XYZ2Lin(cols - 1, 0, 0, cols, rows)],
              XYZ2Lin(cols - 1, 0, 0, cols, rows) - 1);
  }
}

TEST_P(Derivative_3D, BackwardNabla_x)
{
  int cols = GetParam().cols;
  int rows = GetParam().rows;
  bool borderWrap = GetParam().borderWrap;

  print("A: ", cols, rows, slices, matrix_data);

  agile::GPUVector<float> data_gpu;
  data_gpu.assignFromHost(matrix_data.begin(), matrix_data.end());
  agile::GPUVector<float> dx_gpu(rows * cols * slices);
  dx_gpu.assign(rows * cols * slices, 0.0f);

  std::vector<float> dx(cols * rows * slices);

  agile::lowlevel::bdiff3(1, cols, rows, data_gpu.data(), dx_gpu.data(),
                          rows * cols * slices, borderWrap);

  dx_gpu.copyToHost(dx);

  print("bdx: ", cols, rows, slices, dx);

  EXPECT_EQ(dx[XY2Lin(1, 0, cols)], 1);
  if (borderWrap)
  {
    EXPECT_EQ(dx[XY2Lin(0, 0, cols)], -cols + 1);
    EXPECT_EQ(dx[XY2Lin(cols - 1, 0, cols)], 1);
  }
  else
  {
    EXPECT_EQ(dx[XY2Lin(0, 0, cols)], 0);
    EXPECT_EQ(dx[XY2Lin(cols - 1, 0, cols)], -cols + 2);
  }
}

TEST_P(Derivative_3D, BackwardNabla_transpose_x)
{
  int cols = GetParam().cols;
  int rows = GetParam().rows;
  bool borderWrap = GetParam().borderWrap;

  print("A: ", cols, rows, slices, matrix_data);

  agile::GPUVector<float> data_gpu;
  data_gpu.assignFromHost(matrix_data.begin(), matrix_data.end());
  agile::GPUVector<float> dx_gpu(rows * cols * slices);
  dx_gpu.assign(rows * cols * slices, 0.0f);

  std::vector<float> dx(cols * rows * slices);

  agile::lowlevel::bdiff3trans(1, cols, rows, data_gpu.data(), dx_gpu.data(),
                               rows * cols * slices, borderWrap);

  dx_gpu.copyToHost(dx);

  print("bdx: ", cols, rows, slices, dx);
  EXPECT_EQ(dx[XYZ2Lin(0, 0, 0, cols, rows)], -1);
  EXPECT_EQ(dx[XYZ2Lin(2, 0, 1, cols, rows)], -1);

  // true if border is wrapped
  if (borderWrap)
  {
    EXPECT_EQ(dx[XYZ2Lin(cols - 1, 0, 0, cols, rows)], (int)cols - 1);
    EXPECT_EQ(dx[XYZ2Lin(cols - 1, 0, slices - 1, cols, rows)], (int)cols - 1);
  }
  else
  {
    EXPECT_EQ(dx[XYZ2Lin(cols - 1, 0, 0, cols, rows)], 0);
    EXPECT_EQ(dx[XYZ2Lin(cols - 1, 0, slices - 1, cols, rows)], 0);
  }
}

TEST_P(Derivative_3D, Nabla_y)
{
  int cols = GetParam().cols;
  int rows = GetParam().rows;
  bool borderWrap = GetParam().borderWrap;

  print("A: ", cols, rows, slices, matrix_data);

  agile::GPUVector<float> data_gpu;
  data_gpu.assignFromHost(matrix_data.begin(), matrix_data.end());
  agile::GPUVector<float> dy_gpu(rows * cols * slices);
  dy_gpu.assign(rows * cols * slices, 0.0f);

  std::vector<float> dy(cols * rows * slices);

  agile::lowlevel::diff3(2, cols, rows, data_gpu.data(), dy_gpu.data(),
                         rows * cols * slices, borderWrap);

  dy_gpu.copyToHost(dy);

  print("dy: ", cols, rows, slices, dy);
  EXPECT_EQ(dy[XYZ2Lin(0, 0, 0, cols, rows)], +cols);
  EXPECT_EQ(dy[XYZ2Lin(2, 0, 1, cols, rows)], +cols);

  // true if border is wrapped
  if (borderWrap)
  {
    EXPECT_EQ(dy[XYZ2Lin(0, rows - 1, 0, cols, rows)],
              -1 * (cols) * (rows - 1));
    EXPECT_EQ(dy[XYZ2Lin(0, rows - 1, slices - 1, cols, rows)],
              -1 * (cols) * (rows - 1));
  }
  else
  {
    EXPECT_EQ(dy[XYZ2Lin(0, rows - 1, 0, cols, rows)], 0);
    EXPECT_EQ(dy[XYZ2Lin(0, rows - 1, slices - 1, cols, rows)], 0);
  }
}

TEST_P(Derivative_3D, Nabla_transpose_y)
{
  int cols = GetParam().cols;
  int rows = GetParam().rows;
  bool borderWrap = GetParam().borderWrap;

  print("A: ", cols, rows, slices, matrix_data);

  agile::GPUVector<float> data_gpu;
  data_gpu.assignFromHost(matrix_data.begin(), matrix_data.end());
  agile::GPUVector<float> dy_gpu(rows * cols * slices);
  dy_gpu.assign(rows * cols * slices, 0.0f);

  std::vector<float> dy(cols * rows * slices);

  agile::lowlevel::diff3trans(2, cols, rows, data_gpu.data(), dy_gpu.data(),
                              rows * cols * slices, borderWrap);

  dy_gpu.copyToHost(dy);

  print("dy: ", cols, rows, slices, dy);

  // true if border is wrapped
  if (borderWrap)
  {
    EXPECT_EQ(dy[XYZ2Lin(0, rows - 1, 0, cols, rows)], -1 * (cols));
    EXPECT_EQ(dy[XYZ2Lin(0, rows - 1, slices - 1, cols, rows)], -1 * (cols));
    EXPECT_EQ(dy[XYZ2Lin(0, 0, 0, cols, rows)], +cols * (rows - 1));
    EXPECT_EQ(dy[XYZ2Lin(2, 0, 1, cols, rows)], +cols * (rows - 1));
  }
  else
  {
    EXPECT_EQ(dy[XYZ2Lin(0, rows - 1, 0, cols, rows)],
              matrix_data[XYZ2Lin(0, rows - 2, 0, cols, rows)]);
    EXPECT_EQ(dy[XYZ2Lin(0, rows - 1, slices - 1, cols, rows)],
              matrix_data[XYZ2Lin(0, rows - 2, slices - 1, cols, rows)]);
    EXPECT_EQ(dy[XYZ2Lin(0, 0, 0, cols, rows)], 0);
    EXPECT_EQ(dy[XYZ2Lin(2, 0, 1, cols, rows)],
              -1 * matrix_data[XYZ2Lin(2, 0, 1, cols, rows)]);
  }
}

TEST_P(Derivative_3D, BackwardNabla_y)
{
  int cols = GetParam().cols;
  int rows = GetParam().rows;
  bool borderWrap = GetParam().borderWrap;

  print("A: ", cols, rows, slices, matrix_data);

  agile::GPUVector<float> data_gpu;
  data_gpu.assignFromHost(matrix_data.begin(), matrix_data.end());
  agile::GPUVector<float> dy_gpu(rows * cols * slices);
  dy_gpu.assign(rows * cols * slices, 0.0f);

  std::vector<float> dy(cols * rows * slices);

  agile::lowlevel::bdiff3(2, cols, rows, data_gpu.data(), dy_gpu.data(),
                          rows * cols * slices, borderWrap);

  dy_gpu.copyToHost(dy);

  print("bdy: ", cols, rows, slices, dy);

  // true if border is wrapped
  if (borderWrap)
  {
    EXPECT_EQ(dy[XYZ2Lin(0, rows - 1, 0, cols, rows)], (cols));
    EXPECT_EQ(dy[XYZ2Lin(0, rows - 1, slices - 1, cols, rows)], (cols));
    EXPECT_EQ(dy[XYZ2Lin(0, 0, 0, cols, rows)], -cols * (rows - 1));
    EXPECT_EQ(dy[XYZ2Lin(2, 0, 1, cols, rows)], -cols * (rows - 1));
  }
  else
  {
    EXPECT_EQ(dy[XYZ2Lin(0, rows - 1, 0, cols, rows)],
              -1 * matrix_data[XYZ2Lin(0, rows - 2, 0, cols, rows)]);
    EXPECT_EQ(dy[XYZ2Lin(0, rows - 1, slices - 1, cols, rows)],
              -1 * matrix_data[XYZ2Lin(0, rows - 2, slices - 1, cols, rows)]);
    EXPECT_EQ(dy[XYZ2Lin(0, 0, 0, cols, rows)], 0);
    EXPECT_EQ(dy[XYZ2Lin(2, 0, 1, cols, rows)],
              matrix_data[XYZ2Lin(2, 0, 1, cols, rows)]);
  }
}

TEST_P(Derivative_3D, BackwardNabla_transpose_y)
{
  int cols = GetParam().cols;
  int rows = GetParam().rows;
  bool borderWrap = GetParam().borderWrap;

  print("A: ", cols, rows, slices, matrix_data);

  agile::GPUVector<float> data_gpu;
  data_gpu.assignFromHost(matrix_data.begin(), matrix_data.end());
  agile::GPUVector<float> dy_gpu(rows * cols * slices);
  dy_gpu.assign(rows * cols * slices, 0.0f);

  std::vector<float> dy(cols * rows * slices);

  agile::lowlevel::bdiff3trans(2, cols, rows, data_gpu.data(), dy_gpu.data(),
                               rows * cols * slices, borderWrap);

  dy_gpu.copyToHost(dy);

  print("dy: ", cols, rows, slices, dy);
  EXPECT_EQ(dy[XYZ2Lin(0, 0, 0, cols, rows)], -cols);
  EXPECT_EQ(dy[XYZ2Lin(2, 0, 1, cols, rows)], -cols);

  // true if border is wrapped
  if (borderWrap)
  {
    EXPECT_EQ(dy[XYZ2Lin(0, rows - 1, 0, cols, rows)], (cols) * (rows - 1));
    EXPECT_EQ(dy[XYZ2Lin(0, rows - 1, slices - 1, cols, rows)],
              (cols) * (rows - 1));
  }
  else
  {
    EXPECT_EQ(dy[XYZ2Lin(0, rows - 1, 0, cols, rows)], 0);
    EXPECT_EQ(dy[XYZ2Lin(0, rows - 1, slices - 1, cols, rows)], 0);
  }
}

TEST_P(Derivative_3D, Nabla_z)
{
  int cols = GetParam().cols;
  int rows = GetParam().rows;
  bool borderWrap = GetParam().borderWrap;

  print("A: ", cols, rows, slices, matrix_data);

  agile::GPUVector<float> data_gpu;
  data_gpu.assignFromHost(matrix_data.begin(), matrix_data.end());
  agile::GPUVector<float> dz_gpu(rows * cols * slices);
  dz_gpu.assign(rows * cols * slices, 0.0f);

  std::vector<float> dz(cols * rows * slices);

  agile::lowlevel::diff3(3, cols, rows, data_gpu.data(), dz_gpu.data(),
                         rows * cols * slices, borderWrap);

  dz_gpu.copyToHost(dz);

  print("dz: ", cols, rows, slices, dz);
  EXPECT_EQ(dz[XYZ2Lin(0, 0, 0, cols, rows)], rows * cols);
  EXPECT_EQ(dz[XYZ2Lin(2, 1, 0, cols, rows)], rows * cols);
  EXPECT_EQ(dz[XYZ2Lin(0, rows - 1, 0, cols, rows)], rows * cols);
  // true if border is wrapped
  if (borderWrap)
  {
    EXPECT_EQ(dz[XYZ2Lin(0, 0, slices - 1, cols, rows)],
              -1 * (cols)*rows * (slices - 1));
    EXPECT_EQ(dz[XYZ2Lin(0, rows - 1, slices - 1, cols, rows)],
              -1 * (cols)*rows * (slices - 1));
  }
  else
  {
    EXPECT_EQ(dz[XYZ2Lin(0, 0, slices - 1, cols, rows)], 0);
    EXPECT_EQ(dz[XYZ2Lin(0, rows - 1, slices - 1, cols, rows)], 0);
  }
}

TEST_P(Derivative_3D, Nabla_transpose_z)
{
  int cols = GetParam().cols;
  int rows = GetParam().rows;
  bool borderWrap = GetParam().borderWrap;

  print("A: ", cols, rows, slices, matrix_data);

  agile::GPUVector<float> data_gpu;
  data_gpu.assignFromHost(matrix_data.begin(), matrix_data.end());
  agile::GPUVector<float> dz_gpu(rows * cols * slices);
  dz_gpu.assign(rows * cols * slices, 0.0f);

  std::vector<float> dz(cols * rows * slices);

  agile::lowlevel::diff3trans(3, cols, rows, data_gpu.data(), dz_gpu.data(),
                              rows * cols * slices, borderWrap);

  dz_gpu.copyToHost(dz);

  print("dz: ", cols, rows, slices, dz);

  // true if border is wrapped
  if (borderWrap)
  {
    EXPECT_EQ(dz[XYZ2Lin(0, 0, slices - 1, cols, rows)], -1 * cols * rows);
    EXPECT_EQ(dz[XYZ2Lin(0, 0, 0, cols, rows)], rows * cols * (slices - 1));
    EXPECT_EQ(dz[XYZ2Lin(2, 1, 0, cols, rows)], rows * cols * (slices - 1));
    EXPECT_EQ(dz[XYZ2Lin(0, rows - 1, 0, cols, rows)],
              rows * cols * (slices - 1));
    EXPECT_EQ(dz[XYZ2Lin(0, rows - 1, slices - 1, cols, rows)],
              -1 * cols * rows);
  }
  else
  {
    EXPECT_EQ(dz[XYZ2Lin(0, 0, slices - 1, cols, rows)],
              matrix_data[XYZ2Lin(0, 0, slices - 2, cols, rows)]);
    EXPECT_EQ(dz[XYZ2Lin(0, 0, 0, cols, rows)], 0);
    EXPECT_EQ(dz[XYZ2Lin(2, 1, 0, cols, rows)],
              -1 * matrix_data[XYZ2Lin(2, 1, 0, cols, rows)]);
    EXPECT_EQ(dz[XYZ2Lin(0, rows - 1, 0, cols, rows)],
              -1 * matrix_data[XYZ2Lin(0, rows - 1, 0, cols, rows)]);
    EXPECT_EQ(dz[XYZ2Lin(0, rows - 1, slices - 1, cols, rows)],
              matrix_data[XYZ2Lin(0, rows - 1, slices - 2, cols, rows)]);
  }
}

TEST_P(Derivative_3D, BackwardNabla_z)
{
  int cols = GetParam().cols;
  int rows = GetParam().rows;
  bool borderWrap = GetParam().borderWrap;

  print("A: ", cols, rows, slices, matrix_data);

  agile::GPUVector<float> data_gpu;
  data_gpu.assignFromHost(matrix_data.begin(), matrix_data.end());
  agile::GPUVector<float> dz_gpu(rows * cols * slices);
  dz_gpu.assign(rows * cols * slices, 0.0f);

  std::vector<float> dz(cols * rows * slices);

  agile::lowlevel::bdiff3(3, cols, rows, data_gpu.data(), dz_gpu.data(),
                          rows * cols * slices, borderWrap);

  dz_gpu.copyToHost(dz);

  print("dz: ", cols, rows, slices, dz);

  // true if border is wrapped
  if (borderWrap)
  {
    EXPECT_EQ(dz[XYZ2Lin(0, 0, slices - 1, cols, rows)], cols * rows);
    EXPECT_EQ(dz[XYZ2Lin(0, 0, 0, cols, rows)],
              -1 * rows * cols * (slices - 1));
    EXPECT_EQ(dz[XYZ2Lin(2, 1, 0, cols, rows)],
              -1 * rows * cols * (slices - 1));
    EXPECT_EQ(dz[XYZ2Lin(0, rows - 1, 0, cols, rows)],
              -1 * rows * cols * (slices - 1));
    EXPECT_EQ(dz[XYZ2Lin(0, rows - 1, slices - 1, cols, rows)], cols * rows);
  }
  else
  {
    EXPECT_EQ(dz[XYZ2Lin(0, 0, slices - 1, cols, rows)],
              -1.0 * matrix_data[XYZ2Lin(0, 0, slices - 2, cols, rows)]);
    EXPECT_EQ(dz[XYZ2Lin(0, 0, 0, cols, rows)], 0);
    EXPECT_EQ(dz[XYZ2Lin(2, 1, 0, cols, rows)],
              matrix_data[XYZ2Lin(2, 1, 0, cols, rows)]);
    EXPECT_EQ(dz[XYZ2Lin(0, rows - 1, 0, cols, rows)],
              matrix_data[XYZ2Lin(0, rows - 1, 0, cols, rows)]);
    EXPECT_EQ(dz[XYZ2Lin(0, rows - 1, slices - 1, cols, rows)],
              -1.0 * matrix_data[XYZ2Lin(0, rows - 1, slices - 2, cols, rows)]);
  }
}

TEST_P(Derivative_3D, BackwardNabla_transpose_z)
{
  int cols = GetParam().cols;
  int rows = GetParam().rows;
  bool borderWrap = GetParam().borderWrap;

  print("A: ", cols, rows, slices, matrix_data);

  agile::GPUVector<float> data_gpu;
  data_gpu.assignFromHost(matrix_data.begin(), matrix_data.end());
  agile::GPUVector<float> dz_gpu(rows * cols * slices);
  dz_gpu.assign(rows * cols * slices, 0.0f);

  std::vector<float> dz(cols * rows * slices);

  agile::lowlevel::bdiff3trans(3, cols, rows, data_gpu.data(), dz_gpu.data(),
                               rows * cols * slices, borderWrap);

  dz_gpu.copyToHost(dz);

  print("dz: ", cols, rows, slices, dz);
  EXPECT_EQ(dz[XYZ2Lin(0, 0, 0, cols, rows)], -1 * rows * cols);
  EXPECT_EQ(dz[XYZ2Lin(2, 1, 0, cols, rows)], -1 * rows * cols);
  EXPECT_EQ(dz[XYZ2Lin(0, rows - 1, 0, cols, rows)], -1 * rows * cols);
  // true if border is wrapped
  if (borderWrap)
  {
    EXPECT_EQ(dz[XYZ2Lin(0, 0, slices - 1, cols, rows)],
              (cols)*rows * (slices - 1));
    EXPECT_EQ(dz[XYZ2Lin(0, rows - 1, slices - 1, cols, rows)],
              (cols)*rows * (slices - 1));
  }
  else
  {
    EXPECT_EQ(dz[XYZ2Lin(0, 0, slices - 1, cols, rows)], 0);
    EXPECT_EQ(dz[XYZ2Lin(0, rows - 1, slices - 1, cols, rows)], 0);
  }
}

TEST_F(Derivative_2D, Divergence)
{
  print("A: ", cols, rows, matrix_data);

  agile::GPUMatrix<float> data_gpu(rows, cols, &matrix_data[0]);
  agile::GPUMatrix<float> dx_gpu(rows, cols, NULL);
  agile::GPUMatrix<float> dy_gpu(rows, cols, NULL);
  agile::GPUMatrix<float> div_x_gpu(rows, cols, NULL);
  agile::GPUMatrix<float> div_y_gpu(rows, cols, NULL);
  agile::GPUMatrix<float> divergence_gpu(rows, cols, NULL);

  std::vector<float> divergence(cols * rows);

  // compute vector field -> gradient
  agile::diff(1, data_gpu, dx_gpu);
  agile::diff(2, data_gpu, dy_gpu);

  // compute divergence
  // of gradient -> Laplacian
  agile::difftrans(1, dx_gpu, div_x_gpu);
  agile::difftrans(2, dx_gpu, div_y_gpu);

  agile::addMatrix(div_x_gpu, div_y_gpu, divergence_gpu);

  divergence_gpu.copyToHost(divergence);

  print("divergence: ", cols, rows, divergence);
}

TEST_P(Derivative_3D, Divergence)
{
  int cols = GetParam().cols;
  int rows = GetParam().rows;

  print("A: ", cols, rows, slices, matrix_data);

  agile::GPUVector<float> data_gpu;
  data_gpu.assignFromHost(matrix_data.begin(), matrix_data.end());
  agile::GPUVector<float> dx_gpu(rows * cols * slices);
  agile::GPUVector<float> dy_gpu(rows * cols * slices);
  agile::GPUVector<float> dz_gpu(rows * cols * slices);
  agile::GPUVector<float> div_x_gpu(rows * cols * slices);
  agile::GPUVector<float> div_y_gpu(rows * cols * slices);
  agile::GPUVector<float> div_z_gpu(rows * cols * slices);
  agile::GPUVector<float> divergence_gpu(rows * cols * slices);

  std::vector<float> divergence(cols * rows * slices);

  // compute vector field -> gradient
  agile::lowlevel::diff3(1, cols, rows, data_gpu.data(), dx_gpu.data(),
                         rows * cols * slices, false);
  dx_gpu.copyToHost(divergence);
  print("dx_gpu: ", cols, rows, slices, divergence);

  agile::lowlevel::diff3(2, cols, rows, data_gpu.data(), dy_gpu.data(),
                         rows * cols * slices, false);
  dy_gpu.copyToHost(divergence);
  print("dy_gpu: ", cols, rows, slices, divergence);

  agile::lowlevel::diff3(3, cols, rows, data_gpu.data(), dz_gpu.data(),
                         rows * cols * slices, false);
  dz_gpu.copyToHost(divergence);
  print("dz_gpu: ", cols, rows, slices, divergence);

  // compute divergence
  // of gradient -> Laplacian
  agile::lowlevel::diff3trans(1, cols, rows, dx_gpu.data(), div_x_gpu.data(),
                              rows * cols * slices, false);
  div_x_gpu.copyToHost(divergence);
  print("dxt_gpu: ", cols, rows, slices, divergence);

  agile::lowlevel::diff3trans(2, cols, rows, dy_gpu.data(), div_y_gpu.data(),
                              rows * cols * slices, false);
  div_y_gpu.copyToHost(divergence);
  print("dyt_gpu: ", cols, rows, slices, divergence);

  agile::lowlevel::diff3trans(3, cols, rows, dz_gpu.data(), div_z_gpu.data(),
                              rows * cols * slices, false);
  div_z_gpu.copyToHost(divergence);
  print("dzt_gpu: ", cols, rows, slices, divergence);

  agile::addVector(div_x_gpu, div_y_gpu, divergence_gpu);

  divergence_gpu.copyToHost(divergence);
  print("laplacian (2d): ", cols, rows, slices, divergence);

  agile::addVector(divergence_gpu, div_z_gpu, divergence_gpu);

  divergence_gpu.copyToHost(divergence);

  print("divergence: ", cols, rows, slices, divergence);
  EXPECT_EQ(-1, divergence[XYZ2Lin(0, 2, 1, cols, rows)]);
  EXPECT_EQ(0, divergence[XYZ2Lin(1, 2, 1, cols, rows)]);
}

TEST_P(Derivative_3D, BackwardDivergence)
{
  int cols = GetParam().cols;
  int rows = GetParam().rows;
  int N = rows * cols * slices;

  print("A: ", cols, rows, slices, matrix_data);

  agile::GPUVector<float> data_gpu;
  data_gpu.assignFromHost(matrix_data.begin(), matrix_data.begin() + N);

  agile::GPUVector<float> dx_gpu(N);
  agile::GPUVector<float> dy_gpu(N);
  agile::GPUVector<float> dz_gpu(N);
  agile::GPUVector<float> div_x_gpu(N);
  agile::GPUVector<float> div_y_gpu(N);
  agile::GPUVector<float> div_z_gpu(N);
  agile::GPUVector<float> divergence_gpu(N);

  std::vector<float> divergence(N);

  // compute vector field -> gradient
  agile::lowlevel::bdiff3(1, cols, rows, data_gpu.data(), dx_gpu.data(), N,
                          false);
  dx_gpu.copyToHost(divergence);
  print("dx_gpu: ", cols, rows, slices, divergence);

  agile::lowlevel::bdiff3(2, cols, rows, data_gpu.data(), dy_gpu.data(), N,
                          false);
  dy_gpu.copyToHost(divergence);
  print("dy_gpu: ", cols, rows, slices, divergence);

  agile::lowlevel::bdiff3(3, cols, rows, data_gpu.data(), dz_gpu.data(), N,
                          false);
  dz_gpu.copyToHost(divergence);
  print("dz_gpu: ", cols, rows, slices, divergence);

  // compute divergence
  // of gradient -> Laplacian
  agile::lowlevel::bdiff3trans(1, cols, rows, dx_gpu.data(), div_x_gpu.data(),
                               N, false);
  div_x_gpu.copyToHost(divergence);
  print("dxt_gpu: ", cols, rows, slices, divergence);

  agile::lowlevel::bdiff3trans(2, cols, rows, dy_gpu.data(), div_y_gpu.data(),
                               N, false);
  div_y_gpu.copyToHost(divergence);
  print("dyt_gpu: ", cols, rows, slices, divergence);

  agile::lowlevel::bdiff3trans(3, cols, rows, dz_gpu.data(), div_z_gpu.data(),
                               N, false);
  div_z_gpu.copyToHost(divergence);
  print("dzt_gpu: ", cols, rows, slices, divergence);

  agile::addVector(div_x_gpu, div_y_gpu, divergence_gpu);

  divergence_gpu.copyToHost(divergence);
  print("laplacian (2d): ", cols, rows, slices, divergence);

  agile::addVector(divergence_gpu, div_z_gpu, divergence_gpu);

  divergence_gpu.copyToHost(divergence);

  print("divergence: ", cols, rows, slices, divergence);
  EXPECT_EQ(-1, divergence[XYZ2Lin(cols - 1, rows - 1, 0, cols, rows)]);
  EXPECT_EQ(0, divergence[XYZ2Lin(cols - 1, rows - 1, slices - 1, cols, rows)]);
  EXPECT_EQ(0, divergence[XYZ2Lin(cols - 1, rows - 3, slices - 1, cols, rows)]);
}
