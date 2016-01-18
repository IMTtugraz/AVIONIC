#include <gtest/gtest.h>
#include <complex>

#include "agile/gpu_environment.hpp"
#include "agile/gpu_vector.hpp"
#include "agile/gpu_matrix.hpp"
#include "agile/calc/fft.hpp"
#include "./test_utils.h"

typedef std::complex<float> DType;

class FFT_Test : public ::testing::Test
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
              DType(XYZ2Lin(column, row, slice, max_cols, max_rows), 0));
  }
  virtual void SetUp()
  {
    agile::GPUEnvironment::allocateGPU(0);
  }
  static std::vector<DType> matrix_data;
};

std::vector<DType> FFT_Test::matrix_data;

TEST_F(FFT_Test, Forward)
{
  unsigned int cols = 10;
  unsigned int rows = 10;

  print("A: ", cols, rows, matrix_data);

  agile::FFT<DType> *fftOp = new agile::FFT<DType>(rows, cols);

  agile::GPUMatrix<DType> matrix_gpu(rows, cols, &matrix_data[0]);
  agile::GPUMatrix<DType> k_matrix_gpu(rows, cols, NULL);

  agile::copy(matrix_gpu, k_matrix_gpu);

  fftOp->CenterdFFT(matrix_gpu, k_matrix_gpu);

  // Result CPU vector
  std::vector<DType> k_matrix(cols * rows);

  k_matrix_gpu.copyToHost(k_matrix);
  print("FFT(A)", cols, rows, k_matrix);
  EXPECT_NEAR(50.0, std::abs(k_matrix[XY2Lin(5, 0, cols)]), EPS);
  EXPECT_NEAR(52.573, std::abs(k_matrix[XY2Lin(5, 1, cols)]), EPS);
  EXPECT_NEAR(61.803, std::abs(k_matrix[XY2Lin(5, 2, cols)]), EPS);
  EXPECT_NEAR(495.0, std::abs(k_matrix[XY2Lin(5, 5, cols)]), EPS);
  EXPECT_NEAR(5.0, std::abs(k_matrix[XY2Lin(0, 5, cols)]), EPS);
  EXPECT_NEAR(5.257, std::abs(k_matrix[XY2Lin(1, 5, cols)]), EPS);
  EXPECT_NEAR(16.180, std::abs(k_matrix[XY2Lin(6, 5, cols)]), EPS);
}

TEST_F(FFT_Test, Shift)
{
  unsigned int cols = 10;
  unsigned int rows = 10;

  print("A: ", cols, rows, matrix_data);

  agile::GPUVector<DType> vector_gpu(rows * cols);
  vector_gpu.assignFromHost(matrix_data.begin(), matrix_data.end());

  agile::lowlevel::fftshift(vector_gpu.data(), rows, cols);

  // Result CPU vector
  std::vector<DType> result(cols * rows);

  vector_gpu.copyToHost(result);
  print("FFT(A)", cols, rows, result);
}

TEST_F(FFT_Test, ShiftMultiple)
{
  unsigned int cols = 10;
  unsigned int rows = 10;
  unsigned int slices = 2;

  std::vector<DType> matrix_data;
  for (unsigned slice = 0; slice < slices; slice++)
    for (unsigned row = 0; row < rows; ++row)
      for (unsigned column = 0; column < cols; ++column)
        matrix_data.push_back(
            DType(XYZ2Lin(column, row, slice, cols, rows), 0));

  print("A: ", cols, rows, slices, matrix_data);

  agile::GPUVector<DType> vector_gpu(rows * cols);
  vector_gpu.assignFromHost(matrix_data.begin(), matrix_data.end());

  agile::lowlevel::fftshift(vector_gpu.data(), rows, cols);
  agile::lowlevel::fftshift(vector_gpu.data()+rows*cols, rows, cols);

  // Result CPU vector
  std::vector<DType> result(cols * rows);

  vector_gpu.copyToHost(result);
  print("FFT(A)", cols, rows, slices, result);
}

TEST_F(FFT_Test, ForwardWithVector)
{
  unsigned int cols = 10;
  unsigned int rows = 10;

  print("A: ", cols, rows, matrix_data);

  agile::FFT<DType> *fftOp = new agile::FFT<DType>(rows, cols);

  agile::GPUVector<DType> vector_gpu(rows * cols);
  vector_gpu.assignFromHost(matrix_data.begin(), matrix_data.end());
  agile::GPUVector<DType> k_vector_gpu(rows * cols);

  fftOp->CenteredForward(vector_gpu, k_vector_gpu);

  // Result CPU vector
  std::vector<DType> k_vector(cols * rows);

  k_vector_gpu.copyToHost(k_vector);
  print("FFT(A)", cols, rows, k_vector);
  EXPECT_NEAR(50.0, std::abs(k_vector[XY2Lin(5, 0, cols)]), EPS);
  EXPECT_NEAR(52.573, std::abs(k_vector[XY2Lin(5, 1, cols)]), EPS);
  EXPECT_NEAR(61.803, std::abs(k_vector[XY2Lin(5, 2, cols)]), EPS);
  EXPECT_NEAR(495.0, std::abs(k_vector[XY2Lin(5, 5, cols)]), EPS);
  EXPECT_NEAR(5.0, std::abs(k_vector[XY2Lin(0, 5, cols)]), EPS);
  EXPECT_NEAR(5.257, std::abs(k_vector[XY2Lin(1, 5, cols)]), EPS);
  EXPECT_NEAR(16.180, std::abs(k_vector[XY2Lin(6, 5, cols)]), EPS);
}

TEST_F(FFT_Test, ForwardBackward)
{
  unsigned int cols = 10;
  unsigned int rows = 10;

  print("A: ", cols, rows, matrix_data);

  agile::FFT<DType> *fftOp = new agile::FFT<DType>(rows, cols);

  agile::GPUMatrix<DType> matrix_gpu(rows, cols, &matrix_data[0]);
  agile::GPUMatrix<DType> k_matrix_gpu(rows, cols, NULL);

  agile::copy(matrix_gpu, k_matrix_gpu);

  fftOp->CenterdFFT(matrix_gpu, k_matrix_gpu);
  fftOp->CenterdIFFT(k_matrix_gpu, matrix_gpu);

  // Result CPU vector
  std::vector<DType> k_matrix(cols * rows);

  matrix_gpu.copyToHost(k_matrix);
  print("IFFT(FFT(A)): ", cols, rows, k_matrix);

  for (unsigned row = 0; row < rows; row++)
    for (unsigned col = 0; col < cols; col++)
    {
      int ind = XY2Lin(col, row, cols);
      EXPECT_NEAR(std::abs(matrix_data[ind]), std::abs(k_matrix[ind]), EPS);
    }
}

TEST_F(FFT_Test, ForwardBackwardWithVector)
{
  unsigned int cols = 10;
  unsigned int rows = 10;

  print("A: ", cols, rows, matrix_data);

  agile::FFT<DType> *fftOp = new agile::FFT<DType>(rows, cols);

  agile::GPUVector<DType> vector_gpu(rows * cols);
  vector_gpu.assignFromHost(matrix_data.begin(), matrix_data.end());
  agile::GPUVector<DType> k_vector_gpu(rows * cols);

  fftOp->CenteredForward(vector_gpu, k_vector_gpu);
  fftOp->CenteredInverse(k_vector_gpu, vector_gpu);

  // Result CPU vector
  std::vector<DType> k_vector(cols * rows);

  vector_gpu.copyToHost(k_vector);
  print("IFFT(FFT(A)): ", cols, rows, k_vector);

  for (unsigned row = 0; row < rows; row++)
    for (unsigned col = 0; col < cols; col++)
    {
      int ind = XY2Lin(col, row, cols);
      EXPECT_NEAR(std::abs(matrix_data[ind]), std::abs(k_vector[ind]), EPS);
    }
}

TEST_F(FFT_Test, ForwardBackwardWithVectorMultiple)
{
  unsigned int cols = 10;
  unsigned int rows = 10;
  unsigned int slices = 2;

  print("A: ", cols, rows, slices, matrix_data);

  agile::FFT<DType> *fftOp = new agile::FFT<DType>(rows, cols);

  agile::GPUVector<DType> vector_gpu(rows * cols * slices);
  vector_gpu.assignFromHost(matrix_data.begin(), matrix_data.end());
  agile::GPUVector<DType> k_vector_gpu(rows * cols * slices);

  for (unsigned slice = 0; slice < slices; slice++)
  {
    unsigned int offset = cols*rows*slice;
    fftOp->CenteredForward(vector_gpu, k_vector_gpu, offset);
    fftOp->CenteredInverse(k_vector_gpu, vector_gpu, offset);
  }

  // Result CPU vector
  std::vector<DType> k_vector(cols * rows * slices);

  vector_gpu.copyToHost(k_vector);
  print("IFFT(FFT(A)): ", cols, rows, slices, k_vector);

  for (unsigned slice = 0; slice < slices; slice++)
  for (unsigned row = 0; row < rows; row++)
    for (unsigned col = 0; col < cols; col++)
    {
      int ind = XYZ2Lin(col, row, slice, cols, rows);
      EXPECT_NEAR(std::abs(matrix_data[ind]), std::abs(k_vector[ind]), EPS);
    }
}
