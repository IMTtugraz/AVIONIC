#include <gtest/gtest.h>
#include <vector>

#include "./test_utils.h"
#include "agile/io/file.hpp"

TEST(Test_IO, ReadFloatVectorFile2D)
{
  const char *filename = "../test/data/test_vector_float_10x10x1x1.bin";

  unsigned width = 10;
  unsigned height = 10;

  std::vector<float> data;
  agile::readVectorFile(filename, data);

  print("input data:", width, height, data);
}

TEST(Test_IO, ReadFloatVectorFile3D)
{
  const char *filename = "../test/data/test_vector_float_10x10x5x1.bin";

  unsigned width = 10;
  unsigned height = 10;
  unsigned coils = 5;

  std::vector<float> data;
  agile::readVectorFile(filename, data);

  print("input data:", width, height, coils, data);
}

TEST(Test_IO, ReadFloatVectorFile4D)
{
  const char *filename = "../test/data/test_vector_float_10x10x5x3.bin";

  unsigned width = 10;
  unsigned height = 10;
  unsigned frames = 3;
  unsigned coils = 5;

  std::vector<float> data;
  agile::readVectorFile(filename, data);

  print("input data:", width, height, frames, coils, data);
}

TEST(Test_IO, ReadComplexVectorFile2D)
{
  const char *filename = "../test/data/test_vector_complex_10x10x1x1.bin";

  unsigned width = 10;
  unsigned height = 10;

  std::vector<std::complex<double> > data;
  agile::readVectorFile(filename, data);

  print("input data:", width, height, data);

  EXPECT_NEAR(0,data[0].real(),EPS);
  EXPECT_NEAR(0,data[0].imag(),EPS);
  EXPECT_NEAR(45,data[45].real(),EPS);
  EXPECT_NEAR(45,data[45].imag(),EPS);
  EXPECT_NEAR(99,data[99].real(),EPS);
  EXPECT_NEAR(99,data[99].imag(),EPS);
}

TEST(Test_IO, ReadWriteFloatVectorFile2D)
{
  agile::GPUEnvironment::allocateGPU(0);

  const char *filename = "../test/data/test_vector_float_10x10x1x1.bin";
  const char *output = "../test/data/output/test_vector_float_10x10x1x1.bin";

  unsigned width = 10;
  unsigned height = 10;
  unsigned frames = 1;

  std::vector<float> data;
  agile::readVectorFile(filename, data);

  print("input data:", width, height, data);

  //TODO compute nabla x
  agile::GPUVector<float> data_gpu;
  data_gpu.assignFromHost(data.begin(), data.end());
  
  std::vector<float> dx(width * height * frames);
  agile::GPUVector<float> dx_gpu(height * width * frames);
  dx_gpu.assign(height * width * frames, 0.0f);

  agile::lowlevel::diff3(1, width, height, data_gpu.data(), dx_gpu.data(),
                         height * width * frames, false);
  dx_gpu.copyToHost(dx);

  print("dx: ", width, height, frames, dx);
  agile::writeVectorFile(output, dx);
}
