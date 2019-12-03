#ifndef TEST_TEST_UTILS_H_

#define TEST_TEST_UTILS_H_

#include <iostream>
#include <iomanip>
#include <vector>
#include "cuda.h"
#include "agile/agile.hpp"

#define EPS 1E-3

inline int XY2Lin(int x, int y, int cols)
{
  return x + cols * y;
}

inline int2 Lin2XY(int index, int cols)
{
  int2 pos;
  pos.x = index % cols;
  pos.y = (int)(index / cols);
  return pos;
}

inline int XYZ2Lin(int x, int y, int z, int cols, int rows)
{
  return x + cols * (y + z * rows);
}

inline int3 Lin2XYZ(int index, int cols, int rows)
{
  int3 pos;
  int N = cols * rows;

  pos.x = index % cols;
  pos.z = (int)(index / N);
  int r = index - pos.z * N;
  pos.y = (int)(r / cols);
  return pos;
}

template <typename TType>
static void printValue(TType &value, bool printAbs = true)
{
  bool isComplex = agile::is_complex<TType>::value;
  std::cout << std::setw(10) << std::setprecision(4);
  if (isComplex)
  {
    if (printAbs)
      std::cout << std::abs(((std::complex<float>)value));
    else
      std::cout << ((std::complex<double>)value).real() << "+"
                << ((std::complex<double>)value).imag() << "i";
  }
  else
    std::cout << value;
}

static void print(const char *string, std::vector<float> &x)
{
  std::cout << string;
  for (unsigned counter = 0; counter < x.size(); ++counter)
    std::cout << x[counter] << " ";
  std::cout << std::endl;
}

template <typename TType>
static void print(const char *string, unsigned num_columns, unsigned num_rows,
                  TType data, bool printAbs = true)
{
  typename TType::iterator iter = data.begin();
  std::cout << string << std::endl;
  for (unsigned row = 0; row < num_rows; ++row)
  {
    std::cout << " ";
    for (unsigned column = 0; column < num_columns; ++column)
    {
      printValue(*iter++, printAbs);
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

template <typename TType>
static void print(const char *string, unsigned num_columns, unsigned num_rows,
                  unsigned num_slices, TType data, bool printAbs = true)
{
  typename TType::iterator iter = data.begin();
  std::cout << string << std::endl;
  for (unsigned slice = 0; slice < num_slices; ++slice)
  {
    std::cout << " --- slice : " << slice << " --- " << std::endl;
    for (unsigned row = 0; row < num_rows; ++row)
    {
      std::cout << " ";
      for (unsigned column = 0; column < num_columns; ++column)
      {
        printValue(*iter++, printAbs);
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}

template <typename TType>
static void print(const char *string, unsigned num_columns, unsigned num_rows,
                  unsigned num_slices, unsigned num_coils, TType data,
                  bool printAbs = true)
{
  typename TType::iterator iter = data.begin();
  std::cout << string << std::endl;
  for (unsigned coil = 0; coil < num_coils; ++coil)
  {
    std::cout << " --- coil : " << coil << " --- " << std::endl;
    for (unsigned slice = 0; slice < num_slices; ++slice)
    {
      std::cout << " --- slice : " << slice << " --- " << std::endl;
      for (unsigned row = 0; row < num_rows; ++row)
      {
        std::cout << " ";
        for (unsigned column = 0; column < num_columns; ++column)
        {
          printValue(*iter++, printAbs);
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
  }
}

static std::complex<float> randomValue()
{
  return std::complex<float>(
      static_cast<float>(rand()) / static_cast<float>(RAND_MAX),
      static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
}

#endif  // TEST_TEST_UTILS_H_
