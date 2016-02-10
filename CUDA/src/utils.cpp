#include "../include/utils.h"
#include "../include/types.h"
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include "ismrmrd/ismrmrd.h"
#include "ismrmrd/dataset.h"

// Transcribed from MATLAB ellipke function
// for a single value
//
// References:
// [1] M. Abramowitz and I.A. Stegun, "Handbook of Mathematical
//     Functions" Dover Publications", 1965, 17.6.
RType utils::Ellipke(RType value)
{
  if (value < 0 || value > 1)
    throw std::invalid_argument(
        "Ellipke: Only values in the range of [0,1] are allowed.");

  RType a0 = 1.0;
  RType b0 = std::sqrt(1.0 - value);
  RType s0 = value;
  int i1 = 0;
  RType mm = 1;
  RType tol = 1E-7;

  RType a1, b1, c1, w1;
  while (mm > tol)
  {
    a1 = (a0 + b0) / 2.0;
    b1 = std::sqrt(a0 * b0);
    c1 = (a0 - b0) / 2.0;
    i1++;
    w1 = std::pow(2.0, i1) * std::pow(c1, 2.0);
    mm = w1;
    s0 = s0 + w1;
    a0 = a1;
    b0 = b1;
  }
  RType k = M_PI / (2.0 * a1);
  RType e = k * (1.0 - s0 / 2.0);
  return e;
}

void utils::Gradient(CVector &data_gpu, std::vector<CVector> &gradient,
                     unsigned width, unsigned height, DType dx, DType dy,
                     DType dz)
{
  unsigned int N = data_gpu.size();
  agile::lowlevel::diff3(1, width, height, data_gpu.data(), gradient[0].data(),
                         N, false);
  if (dx != 1.0)
    agile::scale((DType)1.0 / dx, gradient[0], gradient[0]);

  agile::lowlevel::diff3(2, width, height, data_gpu.data(), gradient[1].data(),
                         N, false);
  if (dy != 1.0)
    agile::scale((DType)1.0 / dy, gradient[1], gradient[1]);

  agile::lowlevel::diff3(3, width, height, data_gpu.data(), gradient[2].data(),
                         N, false);
  if (dz != 1.0)
    agile::scale((DType)1.0 / dz, gradient[2], gradient[2]);
}

std::vector<CVector> utils::Gradient(CVector &data_gpu, unsigned width,
                                     unsigned height, DType dx, DType dy,
                                     DType dz)
{
  unsigned int N = data_gpu.size();
  std::vector<CVector> gradient;
  gradient.push_back(CVector(N));  //< x
  gradient.push_back(CVector(N));  //< y
  gradient.push_back(CVector(N));  //< z

  Gradient(data_gpu, gradient, width, height, dx, dy, dz);
  return gradient;
}

void utils::Gradient2D(CVector &data_gpu, std::vector<CVector> &gradient,
                       unsigned width, unsigned height, DType dx, DType dy)
{
  unsigned int N = data_gpu.size();
  agile::lowlevel::diff3(1, width, height, data_gpu.data(), gradient[0].data(),
                         N, false);
  if (dx != 1.0)
    agile::scale((DType)1.0 / dx, gradient[0], gradient[0]);

  agile::lowlevel::diff3(2, width, height, data_gpu.data(), gradient[1].data(),
                         N, false);
  if (dy != 1.0)
    agile::scale((DType)1.0 / dy, gradient[1], gradient[1]);
}

std::vector<CVector> utils::Gradient2D(CVector &data_gpu, unsigned width,
                                       unsigned height, DType dx, DType dy)
{
  unsigned int N = data_gpu.size();
  std::vector<CVector> gradient;
  gradient.push_back(CVector(N));  //< x
  gradient.push_back(CVector(N));  //< y

  Gradient2D(data_gpu, gradient, width, height, dx, dy);
  return gradient;
}

void utils::GradientNorm(const std::vector<CVector> &gradient, CVector &norm)
{
  unsigned int N = gradient[0].size();
  CVector temp(N);

  // compute norm sqrt(abs(dx).^2 + abs(dy).^2 + abs(dz).^2)
  agile::multiplyConjElementwise(gradient[0], gradient[0], norm);
  agile::multiplyConjElementwise(gradient[1], gradient[1], temp);
  agile::addVector(temp, norm, norm);
  agile::multiplyConjElementwise(gradient[2], gradient[2], temp);
  agile::addVector(temp, norm, norm);

  agile::sqrt(norm, norm);
}

CVector utils::GradientNorm(const std::vector<CVector> &gradient)
{
  unsigned int N = gradient[0].size();
  CVector norm_gpu(N);
  utils::GradientNorm(gradient, norm_gpu);
  return norm_gpu;
}

void utils::GradientNorm2D(const std::vector<CVector> &gradient, CVector &norm)
{
  unsigned int N = gradient[0].size();
  CVector temp(N);

  // compute norm sqrt(abs(dx).^2 + abs(dy).^2 + abs(dz).^2)
  agile::multiplyConjElementwise(gradient[0], gradient[0], norm);
  agile::multiplyConjElementwise(gradient[1], gradient[1], temp);
  agile::addVector(temp, norm, norm);

  agile::sqrt(norm, norm);
}

CVector utils::GradientNorm2D(const std::vector<CVector> &gradient)
{
  unsigned int N = gradient[0].size();
  CVector norm_gpu(N);
  utils::GradientNorm2D(gradient, norm_gpu);
  return norm_gpu;
}

void utils::SymmetricGradient(const std::vector<CVector> &data_gpu,
                              std::vector<CVector> &gradient, unsigned width,
                              unsigned height, DType dx, DType dy, DType dz)
{
  unsigned int N = data_gpu[0].size();
  CVector temp(N);

  // dxx
  agile::lowlevel::bdiff3(1, width, height, data_gpu[0].data(),
                          gradient[0].data(), N, false);
  if (dx != 1.0)
    agile::scale((DType)1.0 / dx, gradient[0], gradient[0]);

  // dyy
  agile::lowlevel::bdiff3(2, width, height, data_gpu[1].data(),
                          gradient[1].data(), N, false);

  if (dy != 1.0)
    agile::scale((DType)1.0 / dy, gradient[1], gradient[1]);

  // dzz
  agile::lowlevel::bdiff3(3, width, height, data_gpu[2].data(),
                          gradient[2].data(), N, false);

  if (dz != 1.0)
    agile::scale((DType)1.0 / dz, gradient[2], gradient[2]);

  // dxy
  agile::lowlevel::bdiff3(2, width, height, data_gpu[0].data(),
                          gradient[3].data(), N, false);
  agile::lowlevel::bdiff3(1, width, height, data_gpu[1].data(), temp.data(), N,
                          false);
  agile::addVector(gradient[3], temp, gradient[3]);
  agile::scale((DType)(1.0 / (2.0 * dx)), gradient[3], gradient[3]);

  // dxz
  agile::lowlevel::bdiff3(3, width, height, data_gpu[0].data(),
                          gradient[4].data(), N, false);
  agile::scale((DType)1.0 / (dz), gradient[4], gradient[4]);
  agile::lowlevel::bdiff3(1, width, height, data_gpu[2].data(), temp.data(), N,
                          false);
  agile::addScaledVector(gradient[4], (DType)1.0 / dx, temp, gradient[4]);
  agile::scale(0.5f, gradient[4], gradient[4]);

  // dyz
  agile::lowlevel::bdiff3(3, width, height, data_gpu[1].data(),
                          gradient[5].data(), N, false);
  agile::scale((DType)1.0 / (dz), gradient[5], gradient[5]);
  agile::lowlevel::bdiff3(2, width, height, data_gpu[2].data(), temp.data(), N,
                          false);
  agile::addScaledVector(gradient[5], (DType)1.0 / dx, temp, gradient[5]);
  agile::scale(0.5f, gradient[5], gradient[5]);
}

std::vector<CVector>
utils::SymmetricGradient(const std::vector<CVector> &data_gpu, unsigned width,
                         unsigned height, DType dx, DType dy, DType dz)
{
  unsigned int N = data_gpu[0].size();
  std::vector<CVector> gradient;
  for (int i = 0; i < 6; i++)
    gradient.push_back(CVector(N));

  SymmetricGradient(data_gpu, gradient, width, height, dx, dy, dz);
  return gradient;
}

void utils::SymmetricGradient2D(const std::vector<CVector> &data_gpu,
                                std::vector<CVector> &gradient, unsigned width,
                                unsigned height, DType dx, DType dy)
{
  unsigned int N = data_gpu[0].size();
  CVector temp(N);

  // dxx
  agile::lowlevel::bdiff3(1, width, height, data_gpu[0].data(),
                          gradient[0].data(), N, false);
  if (dx != 1.0)
    agile::scale((DType)1.0 / dx, gradient[0], gradient[0]);

  // dyy
  agile::lowlevel::bdiff3(2, width, height, data_gpu[1].data(),
                          gradient[1].data(), N, false);

  if (dy != 1.0)
    agile::scale((DType)1.0 / dy, gradient[1], gradient[1]);

  // dxy
  agile::lowlevel::bdiff3(2, width, height, data_gpu[0].data(),
                          gradient[2].data(), N, false);
  agile::lowlevel::bdiff3(1, width, height, data_gpu[1].data(), temp.data(), N,
                          false);
  agile::addVector(gradient[2], temp, gradient[2]);
  agile::scale((DType)(1.0 / (2.0 * dx)), gradient[2], gradient[2]);
}

std::vector<CVector>
utils::SymmetricGradient2D(const std::vector<CVector> &data_gpu, unsigned width,
                           unsigned height, DType dx, DType dy)
{
  unsigned int N = data_gpu[0].size();
  std::vector<CVector> gradient;
  for (int i = 0; i < 3; i++)
    gradient.push_back(CVector(N));

  SymmetricGradient2D(data_gpu, gradient, width, height, dx, dy);
  return gradient;
}

void utils::SymmetricGradientNorm(const std::vector<CVector> &gradient,
                                  CVector &norm)
{
  unsigned int N = gradient[0].size();
  CVector temp(N);

  // compute norm sqrt(abs(dx).^2 + abs(dy).^2 + abs(dz).^2 + 2.0*abs(dxy).^2 +
  // 2.0*abs(dxz).^2 + 2.0*abs(dyz).^2)
  // TODO bad style
  agile::multiplyConjElementwise(gradient[0], gradient[0], norm);
  agile::multiplyConjElementwise(gradient[1], gradient[1], temp);
  agile::addVector(temp, norm, norm);
  agile::multiplyConjElementwise(gradient[2], gradient[2], temp);
  agile::addVector(temp, norm, norm);
  agile::multiplyConjElementwise(gradient[3], gradient[3], temp);
  agile::addScaledVector(norm, 2.0f, temp, norm);
  agile::multiplyConjElementwise(gradient[4], gradient[4], temp);
  agile::addScaledVector(norm, 2.0f, temp, norm);
  agile::multiplyConjElementwise(gradient[5], gradient[5], temp);
  agile::addScaledVector(norm, 2.0f, temp, norm);
  agile::sqrt(norm, norm);
}

CVector utils::SymmetricGradientNorm(const std::vector<CVector> &gradient)
{
  unsigned int N = gradient[0].size();
  CVector norm_gpu(N);
  utils::SymmetricGradientNorm(gradient, norm_gpu);
  return norm_gpu;
}

void utils::SymmetricGradientNorm2D(const std::vector<CVector> &gradient,
                                    CVector &norm)
{
  unsigned int N = gradient[0].size();
  CVector temp(N);

  // compute norm sqrt(abs(dx).^2 + abs(dy).^2 + 2.0*abs(dxy).^2)
  // TODO bad style
  agile::multiplyConjElementwise(gradient[0], gradient[0], norm);
  agile::multiplyConjElementwise(gradient[1], gradient[1], temp);
  agile::addVector(temp, norm, norm);
  agile::multiplyConjElementwise(gradient[2], gradient[2], temp);
  agile::addScaledVector(norm, 2.0f, temp, norm);
  agile::sqrt(norm, norm);
}

CVector utils::SymmetricGradientNorm2D(const std::vector<CVector> &gradient)
{
  unsigned int N = gradient[0].size();
  CVector norm_gpu(N);
  utils::SymmetricGradientNorm2D(gradient, norm_gpu);
  return norm_gpu;
};

void utils::Divergence(std::vector<CVector> &gradient, CVector &divergence,
                       unsigned width, unsigned height, unsigned frames,
                       DType dx, DType dy, DType dz)
{
  unsigned int N = width * height * frames;
  CVector temp_gpu(N);
  agile::lowlevel::diff3trans(1, width, height, gradient[0].data(),
                              divergence.data(), N, false);
  if (dx != 1.0)
    agile::scale((DType)1.0 / dx, divergence, divergence);

  agile::lowlevel::diff3trans(2, width, height, gradient[1].data(),
                              temp_gpu.data(), N, false);
  if (dy != 1.0)
    agile::scale((DType)1.0 / dy, temp_gpu, temp_gpu);

  agile::addVector(temp_gpu, divergence, divergence);
  agile::lowlevel::diff3trans(3, width, height, gradient[2].data(),
                              temp_gpu.data(), N, false);
  if (dz != 1.0)
    agile::scale((DType)1.0 / dz, temp_gpu, temp_gpu);

  agile::addVector(temp_gpu, divergence, divergence);
  agile::scale(-1.0f, divergence, divergence);
}

CVector utils::Divergence(std::vector<CVector> &gradient, unsigned width,
                          unsigned height, unsigned frames, DType dx, DType dy,
                          DType dz)
{
  unsigned int N = width * height * frames;
  CVector divergence(N);
  Divergence(gradient, divergence, width, height, frames, dx, dy, dz);
  return divergence;
}

void utils::Divergence2D(std::vector<CVector> &gradient, CVector &divergence,
                         unsigned width, unsigned height, DType dx, DType dy)
{
  unsigned int N = width * height;
  CVector temp_gpu(N);
  agile::lowlevel::diff3trans(1, width, height, gradient[0].data(),
                              divergence.data(), N, false);
  if (dx != 1.0)
    agile::scale((DType)1.0 / dx, divergence, divergence);

  agile::lowlevel::diff3trans(2, width, height, gradient[1].data(),
                              temp_gpu.data(), N, false);
  if (dy != 1.0)
    agile::scale((DType)1.0 / dy, temp_gpu, temp_gpu);

  agile::addVector(temp_gpu, divergence, divergence);
  agile::scale(-1.0f, divergence, divergence);
}

CVector utils::Divergence2D(std::vector<CVector> &gradient, unsigned width,
                            unsigned height, DType dx, DType dy)
{
  unsigned int N = width * height;
  CVector divergence(N);
  Divergence(gradient, divergence, width, height, dx, dy);
  return divergence;
}

void utils::SymmetricDivergence(std::vector<CVector> &gradient,
                                std::vector<CVector> &divergence,
                                unsigned width, unsigned height,
                                unsigned frames, DType dx, DType dy, DType dz)
{
  unsigned N = width * height * frames;
  CVector temp_gpu(N);
  // first component
  agile::lowlevel::bdiff3trans(1, width, height, gradient[0].data(),
                               divergence[0].data(), N, false);
  agile::lowlevel::bdiff3trans(2, width, height, gradient[3].data(),
                               temp_gpu.data(), N, false);
  agile::addVector(divergence[0], temp_gpu, divergence[0]);
  agile::scale((DType)1.0 / dx, divergence[0], divergence[0]);
  agile::lowlevel::bdiff3trans(3, width, height, gradient[4].data(),
                               temp_gpu.data(), N, false);
  agile::addScaledVector(divergence[0], (DType)1.0 / dz, temp_gpu,
                         divergence[0]);
  agile::scale(-1.0f, divergence[0], divergence[0]);

  // second component
  agile::lowlevel::bdiff3trans(1, width, height, gradient[3].data(),
                               divergence[1].data(), N, false);
  agile::lowlevel::bdiff3trans(2, width, height, gradient[1].data(),
                               temp_gpu.data(), N, false);
  agile::addVector(divergence[1], temp_gpu, divergence[1]);
  agile::scale((DType)1.0 / dx, divergence[1], divergence[1]);
  agile::lowlevel::bdiff3trans(3, width, height, gradient[5].data(),
                               temp_gpu.data(), N, false);
  agile::addScaledVector(divergence[1], (DType)1.0 / dz, temp_gpu,
                         divergence[1]);
  agile::scale(-1.0f, divergence[1], divergence[1]);

  // third component
  agile::lowlevel::bdiff3trans(1, width, height, gradient[4].data(),
                               divergence[2].data(), N, false);
  agile::lowlevel::bdiff3trans(2, width, height, gradient[5].data(),
                               temp_gpu.data(), N, false);
  agile::addVector(divergence[2], temp_gpu, divergence[2]);
  agile::scale((DType)1.0 / dx, divergence[2], divergence[2]);
  agile::lowlevel::bdiff3trans(3, width, height, gradient[2].data(),
                               temp_gpu.data(), N, false);
  agile::addScaledVector(divergence[2], (DType)1.0 / dz, temp_gpu,
                         divergence[2]);
  agile::scale(-1.0f, divergence[2], divergence[2]);
}

std::vector<CVector> utils::SymmetricDivergence(std::vector<CVector> &gradient,
                                                unsigned width, unsigned height,
                                                unsigned frames, DType dx,
                                                DType dy, DType dz)
{
  unsigned int N = width * height * frames;
  std::vector<CVector> divergence;
  divergence.push_back(CVector(N));
  divergence.push_back(CVector(N));
  divergence.push_back(CVector(N));

  SymmetricDivergence(gradient, divergence, width, height, frames, dx, dy, dz);
  return divergence;
}

void utils::SymmetricDivergence2D(std::vector<CVector> &gradient,
                                  std::vector<CVector> &divergence,
                                  unsigned width, unsigned height, DType dx,
                                  DType dy)
{
  unsigned N = width * height;
  CVector temp_gpu(N);
  // first component
  agile::lowlevel::bdiff3trans(1, width, height, gradient[0].data(),
                               divergence[0].data(), N, false);
  agile::lowlevel::bdiff3trans(2, width, height, gradient[2].data(),
                               temp_gpu.data(), N, false);
  agile::addVector(divergence[0], temp_gpu, divergence[0]);
  agile::scale((DType)1.0 / dx, divergence[0], divergence[0]);
  agile::scale(-1.0f, divergence[0], divergence[0]);

  // second component
  agile::lowlevel::bdiff3trans(1, width, height, gradient[2].data(),
                               divergence[1].data(), N, false);
  agile::lowlevel::bdiff3trans(2, width, height, gradient[1].data(),
                               temp_gpu.data(), N, false);
  agile::addVector(divergence[1], temp_gpu, divergence[1]);
  agile::scale((DType)1.0 / dx, divergence[1], divergence[1]);
  agile::scale(-1.0f, divergence[1], divergence[1]);
}

std::vector<CVector>
utils::SymmetricDivergence2D(std::vector<CVector> &gradient, unsigned width,
                             unsigned height, DType dx, DType dy)
{
  unsigned int N = width * height;
  std::vector<CVector> divergence;
  divergence.push_back(CVector(N));
  divergence.push_back(CVector(N));

  SymmetricDivergence2D(gradient, divergence, width, height, dx, dy);
  return divergence;
}

RType utils::TVNorm(CVector &data_gpu, unsigned width, unsigned height,
                    DType dx, DType dy, DType dt)
{
  std::vector<CVector> gradient =
      utils::Gradient(data_gpu, width, height, dx, dy, dt);

  CVector norm_gpu;
  norm_gpu = utils::GradientNorm(gradient);

  // Compute Sum
  RType tvNorm = agile::norm1(norm_gpu);

  return tvNorm;
}

RType utils::TGV2Norm(CVector &data1_gpu, std::vector<CVector> &data2_gpu,
                      RType alpha0, RType alpha1, unsigned width,
                      unsigned height, DType dx, DType dy, DType dz)
{
  std::vector<CVector> y1 =
      utils::Gradient(data1_gpu, width, height, dx, dy, dz);

  agile::subVector(y1[0], data2_gpu[0], y1[0]);
  agile::subVector(y1[1], data2_gpu[1], y1[1]);
  agile::subVector(y1[2], data2_gpu[2], y1[2]);

  std::vector<CVector> y2 =
      utils::SymmetricGradient(data2_gpu, width, height, dx, dy, dz);

  CVector n1 = utils::GradientNorm(y1);
  CVector n2 = utils::SymmetricGradientNorm(y2);
  RType norm = alpha1 * agile::norm1(n1) + alpha0 * agile::norm1(n2);
  return norm;
}

RType utils::TGV2Norm(CVector &data1_gpu, std::vector<CVector> &data2_gpu,
                      std::vector<CVector> &temp3, std::vector<CVector> &temp6,
                      RType alpha0, RType alpha1, unsigned width,
                      unsigned height, DType dx, DType dy, DType dz)
{
  utils::Gradient(data1_gpu, temp3, width, height, dx, dy, dz);

  agile::subVector(temp3[0], data2_gpu[0], temp3[0]);
  agile::subVector(temp3[1], data2_gpu[1], temp3[1]);
  agile::subVector(temp3[2], data2_gpu[2], temp3[2]);

  CVector n1 = utils::GradientNorm(temp3);
  RType norm = alpha1 * agile::norm1(n1);

  utils::SymmetricGradient(data2_gpu, temp6, width, height, dx, dy, dz);

  n1 = utils::SymmetricGradientNorm(temp6);
  norm += alpha0 * agile::norm1(n1);

  return norm;
}

RType utils::ICTGV2Norm(CVector &data1, std::vector<CVector> &data2,
                        CVector &data3, std::vector<CVector> &data4,
                        RType alpha0, RType alpha1, RType alpha, unsigned width,
                        unsigned height, DType ds, DType dt, DType ds2,
                        DType dt2)
{
  unsigned N = data1.size();
  CVector temp(N);
  agile::subVector(data1, data3, temp);

  RType n1 =
      utils::TGV2Norm(temp, data2, alpha0, alpha1, width, height, ds, ds, dt);

  RType n2 = utils::TGV2Norm(data3, data4, alpha0, alpha1, width, height, ds2,
                             ds2, dt2);

  RType denom = std::min(alpha, (RType)1.0 - alpha);
  RType norm = (alpha / denom) * (n1) + ((1.0 - alpha) / denom) * (n2);
  return norm;
}

RType utils::ICTGV2Norm(CVector &data1, std::vector<CVector> &data2,
                        CVector &data3, std::vector<CVector> &data4,
                        std::vector<CVector> &temp3,
                        std::vector<CVector> &temp6, RType alpha0, RType alpha1,
                        RType alpha, unsigned width, unsigned height, DType ds,
                        DType dt, DType ds2, DType dt2)
{
  unsigned N = data1.size();
  CVector temp(N);
  agile::subVector(data1, data3, temp);

  RType n1 = utils::TGV2Norm(temp, data2, temp3, temp6, alpha0, alpha1, width,
                             height, ds, ds, dt);

  RType n2 = utils::TGV2Norm(data3, data4, temp3, temp6, alpha0, alpha1, width,
                             height, ds2, ds2, dt2);

  RType denom = std::min(alpha, (RType)1.0 - alpha);
  RType norm = (alpha / denom) * (n1) + ((1.0 - alpha) / denom) * (n2);
  return norm;
}

void utils::DivideVectorElementwise(std::vector<CVector> &y, CVector scaleVec,
                                    unsigned vecElements)
{
  agile::max(scaleVec, CType(1.0f), scaleVec);
  for (unsigned cnt = 0; cnt < vecElements; cnt++)
    agile::divideElementwise(y[cnt], scaleVec, y[cnt]);
}

void utils::DivideVectorScaledElementwise(std::vector<CVector> &y, CVector &vec,
                                          RType scale, unsigned vecElements)
{
  agile::scale(scale, vec, vec);
  DivideVectorElementwise(y, vec, vecElements);
}

void utils::ProximalMap2D(std::vector<CVector> &y, RType scale)
{
  CVector norm_gpu = utils::GradientNorm2D(y);
  DivideVectorScaledElementwise(y, norm_gpu, scale, 2);
}

void utils::ProximalMap2DSym(std::vector<CVector> &y, RType scale)
{
  CVector norm_gpu = utils::SymmetricGradientNorm2D(y);
  DivideVectorScaledElementwise(y, norm_gpu, scale, 3);
}

void utils::ProximalMap3(std::vector<CVector> &y, RType scale)
{
  CVector norm_gpu = utils::GradientNorm(y);
  DivideVectorScaledElementwise(y, norm_gpu, scale, 3);
}

void utils::ProximalMap6(std::vector<CVector> &y, RType scale)
{
  CVector norm_gpu = utils::SymmetricGradientNorm(y);
  DivideVectorScaledElementwise(y, norm_gpu, scale, 6);
}

void utils::SumOfSquares3(std::vector<CVector> &x, CVector &sum)
{
  CVector temp(sum.size());
  for (unsigned cnt = 0; cnt < 3; cnt++)
  {
    agile::multiplyConjElementwise(x[cnt], x[cnt], temp);
    agile::addVector(sum, temp, sum);
  }
}

void utils::SumOfSquares6(std::vector<CVector> &x, CVector &sum)
{
  CVector temp(sum.size());
  for (unsigned cnt = 0; cnt < 3; cnt++)
  {
    agile::multiplyConjElementwise(x[cnt], x[cnt], temp);
    agile::addVector(sum, temp, sum);
    agile::multiplyConjElementwise(x[cnt + 3], x[cnt + 3], temp);
    agile::addScaledVector(sum, (DType)2.0, temp, sum);
  }
}

RType utils::Median(std::vector<RType> data)
{
  if (data.size() > 0)
  {

    unsigned ind = data.size() * 0.5;
    std::nth_element(data.begin(), data.begin() + ind, data.end());

    // check if length is even or odd
    if ((data.size() % 2) == 0)
    {
      RType val = data[ind];
      std::nth_element(data.begin(), data.begin() + ind - 1, data.end());
      return 0.5 * (val + data[ind - 1]);
    }
    else
      return data[ind];
  }
  else
    throw std::runtime_error("Median: Input array empty!");
}

std::string utils::GetParentDirectory(const std::string &filename)
{
  boost::filesystem::path p(filename);
  boost::filesystem::path dir = boost::filesystem::absolute(p).parent_path();
  return dir.string();
}

std::string utils::GetFilename(const std::string &filename)
{
  boost::filesystem::path p(filename);
  return p.stem().string();
}

std::string utils::GetFileExtension(const std::string &filename)
{
  boost::filesystem::path p(filename);
  return p.extension().string();
}

void utils::WriteH5File(const std::string &filename,
                        const std::string &fieldname, std::vector<size_t> dims,
                        std::vector<CType> data)
{
  ISMRMRD::Dataset d(filename.c_str(), "dataset", true);
  ISMRMRD::NDArray<CType> dataArray(dims);
  std::copy(data.begin(), data.end(), dataArray.begin());
  d.appendNDArray(fieldname, dataArray);
}

void utils::GetSubVector(CVector &full, CVector &stride, unsigned index,
                         unsigned strideLength)
{
  agile::lowlevel::get_content(full.data(), 1, strideLength, 0,
                               index * strideLength, stride.data(), 1,
                               strideLength);
}

void utils::SetSubVector(CVector &stride, CVector &full, unsigned index,
                         unsigned strideLength)
{
  agile::lowlevel::get_content(stride.data(), 1, strideLength, 0, 0,
                               full.data() + index * strideLength, 1,
                               strideLength);
}

