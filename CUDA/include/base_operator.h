#ifndef INCLUDE_BASE_OPERATOR_H_

#define INCLUDE_BASE_OPERATOR_H_

#include <vector>
#include <complex>
#include "./types.h"
#include "./utils.h"
#include "agile/gpu_vector.hpp"

/**
 * \brief Abstract MR Operator responsible for forward and backward MR
 *operations
 *
 * The forward MR operator performs the computation of the image starting from
 * the k-space domain: K^H(x) = sum(c_i* \cdot iFFT(x_i))
 *
 * The backward MR operation computes the k-space based on
 * image data: K(x)_i = FFT(x_i) * c_i
 *
 * Both operations apply the coil sensitivities accordingly.
 *
 */
class BaseOperator
{
 public:
  BaseOperator(unsigned width, unsigned height, unsigned coils,
               unsigned frames);

  virtual ~BaseOperator();

  /** \brief Forward operation: computation of coil-summation image based on
   *k-space data
   *
   * \param x_gpu k-space data (multiple coils), dims: depending on k-space
   *trajectory
   * \param sum coil summation image, dims: width * height * frames
   * \param b1_gpu coil sensitivities, dims: width * height * frames
   * */
  virtual void ForwardOperation(CVector &x_gpu, CVector &sum,
                                CVector &b1_gpu) = 0;

  /** \brief Forward operation: computation of coil-summation image based on
   *k-space data
   *
   * \param x_gpu k-space data (multiple coils), dims: depending on k-space
   *trajectory
   * \param b1_gpu coil sensitivities, dims: width * height * frames
   * \return coil summation image, dims: width * height * frames
   * */
  virtual CVector ForwardOperation(CVector &x_gpu, CVector &b1_gpu) = 0;

  /** \brief Backward operation: computation of coil-wise k-space based on image
   *data
   *
   * \param x_gpu image data, dims: width * height * frames
   * \param z_gpu k-space data (multiple coils)
   * \param b1_gpu coil sensitivities, dims: width * height * frames
   * */
  virtual void BackwardOperation(CVector &x_gpu, CVector &z_gpu,
                                 CVector &b1_gpu) = 0;

  /** \brief Backward operation: computation of coil-wise k-space based on image
   *data
   *
   * \param x_gpu image data, dims: width * height * frames
   * \param b1_gpu coil sensitivities, dims: width * height * frames
   * \return k-space data (multiple coils)
   * */
  virtual CVector BackwardOperation(CVector &x_gpu, CVector &b1_gpu) = 0;

  /** \brief Initial lambda parameter computation. Depends on operator type
   * (Cartesian, Non-Cartesian).*/
  virtual RType AdaptLambda(RType k, RType d) = 0;

 protected:
  /** \brief Image dimension width */
  unsigned int width;
  /** \brief Image dimension height */
  unsigned int height;
  /** \brief Image dimension amount of coils */
  unsigned int coils;
  /** \brief Image dimension amount of frames */
  unsigned int frames;

 private:
};

#endif  // INCLUDE_BASE_OPERATOR_H_
