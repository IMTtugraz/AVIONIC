#ifndef INCLUDE_CARTESIAN_OPERATOR3D_H_

#define INCLUDE_CARTESIAN_OPERATOR3D_H_

#include "./base_operator.h"
#include "agile/calc/fft.hpp"
#include "cufft.h"

/**
 * \brief Cartesian MR operator utilizing simple masked FFT functionality
 *
 * The mask defines the sub-sampling pattern for each frame.
 *
 */
class CartesianOperator3D : public BaseOperator
{
 public:
 //-----------------------------------------------------------------------------------
  // 3D Operators
  //-----------------------------------------------------------------------------------
  CartesianOperator3D(unsigned width, unsigned height, unsigned depth, 
                    unsigned coils);
  CartesianOperator3D(unsigned width, unsigned height, unsigned depth,
                    unsigned coils, RVector &mask, bool centered);
  CartesianOperator3D(unsigned width, unsigned height, unsigned depth,
                    unsigned coils, RVector &mask);
  CartesianOperator3D(unsigned width, unsigned height, unsigned depth, 
                    unsigned coils, bool centered);
  //-----------------------------------------------------------------------------------
  
  virtual ~CartesianOperator3D();

  /** \brief Cartesian forward operation of 3D data: computation of coil-summation
   *image based on
   *k-space data
   *
   * \param x_gpu k-space data (multiple coils), dims: width * height * depth * coils
   * \param sum coil summation image, dims: width * height * depth
   * \param b1_gpu coil sensitivities, dims: width * height * depth
   * */
  void ForwardOperation(CVector &x_gpu, CVector &sum, CVector &b1_gpu);

  /** \brief Cartesian Forward operation of 3D data: computation of coil-summation image
   *based on
   *k-space data
   *
   * \param x_gpu k-space data (multiple coils), dims: width * height * depth * coils 
   * \param b1_gpu coil sensitivities, dims: width * height * depth
   * \return coil summation image, dims: width * height * depth
   * */

  CVector ForwardOperation(CVector &x_gpu, CVector &b1_gpu);
  /** \brief Cartesian Backward operation of 3D data: computation of coil-wise k-space
   *based on image
   *data
   *
   * \param x_gpu image data, dims: width * height * depth
   * \param z_gpu k-space data (multiple coils), dims: width * height depth * coils 
   * \param b1_gpu coil sensitivities, dims: width * height * depth
   * */
  void BackwardOperation(CVector &x_gpu, CVector &z_gpu, CVector &b1_gpu);

  /** \brief Cartesian Backward operation of 3D data: computation of coil-wise k-space
   *based on image
   *data
   *
   * \param x_gpu image data, dims: width * height * depth
   * \param b1_gpu coil sensitivities, dims: width * height * depth
   * \return z_gpu k-space data (multiple coils), dims: width * height * depth * coils
   * */
  CVector BackwardOperation(CVector &x_gpu, CVector &b1_gpu);

  RType AdaptLambda(RType k, RType d);

  /** \brief Determines whether the centered (shifted) or non-centered FFT has
   * to be applied. */
  bool centered;

  /** \brief k-Space pattern used for undersampling. Dimensions: width * height
   ** frames
   *
   * Is automatically applied before/after the FFT/iFFT operations
   * respectively.
   */
  RVector &mask;

 /** \brief FFT Operator used in Forward/Backward operations. */
 // agile::FFT<CType> *fftOp;

  /** \brief Needed for computation of 3D Forward/Backward FFT */ 
  const std::complex<float>* in_data;
  std::complex<float>* out_data;

  /** \brief cufft Result handle */  
  cufftResult cres;

  /** \brief cufft Handle used in 3D Forward/Backward operations. */ 
  cufftHandle fftplan3d;

 private:
  /** \brief Initialize 3D-FFT operator */
  void Init();
};

#endif  // INCLUDE_CARTESIAN_OPERATOR_H_
