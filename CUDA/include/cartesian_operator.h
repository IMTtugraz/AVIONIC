#ifndef INCLUDE_CARTESIAN_OPERATOR_H_

#define INCLUDE_CARTESIAN_OPERATOR_H_

#include "./base_operator.h"
#include "agile/calc/fft.hpp"
#include "cufft.h"

/**
 * \brief Cartesian MR operator utilizing simple masked FFT functionality
 *
 * The mask defines the sub-sampling pattern for each frame.
 *
 */
class CartesianOperator : public BaseOperator
{
 public:
  CartesianOperator(unsigned width, unsigned height, unsigned coils,
                    unsigned frames);
  CartesianOperator(unsigned width, unsigned height, unsigned coils,
                    unsigned frames, RVector &mask, bool centered);
  CartesianOperator(unsigned width, unsigned height, unsigned coils,
                    unsigned frames, RVector &mask);
  CartesianOperator(unsigned width, unsigned height, unsigned coils,
                    unsigned frames, bool centered);
/*
  //-----------------------------------------------------------------------------------
  // 3D Operators
  //-----------------------------------------------------------------------------------
  CartesianOperator(unsigned width, unsigned height, unsigned depth, 
                    unsigned coils, bool is3D);
  CartesianOperator(unsigned width, unsigned height, unsigned depth,
                    unsigned coils, RVector &mask, bool centered,  bool is3D);
  CartesianOperator(unsigned width, unsigned height, unsigned depth,
                    unsigned coils, RVector &mask,  bool is3D);
  CartesianOperator(unsigned width, unsigned height, unsigned depth, 
                    unsigned coils, bool centered, bool is3D);
  //-----------------------------------------------------------------------------------
  */
  virtual ~CartesianOperator();

  /** \brief Cartesian forward operation: computation of coil-summation
   *image based on
   *k-space data
   *
   * \param x_gpu k-space data (multiple coils), dims: width * height * coils *
   *frames
   * \param sum coil summation image, dims: width * height * frames
   * \param b1_gpu coil sensitivities, dims: width * height * frames
   * */
  void ForwardOperation(CVector &x_gpu, CVector &sum, CVector &b1_gpu);

  /** \brief Cartesian Forward operation: computation of coil-summation image
   *based on
   *k-space data
   *
   * \param x_gpu k-space data (multiple coils), dims: width * height * coils *
   *frames
   * \param b1_gpu coil sensitivities, dims: width * height * frames
   * \return coil summation image, dims: width * height * frames
   * */
  CVector ForwardOperation(CVector &x_gpu, CVector &b1_gpu);

  
  /** \brief Cartesian forward operation of 3D data: computation of coil-summation
   *image based on
   *k-space data
   *
   * \param x_gpu k-space data (multiple coils), dims: width * height * depth * coils
   * \param sum coil summation image, dims: width * height * depth
   * \param b1_gpu coil sensitivities, dims: width * height * depth
   * */
  //void ForwardOperation3D(CVector &x_gpu, CVector &sum, CVector &b1_gpu);

  /** \brief Cartesian Forward operation of 3D data: computation of coil-summation image
   *based on
   *k-space data
   *
   * \param x_gpu k-space data (multiple coils), dims: width * height * depth * coils 
   * \param b1_gpu coil sensitivities, dims: width * height * depth
   * \return coil summation image, dims: width * height * depth
   * */

  //CVector ForwardOperation3D(CVector &x_gpu, CVector &b1_gpu);
  /** \brief Cartesian Backward operation: computation of coil-wise k-space
   *based on image
   *data
   *
   * \param x_gpu image data, dims: width * height * frames
   * \param z_gpu k-space data (multiple coils), dims: width * height * coils *
   *frames
   * \param b1_gpu coil sensitivities, dims: width * height * frames
   * */
  void BackwardOperation(CVector &x_gpu, CVector &z_gpu, CVector &b1_gpu);

  /** \brief Cartesian Backward operation: computation of coil-wise k-space
   *based on image
   *data
   *
   * \param x_gpu image data, dims: width * height * frames
   * \param b1_gpu coil sensitivities, dims: width * height * frames
   * \return z_gpu k-space data (multiple coils), dims: width * height * coils *
   *frames
   * */
  CVector BackwardOperation(CVector &x_gpu, CVector &b1_gpu);

  /** \brief Cartesian Backward operation of 3D data: computation of coil-wise k-space
   *based on image
   *data
   *
   * \param x_gpu image data, dims: width * height * depth
   * \param z_gpu k-space data (multiple coils), dims: width * height depth * coils 
   * \param b1_gpu coil sensitivities, dims: width * height * depth
   * */
  //void BackwardOperation3D(CVector &x_gpu, CVector &z_gpu, CVector &b1_gpu);

  /** \brief Cartesian Backward operation of 3D data: computation of coil-wise k-space
   *based on image
   *data
   *
   * \param x_gpu image data, dims: width * height * depth
   * \param b1_gpu coil sensitivities, dims: width * height * depth
   * \return z_gpu k-space data (multiple coils), dims: width * height * depth * coils
   * */
  //CVector BackwardOperation3D(CVector &x_gpu, CVector &b1_gpu);

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
  agile::FFT<CType> *fftOp;

  /** \brief Needed for computation of 3D Forward/Backward FFT */ 
 // const std::complex<float>* in_data;
 // std::complex<float>* out_data;

  /** \brief cufft Handle used in 3D Forward/Backward operations. */ 
  //cufftHandle fftplan3d;

 private:
  /** \brief Initialize FFT operator */
  void Init();

  /** \brief Initialize 3D-FFT operator */
  //void Init3D();
};

#endif  // INCLUDE_CARTESIAN_OPERATOR_H_
