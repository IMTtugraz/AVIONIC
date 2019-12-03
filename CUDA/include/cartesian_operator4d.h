#ifndef INCLUDE_CARTESIAN_OPERATOR4D_H_

#define INCLUDE_CARTESIAN_OPERATOR4D_H_

#include "./base_operator.h"
#include "agile/calc/fft.hpp"
#include "cufft.h"

/**
 * \brief Cartesian MR operator for 4D dat utilizing simple masked FFT functionality
 *
 * The mask defines the sub-sampling pattern for each frame.
 *
 */
class CartesianOperator4D : public BaseOperator
{
 public:
 //-----------------------------------------------------------------------------------
  // 3D Operators
  //-----------------------------------------------------------------------------------
  CartesianOperator4D(unsigned width, unsigned height, unsigned depth,
                    unsigned coils, unsigned frames, RVector &mask);
  CartesianOperator4D(unsigned width, unsigned height, unsigned depth,
                    unsigned coils, unsigned frames, RVector &mask, bool centered);
  CartesianOperator4D(unsigned width, unsigned height, unsigned depth, 
                    unsigned coils, unsigned frames, bool centered);
  CartesianOperator4D(unsigned width, unsigned height, unsigned depth, 
                    unsigned coils, unsigned frames);
  //-----------------------------------------------------------------------------------
  
  virtual ~CartesianOperator4D();
  
    /** \brief Cartesian forward operation of 3D data: computation of coil-summation
   *image based on
   *k-space data
   *
   * \param x_gpu k-space data (multiple coils), dims: width * height * depth * coils *frames
   * \param sum coil summation image, dims: width * height * depth
   * \param b1_gpu coil sensitivities, dims: width * height * depth * coils
   * */
  void ForwardOperation(CVector &x_gpu, CVector &sum, CVector &b1_gpu);

  /** \brief Cartesian Forward operation of 3D data: computation of coil-summation image
   *based on
   *k-space data
   *
   * \param x_gpu k-space data (multiple coils), dims: width * height * depth * coils * frames
   * \param b1_gpu coil sensitivities, dims: width * height * depth * coils
   * \return coil summation image, dims: width * height * depth
   * */
  CVector ForwardOperation(CVector &x_gpu, CVector &b1_gpu);
  /** \brief Forward operation: computation of coil-summation image based on
   *k-space data
   *
   * \param x_gpu k-space data (multiple coils), dims: depending on k-space
   *trajectory
   * \param sum coil summation image, dims: width * height * frames
   * \param b1_gpu coil sensitivities, dims: width * height * frames
   * */
  void ForwardOperation(CVector &x_gpu, CVector &sum,
                                CVector &b1_gpu, CVector &z_gpu)
  {
	  std::cerr<<"Not yet implemented!"<<std::endl;
  } 
  
  /** \brief Cartesian Backward operation of 3D data: computation of coil-wise k-space
   *based on image
   *data
   *
   * \param x_gpu image data, dims: width * height * depth
   * \param z_gpu k-space data (multiple coils), dims: width * height * depth * coils * frames
   * \param b1_gpu coil sensitivities, dims: width * height * depth * coils
   * */
  void BackwardOperation(CVector &x_gpu, CVector &z_gpu, CVector &b1_gpu);
  
  /** \brief Cartesian Backward operation of 3D data: computation of coil-wise k-space
   *based on image
   *data
   *
   * \param x_gpu image data, dims: width * height * depth
   * \param b1_gpu coil sensitivities, dims: width * height * depth * coils
   * \return z_gpu k-space data (multiple coils), dims: width * height * depth * coils * frames
   * */
  CVector BackwardOperation(CVector &x_gpu, CVector &b1_gpu);
  
   /** \brief Backward operation: computation of coil-wise k-space based on image
   *data
   *
   * \param x_gpu image data, dims: width * height * frames
   * \param z_gpu k-space data (multiple coils)
   * \param b1_gpu coil sensitivities, dims: width * height * frames
   * */
  void BackwardOperation(CVector &x_gpu, CVector &z_gpu,
                                 CVector &b1_gpu, CVector &x_hat_gpu) 
  {
	  std::cerr<<"Not yet implemented!"<<std::endl;
  }

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
