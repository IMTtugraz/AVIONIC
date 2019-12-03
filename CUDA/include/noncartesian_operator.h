#ifndef INCLUDE_NONCARTESIAN_OPERATOR_H_

#define INCLUDE_NONCARTESIAN_OPERATOR_H_

#include "./base_operator.h"
#include "gpuNUFFT_operator_factory.hpp"

/**
 * \brief Radial non-Cartesian MR Operator using gpuNUFFT.
 *
 * This operator can be used in combination with radial (e.g. stack-of-stars)
 * trajectories. Each frame is reconstructed using its own
 * gpuNUFFT operator.
 *
 */
class NoncartesianOperator : public BaseOperator
{
 public:
  NoncartesianOperator(unsigned width, unsigned height, unsigned coils,
                       unsigned frames, unsigned nSpokes, unsigned nFE,
                       unsigned spokesPerFrame, RVector &kTraj, RVector &dens,
                       CVector &sens, DType kernelWidth = 3.0,
                       DType sectorWidth = 8, DType osf = 2.0);

  NoncartesianOperator(unsigned width, unsigned height, unsigned coils,
                       unsigned frames, unsigned nSpokes, unsigned nFE,
                       unsigned spokesPerFrame, RVector &kTraj, RVector &dens,
                       DType kernelWidth = 3.0, DType sectorWidth = 8,
                       DType osf = 2.0);

  virtual ~NoncartesianOperator();

  /** \brief Non-Cartesian forward operation: computation of coil-summation
   *image based on k-space data
   *
   * \param x_gpu k-space data (multiple coils), dims: nSpokes * nFE * coils *
   *frames
   * \param sum coil summation image, dims: width * height * frames
   * \param b1_gpu coil sensitivities, dims: width * height * frames
   * */
  void ForwardOperation(CVector &x_gpu, CVector &sum, CVector &b1_gpu);

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

  /** \brief Non-Cartesian Forward operation: computation of coil-summation
   *image based on k-space data
   *
   * \param x_gpu k-space data (multiple coils), dims: nSpokes * nFE * coils *
   *frames
   * \param b1_gpu coil sensitivities, dims: width * height * frames
   * \return coil summation image, dims: width * height * frames
   * */
  CVector ForwardOperation(CVector &x_gpu, CVector &b1_gpu);

    /** \brief Non-Cartesian Backward operation: computation of coil-wise k-space
   *based on image data
   *
   * \param x_gpu image data, dims: width * height * frames
   * \param z_gpu k-space data (multiple coils), dims: nSpokes * nFE * coils *
   *frames
   * \param b1_gpu coil sensitivities, dims: width * height * frames
   * */
  void BackwardOperation(CVector &x_gpu, CVector &z_gpu, CVector &b1_gpu);

  /** \brief Non-Cartesian Backward operation: computation of coil-wise k-space
   *based on image data
   *
   * \param x_gpu image data, dims: width * height * frames
   * \param b1_gpu coil sensitivities, dims: width * height * frames
   * \return z_gpu k-space data (multiple coils), dims: nSpokes * nFE * coils *
   *frames
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

  /** \brief Array of gpuNUFFT operators.
   * Since each trajectory differs from frame to frame, it is necessary to
   * create multiple gpuNUFFT operators.
   * */
  std::vector<gpuNUFFT::GpuNUFFTOperator *> gpuNUFFTOps;

  /** \brief K-space trajectory data vector. */
  RVector &kTraj;
  /** \brief K-space trajectory density compensation vector. */
  RVector &dens;
  /** \brief Coil sensitivity data vector. */
  CVector &sens;

  /** \brief Total number of spokes (over all frames) */
  unsigned nSpokes;
  /** \brief Total number of read-out steps  (frequency encoding) */
  unsigned nFE;
  /** \brief Number of spokes per frame */
  unsigned spokesPerFrame;

 private:
  /** \brief Initialization of gpuNUFFT operator */
  void Init();

  /** \brief Number of samples per frame, i.e. nFE * spokesPerFrame */
  unsigned int nSamplesPerFrame;

  std::vector<DType> kTrajHost;
  /** \brief K-space trajectory data as gpuNUFFT compatible array. */
  gpuNUFFT::Array<DType> kTrajData;

  std::vector<DType> densHost;
  /** \brief K-space density data as gpuNUFFT compatible array. */
  gpuNUFFT::Array<DType> densData;

  /** \brief Image dimensions used for gpuNUFFT */
  gpuNUFFT::Dimensions imgDims;

  std::vector<DType2> sensHost;
  /** \brief Coil sensitivity data as gpuNUFFT compatible array. */
  gpuNUFFT::Array<DType2> sensData;

  DType kernelWidth;
  DType sectorWidth;
  DType osf;
};

#endif  // INCLUDE_NONCARTESIAN_OPERATOR_H_
