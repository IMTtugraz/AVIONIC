#ifndef INCLUDE_H1_RECON_H_

#define INCLUDE_H1_RECON_H_

#include <vector>
#include <complex>
#include <cstdio>
#include <cstdarg>
#include "./types.h"
#include "./utils.h"
#include "./cg_forward_operation_BS.h"
#include "agile/calc/fft.hpp"
#include "agile/gpu_vector.hpp"
#include "agile/operator/cg.hpp"



typedef agile::GPUCommunicator<unsigned, CType, CType> communicator_type;

typedef ForwardOperation<communicator_type, CVector> forward_type;




typedef void (*ResultExportCallback)(const char* outputDir, const char* filename, CVector& result);

/** \brief H1 reconstruction class.
 *
 * Holds the basic functions used in every CG reconstruction.
 *
 */
class H1Recon
{
 public:
  H1Recon(unsigned width, unsigned height, unsigned depth, unsigned coils,
          H1Params &params, CartesianOperator3D* mrOp, communicator_type &com);

  H1Recon(unsigned width, unsigned height, unsigned depth, unsigned coils,
           CartesianOperator3D *mrOp, communicator_type &com);
 
  ~H1Recon();

  /** \brief Return PDParams reference (abstract method). */
  H1Params &GetParams() ;

  /** \brief Perform the iterative reconstruction.
   *
   * \param data_gpu the measured k-space data
   * \param x reconstructed image, dims: width * height * frames
   * \param b1_gpu coil sensitivies
   */
  void IterativeReconstruction(CVector &data_gpu, CVector &x,
                                       CVector &b1_gpu);

  /** 
   * \brief Method to allow file export of additional result data
   *
   * May also be used to export debug information if necessary.
   */ 
  //void ExportAdditionalResults(const char* outputDir, ResultExportCallback callback);

  /** \brief Enable console output
   *
   * \param verbose true, if console output is active
   */
  void SetVerbose(bool verbose);

 protected:
  /** \brief Image dimension width */
  unsigned int width;
  /** \brief Image dimension height */
  unsigned int height;
  /** \brief Image dimension depth */
  unsigned int depth; 
  /** \brief Image dimension amount of coils */
  unsigned int coils;
  /** \brief Image dimension amount of frames */
  unsigned int frames;

  /** \brief Pointer to MR operator used in forward/backward operations. */
  CartesianOperator3D *mrOp;


  /** \brief Enable verbose console output. */
  bool verbose;


 private:
  H1Params params;
  communicator_type com;

  void InitParams();

};

#endif  // INCLUDE_H1_RECON_H_
