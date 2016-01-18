#ifndef INCLUDE_PD_RECON_H_

#define INCLUDE_PD_RECON_H_

#include <vector>
#include <complex>
#include <cstdio>
#include <cstdarg>
#include "./types.h"
#include "./utils.h"
#include "./base_operator.h"
#include "agile/calc/fft.hpp"
#include "agile/gpu_vector.hpp"

/**
 * \brief Parameter collection for adapt lambda step
 */
typedef struct AdaptLambdaParams
{
  AdaptLambdaParams()
  {
  }
  AdaptLambdaParams(DType k, DType d)
    : k(k), d(d)
  {
  }

  bool adaptLambda;
  DType k;
  DType d;
} AdaptLambdaParams;

/** \brief Basic parameter struct used in primal-dual (PD) reconstructions. */
typedef struct PDParams
{
  /** \brief Maximum number of iterations for main iteration. */
  unsigned maxIt;

  /** \brief Step size for difference computation in spatial direction */
  RType ds;

  /** \brief Step size for difference computation in temporal direction */
  RType dt;

  /** \brief Dual variable step size. */
  RType sigma;

  /** \brief Primal variable step size. */
  RType tau;

  /** \brief Ratio between primal and dual step-sizes. */
  RType sigmaTauRatio;

  /** \brief spatio-temporal weight.*/
  RType timeSpaceWeight;

  /** \brief Regularization parameter */
  RType lambda;

  /** \brief Parameters for adaptation of lambda */
  AdaptLambdaParams adaptLambdaParams;
} PDParams;

typedef void (*ResultExportCallback)(const char* outputDir, const char* filename, CVector& result);

/** \brief Base Primal-dual reconstruction class.
 *
 * Holds the basic functions used in every PD reconstruction.
 *
 * \see TV
 * \see TGV2
 * \see ICTGV2
 * \see CoilConstruction
 */
class PDRecon
{
 public:
  PDRecon(unsigned width, unsigned height, unsigned coils, unsigned frames,
          BaseOperator *mrOp);
  virtual ~PDRecon();

  /** \brief Redirects the adapt lambda step to the corresponding MR operator.
   */
  RType AdaptLambda(RType k, RType d);

  /** \brief Adapt stepsizes (sigma, tau) by checking the convergence condition
   * based on nKx and nx.
   */
  void AdaptStepSize(RType nKx, RType nx);

  /** \brief Compute the weights ds, dt based on the timeSpaceWeight */
  void ComputeTimeSpaceWeights(RType timeSpaceWeight, RType &ds, RType &dt);

  /** \brief Return PDParams reference (abstract method). */
  virtual PDParams &GetParams() = 0;

  /** \brief Perform the iterative reconstruction.
   *
   * \param data_gpu the measured k-space data
   * \param x reconstructed image, dims: width * height * frames
   * \param b1_gpu coil sensitivies
   */
  virtual void IterativeReconstruction(CVector &data_gpu, CVector &x,
                                       CVector &b1_gpu);

  /** 
   * \brief Method to allow file export of additional result data
   *
   * May also be used to export debug information if necessary.
   */ 
  virtual void ExportAdditionalResults(const char* outputDir, ResultExportCallback callback) = 0;

  /** \brief Enable console output
   *
   * \param verbose true, if console output is active
   */
  void SetVerbose(bool verbose);

  /** \brief Enable PD-Gap calculation every "debugstep" iterations  */
  void SetDebug(bool debug,int debugstep);

 protected:
  /** \brief Image dimension width */
  unsigned int width;
  /** \brief Image dimension height */
  unsigned int height;
  /** \brief Image dimension amount of coils */
  unsigned int coils;
  /** \brief Image dimension amount of frames */
  unsigned int frames;

  /** \brief Pointer to MR operator used in forward/backward operations. */
  BaseOperator *mrOp;

  /** \brief Enable verbose console output. */
  bool verbose;

  /** \brief Enable debug console output,i.e. PD GAP */
  bool debug;
  int debugstep;


  /** \brief Log message to console output */
  virtual void Log(const char *format, ...);

 private:

};

#endif  // INCLUDE_PD_RECON_H_
