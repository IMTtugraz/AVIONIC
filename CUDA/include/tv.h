#ifndef INCLUDE_TV_H_

#define INCLUDE_TV_H_

#include <vector>
#include <complex>
#include "./types.h"
#include "./utils.h"
#include "./pd_recon.h"
#include "agile/calc/fft.hpp"
#include "agile/gpu_vector.hpp"

/** \brief Parameter struct used in TV reconstruction. */
typedef struct TVParams : public PDParams
{
} TVParams;

/** \brief TV regularized iterative reconstruction
 *
 * The necessary MR Operator (Cartesian, Non-Cartesian)
 * used for forward/backward operation is
 * passed by the constructor.
 *
 */
class TV : public PDRecon
{
 public:
  TV(unsigned width, unsigned height, unsigned coils, unsigned frames,
     BaseOperator *mrOp);

  TV(unsigned width, unsigned height, unsigned coils, unsigned frames,
     TVParams &params, BaseOperator *mrOp);

  virtual ~TV();

  /** \brief Test adjointness of forward and backward operation, i.e. 
   * evaluates <K^H v, u> = <v, Ku> on random vectors u,v.
   */
  void TestAdjointness(CVector &b1);


  /** \brief Adapt stepsizes (sigma, tau) by checking the convergence condition
   * based on nKx and nx.
   */
  void AdaptStepSize(CVector &extDiff, CVector &b1);

  /** \brief Perform the iterative reconstruction.
   *
   * \param data_gpu the measured k-space data
   * \param x reconstructed image, dims: width * height * frames
   * \param b1_gpu coil sensitivies
   */
  void IterativeReconstruction(CVector &data_gpu, CVector &x, CVector &b1_gpu);

  void ExportAdditionalResults(const char *outputDir,
                               ResultExportCallback callback);

  /** \brief Return reference to TVParams.*/
  PDParams &GetParams();

  /** \brief G* Computation, needed in ComputePDGap function. */
  RType ComputeGStar(CVector &x, std::vector<CVector> &y, CVector &z,
                     CVector &data_gpu, CVector &b1_gpu);

  /** \brief Computation of the primal-dual gap
   *
   */
  RType ComputePDGap(CVector &x, std::vector<CVector> &y, CVector &z,
                     CVector &data_gpu, CVector &b1_gpu);

 private:
  TVParams params;

  void InitParams();
  void InitLambda(bool adaptLambda);

  std::vector<CType> pdGapExport;

  void InitTempVectors();

  // Temp vectors
  CVector imgTemp;
  CVector divTemp;
  CVector zTemp;
};

#endif  // INCLUDE_TV_H_
