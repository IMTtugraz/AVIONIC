#ifndef INCLUDE_TGV2_H_

#define INCLUDE_TGV2_H_

#include "./pd_recon.h"

/** \brief Parameter struct used in TGV reconstruction. */
typedef struct TGV2Params : public PDParams
{
  /** \brief TGV-norm weight, trade-off between first and second derivative */
  RType alpha0;
  /** \brief TGV-norm weight, trade-off between first and second derivative */
  RType alpha1;
} TGV2Params;

/** \brief TGV2 regularized iterative reconstruction
 *
 * The necessary MR Operator (Cartesian, Non-Cartesian)
 * used for forward/backward operation is
 * passed by the constructor.
 */
class TGV2 : public PDRecon
{
 public:
  TGV2(unsigned width, unsigned height, unsigned coils, unsigned frames,
       BaseOperator *mrOp);

  TGV2(unsigned width, unsigned height, unsigned coils, unsigned frames,
       TGV2Params &params, BaseOperator *mrOp);

  virtual ~TGV2();

  /** \brief Return reference to the TGV2Params.*/
  PDParams &GetParams();

  /** \brief Adapt stepsizes (sigma, tau) by checking the convergence condition
   * based on nKx and nx.
   */
  void AdaptStepSize(CVector &extDiff1, std::vector<CVector> &extDiff2,
                     CVector &b1);

  /** \brief G* Computation, needed in ComputePDGap function. */
  RType ComputeGStar(CVector &x, std::vector<CVector> &y1,
                     std::vector<CVector> &y2, CVector &z, CVector &data_gpu,
                     CVector &b1_gpu);

  /** \brief Computation of the primal-dual gap
   *
   */
  RType ComputePDGap(CVector &x1, std::vector<CVector> &x2,
                     std::vector<CVector> &y1, std::vector<CVector> &y2,
                     CVector &z, CVector &data_gpu, CVector &b1_gpu);

  /** \brief Perform the iterative reconstruction.
   *
   * \param data_gpu the measured k-space data
   * \param x reconstructed image, dims: width * height * frames
   * \param b1_gpu coil sensitivies
   */
  void IterativeReconstruction(CVector &data_gpu, CVector &x, CVector &b1_gpu);

  void ExportAdditionalResults(const char *outputDir,
                               ResultExportCallback callback);

 private:
  TGV2Params params;
  void InitParams();
  void InitLambda(bool adaptLambda);

  std::vector<CType> pdGapExport;

  void InitTempVectors();

  // Temp vectors
  CVector imgTemp;
  CVector div1Temp;
  CVector zTemp;

  std::vector<CVector> div2Temp;
  std::vector<CVector> y1Temp;
  std::vector<CVector> y2Temp;
};

#endif  // INCLUDE_TGV2_H_
