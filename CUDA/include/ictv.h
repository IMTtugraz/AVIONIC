#ifndef INCLUDE_ICTV_H_

#define INCLUDE_ICTV_H_

#include "./pd_recon.h"

/** \brief Parameter struct used in ICTV reconstruction. */
typedef struct ICTVParams : public PDParams
{
  /** \brief TV-norm weight */
  RType alpha1;
  /** \brief Trade-off between first and second TV functional */
  RType alpha;
  /** \brief spatio-temporal weight.*/
  RType timeSpaceWeight2;
  /** \brief Step size for difference computation in second x-spatial direction */
  RType dx2; 
  /** \brief Step size for difference computation in second y-spatial direction */
  RType dy2;
  /** \brief Step size for difference computation in second temporal direction  */
  RType dt2;

} ICTVParams;

/** \brief ICTV regularized iterative reconstruction
 *
 * The necessary MR Operator (Cartesian, Non-Cartesian)
 * used for forward/backward operation is
 * passed by the constructor.
 */
class ICTV : public PDRecon
{
 public:
  ICTV(unsigned width, unsigned height, unsigned coils, unsigned frames,
         BaseOperator *mrOp);

  ICTV(unsigned width, unsigned height, unsigned coils, unsigned frames,
         ICTVParams &params, BaseOperator *mrOp);

  virtual ~ICTV();

  /** \brief Return reference to the ICTVParams.*/
  PDParams &GetParams();

  /** \brief Test adjointness of forward and backward operation, i.e. 
   * evaluates <K^H v, u> = <v, Ku> on random vectors u,v.
   */
  void TestAdjointness(CVector &b1);


  /** \brief Adapt stepsizes (sigma, tau) by checking the convergence condition
   * based on nKx and nx.
   */
  void AdaptStepSize(CVector &extDiff1, CVector &extDiff3, CVector &b1);

  /** \brief Compute Datafidelity
   */
  RType ComputeDataFidelity(CVector &x1, CVector &data_gpu, CVector &b1_gpu);

  /** \brief G* Computation, needed in ComputePDGap function. */
  RType ComputeGStar(CVector &x1, std::vector<CVector> &y1,
                     std::vector<CVector> &y3,
                     CVector &z, CVector &data_gpu,
                     CVector &b1_gpu);

  /** \brief Computation of the primal-dual gap
   *
   */
  RType ComputePDGap(CVector &x1, CVector &x3,
                     std::vector<CVector> &y1,
                     std::vector<CVector> &y3,
                     CVector &z, CVector &data_gpu, CVector &b1_gpu);

  /** \brief Perform the iterative reconstruction.
   *
   * \param data_gpu the measured k-space data
   * \param x reconstructed image, dims: width * height * frames
   * \param b1_gpu coil sensitivies
   */
  void IterativeReconstruction(CVector &data_gpu, CVector &x, CVector &b1_gpu);

  void ExportAdditionalResults(const char* outputDir, ResultExportCallback callback);

 private:
  ICTVParams params;
  void InitParams();
  void InitLambda(bool adaptLambda);
  void InitTempVectors();
  void InitPrimalVectors(unsigned N);
  void InitDualVectors(unsigned N);

  std::vector<CType> pdGapExport;
  std::vector<CType> dataFidelityExport;
  std::vector<CType> ictvNormExport;
 
  RType datafidelity;

  CVector imgTemp;
  CVector zTemp;
  CVector div1Temp;
  CVector div3Temp;
  std::vector<CVector> div2Temp;
  std::vector<CVector> y2Temp;
  std::vector<CVector> y4Temp;

  // primal vectors
  CVector ext1;
  CVector x1_old;

  CVector x3;
  CVector ext3;
  CVector x3_old;

  // dual vectors
  std::vector<CVector> y1;
  std::vector<CVector> y3;
};

#endif  // INCLUDE_ICTV_H_
