#ifndef INCLUDE_ICTGV2_H_

#define INCLUDE_ICTGV2_H_

#include "./pd_recon.h"
#include "./tgv2.h"

/** \brief Parameter struct used in ICTGV2 reconstruction. */
typedef struct ICTGV2Params : public TGV2Params
{
  /** \brief Trade-off between first and second TGV functional */
  RType alpha;
  /** \brief spatio-temporal weight.*/
  RType timeSpaceWeight2;
  /** \brief Step size for difference computation in second x-spatial direction */
  RType dx2; 
  /** \brief Step size for difference computation in second y-spatial direction */
  RType dy2;
  /** \brief Step size for difference computation in second temporal direction  */
  RType dt2;
} ICTGV2Params;

/** \brief ICTGV2 regularized iterative reconstruction
 *
 * The necessary MR Operator (Cartesian, Non-Cartesian)
 * used for forward/backward operation is
 * passed by the constructor.
 */
class ICTGV2 : public PDRecon
{
 public:
  ICTGV2(unsigned width, unsigned height, unsigned coils, unsigned frames,
         BaseOperator *mrOp);

  ICTGV2(unsigned width, unsigned height, unsigned coils, unsigned frames,
         ICTGV2Params &params, BaseOperator *mrOp);

  virtual ~ICTGV2();

  /** \brief Return reference to the ICTGV2Params.*/
  PDParams &GetParams();

  /** \brief Test adjointness of forward and backward operation, i.e. 
   * evaluates <K^H v, u> = <v, Ku> on random vectors u,v.
   */
  void TestAdjointness(CVector &b1);


  /** \brief Adapt stepsizes (sigma, tau) by checking the convergence condition
   * based on nKx and nx.
   */
  void AdaptStepSize(CVector &extDiff1, std::vector<CVector> &extDiff2,
                     CVector &extDiff3, std::vector<CVector> &extDiff4,
                     CVector &b1);

  /** \brief Compute Datafidelity
   */
  RType ComputeDataFidelity(CVector &x1, CVector &data_gpu, CVector &b1_gpu);

  /** \brief G* Computation, needed in ComputePDGap function. */
  RType ComputeGStar(CVector &x1, std::vector<CVector> &y1,
                     std::vector<CVector> &y2, std::vector<CVector> &y3,
                     std::vector<CVector> &y4, CVector &z, CVector &data_gpu,
                     CVector &b1_gpu);

  /** \brief Computation of the primal-dual gap
   *
   */
  RType ComputePDGap(CVector &x1, std::vector<CVector> &x2, CVector &x3,
                     std::vector<CVector> &x4, std::vector<CVector> &y1,
                     std::vector<CVector> &y2, std::vector<CVector> &y3,
                     std::vector<CVector> &y4, CVector &z, CVector &data_gpu,
                     CVector &b1_gpu);

  /** \brief Perform the iterative reconstruction.
   *
   * \param data_gpu the measured k-space data
   * \param x reconstructed image, dims: width * height * frames
   * \param b1_gpu coil sensitivies
   */
  void IterativeReconstruction(CVector &data_gpu, CVector &x, CVector &b1_gpu);

  void ExportAdditionalResults(const char* outputDir, ResultExportCallback callback);

 private:
  ICTGV2Params params;
  void InitParams();
  void InitLambda(bool adaptLambda);
  void InitTempVectors();
  void InitPrimalVectors(unsigned N);
  void InitDualVectors(unsigned N);

  std::vector<CType> pdGapExport;
  std::vector<CType> dataFidelityExport;
  std::vector<CType> ictgvNormExport;

  RType tsw1parnorm;
  RType tsw2parnorm;
  RType alphanorm;
  RType dxnorm;
  RType dynorm;
  RType dx2norm;
  RType dy2norm;
  RType dtnorm;
  RType dt2norm;
  RType ictgv2NormParnorm;
  RType ictgv2NormNow;
  RType normfac;
  
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

  std::vector<CVector> x2;
  std::vector<CVector> ext2;
  std::vector<CVector> x2_old;

  CVector x3;
  CVector ext3;
  CVector x3_old;

  std::vector<CVector> x4;
  std::vector<CVector> ext4;
  std::vector<CVector> x4_old;

  // dual vectors
  std::vector<CVector> y1;
  std::vector<CVector> y2;
  std::vector<CVector> y3;
  std::vector<CVector> y4;
};

#endif  // INCLUDE_ICTGV2_H_
