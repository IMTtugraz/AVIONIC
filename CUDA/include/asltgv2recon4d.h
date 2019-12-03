#ifndef INCLUDE_ASLTGV2RECON4D_H_

#define INCLUDE_ASLTGV2RECON4D_H_

#include "./tgv2.h"
#include "./pd_recon.h"
#include <algorithm>

/** \brief Parameter struct used in ASLTGV2IMG reconstruction. */
typedef struct ASLTGV2RECON4DParams : public TGV2Params
{
  RType lambda_l;
  RType lambda_c;
  /** \brief Trade-off between first and second TGV functional */
  RType alpha;
  /** \brief Step size for difference computation in second spatial direction */
  RType dx;
  /** \brief Step size for difference computation in second spatial direction */
  RType dy;
  /** \brief Step size for difference computation in third spatial direction */
  RType dz;
  /** \brief Step size for difference computation in time direction */
  RType dt;
  /** \brief spatio-temporal weight.*/
  RType timeSpaceWeight;

} ASLTGV2RECON4DParams;

/** \brief ASLTGV2IMG regularized iterative reconstruction
 *
 * The necessary MR Operator (Cartesian, Non-Cartesian)
 * used for forward/backward operation is
 * passed by the constructor.
 */
class ASLTGV2RECON4D : public PDRecon
{
 public:
  ASLTGV2RECON4D(unsigned width, unsigned height, unsigned depth, unsigned coils,
           unsigned frames, BaseOperator *mrOp);

  ASLTGV2RECON4D(unsigned width, unsigned height, unsigned depth, unsigned coils,
            unsigned frames, ASLTGV2RECON4DParams &params, BaseOperator *mrOp);

  virtual ~ASLTGV2RECON4D();

  /** \brief Return reference to the ASLTGV2IMGParams.*/
  PDParams &GetParams();

  /** \brief Test adjointness of forward and backward operation, i.e. 
   * evaluates <K^H v, u> = <v, Ku> on random vectors u,v.
   */
  void TestAdjointness(CVector &b1);


  /** \brief Adapt stepsizes (sigma, tau) by checking the convergence condition
   * based on nKx and nx.
   */
  void AdaptStepSize(CVector &extDiff1, std::vector<CVector> &extDiff2,
                     CVector &extDiff3, std::vector<CVector> &extDiff4, std::vector<CVector> &extDiff5, CVector &b1_gpu);

  /** \brief Compute Datafidelity
   */
  RType ComputeDataFidelity(CVector &x1, CVector &data_gpu, CVector &b1_gpu, RType lambda);

  /** \brief G* Computation, needed in ComputePDGap function. */
  /*RType ComputeGStar(CVector &x1, std::vector<CVector> &y1,
                     std::vector<CVector> &y2, std::vector<CVector> &y3,
                     std::vector<CVector> &y4, CVector &z, CVector &data_gpu,
                     CVector &b1_gpu);*/

  /** \brief Computation of the primal-dual gap
   *
   */
  /*RType ComputePDGap(CVector &x1, std::vector<CVector> &x2, CVector &x3,
                     std::vector<CVector> &x4, std::vector<CVector> &y1,
                     std::vector<CVector> &y2, std::vector<CVector> &y3,
                     std::vector<CVector> &y4, CVector &z, CVector &data_gpu,
                     CVector &b1_gpu);*/

RType ComputePDGap(std::vector<CVector> &y1, std::vector<CVector> &y2, std::vector<CVector> &y3, std::vector<CVector> &y4, CVector &y5, CVector &y6, std::vector<CVector> &y7, std::vector<CVector> &y8, CVector &data_gpu_c, CVector &data_gpu_l, CVector &b1_gpu);

  /** \brief Perform the iterative reconstruction.
   *
   * \param data_gpu_c the measured control data
   * \param data_gpu_l the measured label data
   * \param x1 mean label image, dims: width * height * depth
   * \param x3 mean control image, dims: width * height * depth
   */
  void IterativeReconstructionASL(CVector &data_gpu_c, CVector &data_gpu_l, CVector &x1, CVector &x3, CVector &b1_gpu);

  void ExportAdditionalResults(const char* outputDir, ResultExportCallback callback);

 private:
  ASLTGV2RECON4DParams params;
  void InitParams();
  void InitLambda(bool adaptLambda);
  void InitTempVectors();
  void InitPrimalVectors(unsigned N);
  void InitDualVectors(unsigned N);

  std::vector<CType> pdGapExport;
  // RType datafidelity;

  CVector norm_gpu;
  CVector temp;
  CVector imgTemp;
  CVector zTemp;
  CVector data_aver_c;
  CVector data_aver_l;
  CVector div1Temp;
  CVector div3Temp;
  std::vector<CVector> div2Temp;
  std::vector<CVector> y1Temp;
  std::vector<CVector> y2Temp;
  std::vector<CVector> y3Temp;
  //std::vector<CVector> y4Temp;
  CVector y5Temp;
  //CVector y6Temp;
  std::vector<CVector> y7Temp;
  //std::vector<CVector> y8Temp;

  // primal vectors
  CVector ext1;
  CVector x1_old;

  std::vector<CVector> x2;
  std::vector<CVector> ext2;
  std::vector<CVector> x2_old;

  //CVector x3;
  CVector ext3;
  CVector x3_old;

  std::vector<CVector> x4;
  std::vector<CVector> ext4;
  std::vector<CVector> x4_old;
  
  std::vector<CVector> x5;
  std::vector<CVector> ext5;
  std::vector<CVector> x5_old;

  // dual vectors
  std::vector<CVector> y1;
  std::vector<CVector> y2;
  std::vector<CVector> y3;
  std::vector<CVector> y4;
  CVector y5;
  CVector y6;
  std::vector<CVector> y7;
  std::vector<CVector> y8;

};

#endif  // INCLUDE_ASLTGV2RECON4D_H_
