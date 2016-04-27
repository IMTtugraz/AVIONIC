#ifndef INCLUDE_TGV2_3D_H_

#define INCLUDE_TGV2_3D_H_

#include "./pd_recon.h"

/** \brief Parameter struct used in TGV reconstruction. */
typedef struct TGV2_3DParams : public PDParams
{
  /** \brief TGV-norm weight, trade-off between first and second derivative */
  RType alpha0;
  /** \brief TGV-norm weight, trade-off between first and second derivative */
  RType alpha1;
} TGV2_3DParams;

/** \brief TGV2 regularized iterative reconstruction
 *
 * The necessary MR Operator (Cartesian, Non-Cartesian)
 * used for forward/backward operation is
 * passed by the constructor.
 */
class TGV2_3D : public PDRecon
{
 public:
  TGV2_3D(unsigned width, unsigned height, unsigned depth, unsigned coils,
       BaseOperator *mrOp);

  TGV2_3D(unsigned width, unsigned height, unsigned depth, unsigned coils,
       TGV2_3DParams &params, BaseOperator *mrOp);

  virtual ~TGV2_3D();

   /** \brief Test adjointness of forward and backward operation, i.e. 
   * evaluates <K^H v, u> = <v, Ku> on random vectors u,v.
   */
  void TestAdjointness(CVector &b1);


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

  void ExportAdditionalResults(const char* outputDir, ResultExportCallback callback);

 private:
  TGV2_3DParams params;
  void InitParams();
  void InitLambda(bool adaptLambda);

  std::vector<CType> pdGapExport;

};

#endif  // INCLUDE_TGV2_3D_H_
