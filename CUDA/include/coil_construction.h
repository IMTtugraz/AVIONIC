#ifndef INCLUDE_COIL_CONSTRUCTION_H_

#define INCLUDE_COIL_CONSTRUCTION_H_

#include <vector>
#include <complex>
#include "./types.h"
#include "./utils.h"
#include "./cg_forward_operation.h"
#include "./pd_recon.h"

#include "agile/agile.hpp"

#include "agile/gpu_vector.hpp"
#include "agile/calc/fft.hpp"
#include "agile/operator/cg.hpp"

/** \brief Parameter struct used in coil reconstruction. */
typedef struct CoilConstructionParams : public PDParams
{
  /** \brief H1-regularization parameter \f$\mu\f$ */
  RType uH1mu;

  RType uReg;
  unsigned uNrIt;
  RType uTau;
  RType uSigma;
  RType uSigmaTauRatio;
  RType uAlpha0;
  RType uAlpha1;

  RType b1Reg;
  unsigned b1NrIt;
  RType b1Tau;
  RType b1Sigma;
  RType b1SigmaTauRatio;

  RType b1FinalReg;
  unsigned b1FinalNrIt;

} CoilConstructionParams;

typedef agile::GPUCommunicator<unsigned, CType, CType> communicator_type;

typedef ForwardOperation<communicator_type, CVector> forward_type;

/**
 * \brief Class to estimate sensitivity (b1) data based on kspace data and
 * pattern
 *
 * Iteratively constructs coil sensitivities. Basic approach:
 *
 * 1. Compute time-averaged reconstruction (TimeAveragedReconstruction)
 * 2. Compute absolute values by H1-regularization (B1FromUH1)
 * 3. Iteratively update sensitivies with UTGV2Recon and B1Recon
 *
 */
class CoilConstruction : public PDRecon
{
 public:
  CoilConstruction(unsigned width, unsigned height, unsigned coils,
                   unsigned frames);

  CoilConstruction(unsigned width, unsigned height, unsigned coils,
                   unsigned frames, CoilConstructionParams &params);

  virtual ~CoilConstruction();

  PDParams &GetParams();

  /** \brief Find index of coil with maximum sum over elements (l1-norm) */
  unsigned FindMaximumSum(CVector &crec);

  /** \brief Reconstruct the time averaged k-space, i.e. frame-wise k-space data
   *is accumulated to one full k-space
   * \param[in] kdata coil-wise k-space data, dims: width * height * coils *
   *frames
   * \param[out] u reconstructed time averaged magnitude image (sum-of-squares
   *over coils), but with phase-sensitive information
   * \param[out] crec reconstructed coil images
   * */
  virtual void TimeAveragedReconstruction(CVector &kdata, CVector &u,
                                          CVector &crec,
                                          bool applyPhase = true) = 0;

  /** \brief Compute absolute value of b1 sensitivities by H1-regularization.
   *
   * \f$|b_j| = arg \min \frac{\mu}{2}\|bu_0-|crec_j|\|^2_2+\|\nabla b\|_2\f$
   *
   * */
  void B1FromUH1(CVector &u, CVector &crec, communicator_type &com,
                 CVector &b1);

  /** \brief Compute u over given set of coil indices.
   *
   * \f$u = arg \min_u (\nu TGV(u) + \sum_k\|\sigma_ku-v_k\|^2_2)\f$
   *
   * The set of coil indices defines which coils have to be processed.
   * The order is defined in PerformCoilConstruction method.
   *
   * \param[in] b10 Current state of already estimated sensitivities
   * \param[in] crec0 Coil reconstructions performed by
   *TimeAveragedReconstruction
   * \param[in] inds Set of coil indices, which have to be processed
   * \param[out] x1 Reconstructed u
   *
   * */
  void UTGV2Recon(CVector &b10, CVector &crec0, CVector &x1,
                  std::vector<unsigned> inds);

  /** \brief Compute new b1 from iteratively updated u
   *
   * \f$\phi = arg \min_p (\frac{\mu}{2}\|\nabla p\|^2_2 +
   *\frac{1}{2}\|p|\sigma|u-v\|^2_2\|^2_2)\f$
   *
   * \param[in] u0 Current u computed by B1FromUH1
   * \param[in] crec Coil reconstructions created by TimeAveragedReconstruction
   * \param[in] coils Amount of already processed coils
   * \param[in] maxIt Maximum number of iterations
   * \param[in] mu Regularization parameter
   * \param[out] b1 new coil sensitivities for the defined amount of coils
   *
   * */
  void B1Recon(CVector &u0, CVector &crec, CVector &b1, unsigned coils,
               unsigned maxIt = 500, RType mu = 2);

  /** \brief Construct coil sensitivities and initial image estimate from
   *k-space data.
   *
   * \param[in] kdata k-space data, dims: width*height*coils*frames
   * \param[out] u image estimate
   * \param[b1] b1 coil sensitivities
   * */
  void PerformCoilConstruction(CVector &kdata, CVector &u, CVector &b1,
                               communicator_type &com);

  void ExportAdditionalResults(const char *outputDir,
                               ResultExportCallback callback);

 private:
  CoilConstructionParams params;
  void InitParams();
};

#endif  // INCLUDE_COIL_CONSTRUCTION_H_
