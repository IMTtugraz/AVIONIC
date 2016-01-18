#ifndef INCLUDE_NONCARTESIAN_COIL_CONSTRUCTION_H_

#define INCLUDE_NONCARTESIAN_COIL_CONSTRUCTION_H_

#include "../include/coil_construction.h"
#include "./noncartesian_operator.h"

/**
 * \brief
 *
 *
 */
class NoncartesianCoilConstruction : public CoilConstruction
{
 public:
  NoncartesianCoilConstruction(unsigned width, unsigned height, unsigned coils,
                               unsigned frames, NoncartesianOperator *mrOp);

  NoncartesianCoilConstruction(unsigned width, unsigned height, unsigned coils,
                               unsigned frames, CoilConstructionParams &params,
                               NoncartesianOperator *mrOp);

  virtual ~NoncartesianCoilConstruction();

  void TimeAveragedReconstruction(CVector &kdata, CVector &u, CVector &crec,
                                  bool applyPhase = true);

 private:
  NoncartesianOperator *mrOp;
};

#endif  // INCLUDE_NONCARTESIAN_COIL_CONSTRUCTION_H_
