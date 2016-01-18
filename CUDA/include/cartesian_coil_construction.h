#ifndef INCLUDE_CARTESIAN_COIL_CONSTRUCTION_H_

#define INCLUDE_CARTESIAN_COIL_CONSTRUCTION_H_

#include "../include/coil_construction.h"
#include "./cartesian_operator.h"

/**
 * \brief
 *
 *
 */
class CartesianCoilConstruction : public CoilConstruction
{
 public:
  CartesianCoilConstruction(unsigned width, unsigned height, unsigned coils,
                            unsigned frames, CartesianOperator *mrOp);

  CartesianCoilConstruction(unsigned width, unsigned height, unsigned coils,
                            unsigned frames, CoilConstructionParams &params,
                            CartesianOperator *mrOp);

  virtual ~CartesianCoilConstruction();

  void TimeAveragedReconstruction(CVector &kdata, CVector &u, CVector &crec, bool applyPhase = true);

 private:
  CartesianOperator *mrOp;
};

#endif  // INCLUDE_CARTESIAN_COIL_CONSTRUCTION_H_
