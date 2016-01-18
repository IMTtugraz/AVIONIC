#include "../include/cartesian_coil_construction.h"

CartesianCoilConstruction::CartesianCoilConstruction(unsigned width,
                                                     unsigned height,
                                                     unsigned coils,
                                                     unsigned frames,
                                                     CartesianOperator *mrOp)
  : CoilConstruction(width, height, coils, frames), mrOp(mrOp)
{
}

CartesianCoilConstruction::CartesianCoilConstruction(
    unsigned width, unsigned height, unsigned coils, unsigned frames,
    CoilConstructionParams &params, CartesianOperator *mrOp)
  : CoilConstruction(width, height, coils, frames, params), mrOp(mrOp)
{
}

CartesianCoilConstruction::~CartesianCoilConstruction()
{
}

void CartesianCoilConstruction::TimeAveragedReconstruction(CVector &kdata,
                                                           CVector &u,
                                                           CVector &crec,
                                                           bool applyPhase)
{
  // normalize coil-wise kspace data
  CVector dataTemp(width * height * coils);
  RVector maskTemp(width * height);
  maskTemp.assign(maskTemp.size(), 0);
  dataTemp.assign(dataTemp.size(), 0);

  for (unsigned cnt = 0; cnt < frames; cnt++)
  {
    unsigned cOff = cnt * width * height * coils;
    agile::lowlevel::addVector(dataTemp.data(), kdata.data() + cOff,
                               dataTemp.data(), width * height * coils);

    agile::lowlevel::addVector(maskTemp.data(),
                               mrOp->mask.data() + cnt * width * height,
                               maskTemp.data(), width * height);
  }
  agile::max(maskTemp, RType(1.0), maskTemp);

  CVector temp(width * height);
  CVector angle(width * height);
  angle.assign(angle.size(), 0);
  crec.assign(crec.size(), 0);
  RVector angleReal(width * height);

  for (unsigned cnt = 0; cnt < coils; cnt++)
  {
    unsigned cOff = cnt * width * height;
    agile::lowlevel::divideElementwise(dataTemp.data() + cOff, maskTemp.data(),
                                       dataTemp.data() + cOff, width * height);

    temp.assign(temp.size(), 0);
    if (mrOp->centered)
      mrOp->fftOp->CenteredForward(dataTemp, temp, cOff, 0);
    else
      mrOp->fftOp->Forward(dataTemp, temp, cOff, 0);

    utils::SetSubVector(temp, crec, cnt, width * height);

    agile::addVector(angle, temp, angle);
    agile::multiplyConjElementwise(temp, temp, temp);
    agile::addVector(u, temp, u);
  }
  agile::sqrt(u, u);

  if (applyPhase)
  {
    agile::phaseVector(angle, angleReal);
    agile::scale(CType(0, 1.0), angleReal, angle);
    agile::expVector(angle, angle);

    agile::multiplyElementwise(u, angle, u);
  }
}

