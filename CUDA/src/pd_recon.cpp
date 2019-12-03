#include "../include/pd_recon.h"
#include "../include/cartesian_operator.h"
#include "../include/cartesian_operator3d.h"
#include "../include/cartesian_operator4d.h"
#include <stdexcept>

PDRecon::PDRecon(unsigned width, unsigned height, unsigned depth, unsigned coils,
                 unsigned frames, BaseOperator *mrOp)
  : width(width), height(height), depth(depth), coils(coils), frames(frames), mrOp(mrOp),
    debug(false), debugstep(1)
{
}

PDRecon::~PDRecon()
{
}

void PDRecon::Log(const char *format, ...)
{
  if (verbose)
  {
    va_list argptr;
    va_start(argptr, format);
    vfprintf(stdout, format, argptr);
    va_end(argptr);
  }
}

void PDRecon::SetVerbose(bool verbose)
{
  this->verbose = verbose;
}

void PDRecon::SetDebug(bool debug, int debugstep)
{
  this->debug = debug;
  this->debugstep = debugstep;
}

void PDRecon::AdaptStepSize(RType nKx, RType nx)
{
  RType tmp = nx / nKx;
  RType theta = 0.95;

  PDParams &params = GetParams();

  // Check convergence condition
  if (params.sigma * params.tau > std::pow(tmp, 2))
  {
    if (std::pow(theta, 2) * params.sigma * params.tau < std::pow(tmp, 2))
    {
      params.sigma *= theta;
      params.tau *= theta;
    }
    else
    {
      params.sigma = tmp * params.sigmaTauRatio;
      params.tau = tmp / params.sigmaTauRatio;
    }
  }
}

RType PDRecon::AdaptLambda(RType k, RType d)
{
  return mrOp->AdaptLambda(k, d);
}

void PDRecon::TestAdjointness(CVector &b1)
{
}

void PDRecon::ComputeTimeSpaceWeights(RType timeSpaceWeight, RType &ds, RType &dt)
{

  std::cout << "timeSpaceWeight : " << timeSpaceWeight << std::endl;
  RType timeSpaceRatio = 1.0 / timeSpaceWeight;

  RType tmp, w1, w2;

  if (timeSpaceRatio > 0.0 && timeSpaceRatio <= 1)
  {
    tmp = utils::Ellipke(1 - std::pow(timeSpaceRatio, 2.0));
    w2 = M_PI / (2.0 * tmp);
    w1 = timeSpaceRatio * w2;
  }
  else if (timeSpaceRatio > 1)
  {
    timeSpaceRatio = 1.0 / timeSpaceRatio;
    tmp = utils::Ellipke(1 - std::pow(timeSpaceRatio, 2.0));
    w1 = M_PI / (2.0 * tmp);
    w2 = timeSpaceRatio * w1;
  }
  else
    throw std::invalid_argument("ComputeTimeSpaceWeights: Invalid ratio.");

  ds = (RType) 1.0 / w1;
  dt = (RType) 1.0 / w2;
}

void PDRecon::IterativeReconstructionASL(CVector &data_gpu_c, CVector &data_gpu_l, CVector &x1, CVector &x3, CVector &b1_gpu)
{
}																	   
void PDRecon::IterativeReconstruction(CVector &data_gpu, CVector &x,
                                      CVector &b1_gpu)
{
}
