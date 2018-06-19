#include "../include/h1_recon.h"
#include "../include/cartesian_operator.h"
#include "../include/cartesian_operator3d.h"
#include "agile/gpu_vector_base.hpp"
#include <stdexcept>

H1Recon::H1Recon(unsigned width, unsigned height, unsigned depth, unsigned coils,
                 H1Params &params, CartesianOperator3D* mrOp, communicator_type &com)
  : width(width), height(height), depth(depth), coils(coils), mrOp(mrOp),
                 params(params), com(com)
{
}

H1Recon::H1Recon(unsigned width, unsigned height, unsigned depth, unsigned coils,
                 CartesianOperator3D *mrOp, communicator_type &com)
  : width(width), height(height), depth(depth), coils(coils), mrOp(mrOp), com(com)
{
    InitParams();
}

H1Recon::~H1Recon()
{
}


void H1Recon::SetVerbose(bool verbose)
{
  this->verbose = verbose;
}


void H1Recon::InitParams()
{
  params.maxIt = 500;

  params.dx = 1.0;
  params.dy = 1.0;
  params.dz = 1.0;

  params.absTol = 1e-6;
  params.mu = 1e-3;
  params.relTol = 1e-12;
}



void H1Recon::IterativeReconstruction(CVector &data_gpu, CVector &x,
                                      CVector &b1_gpu)
{
    forward_type forward(com, mrOp, params, b1_gpu, width, height, depth, coils);
    unsigned N = width*height*depth;
    CVector rhs = CVector(N);
    rhs.assign(N, 0.0);
    mrOp->ForwardOperation(data_gpu, rhs, b1_gpu);
    agile::scale(params.mu, rhs, rhs);

    typedef agile::ScalarProductMeasure<communicator_type> measure_type;
    measure_type scalar_product(com);

    agile::ConjugateGradient<communicator_type, forward_type, measure_type> cg(
          com, forward, scalar_product, params.relTol, params.absTol,
          params.maxIt);

    cg(rhs, x);

    if (cg.convergence())
        std::cout<<"CG converged in "<<cg.getIteration()<<" with a residual of "<<cg.getRho()<<std::endl;
    else
        std::cout<<"CG did not converged in "<<params.maxIt<<" with a residual of "<<cg.getRho()<<std::endl;


}

H1Params &H1Recon::GetParams()
{
    return params;
}
