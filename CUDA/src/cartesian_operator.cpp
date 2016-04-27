#include "../include/cartesian_operator.h"

CartesianOperator::CartesianOperator(unsigned width, unsigned height,
                                     unsigned coils, unsigned frames,
                                     RVector &mask, bool centered)
  : BaseOperator(width, height, 0, coils, frames), centered(centered), mask(mask)
{
  Init();
}

CartesianOperator::CartesianOperator(unsigned width, unsigned height,
                                     unsigned coils, unsigned frames,
                                     RVector &mask)
  : BaseOperator(width, height, 0, coils, frames), centered(true), mask(mask)
{
  Init();
}

RVector ZeroMask(0);

CartesianOperator::CartesianOperator(unsigned width, unsigned height,
                                     unsigned coils, unsigned frames,
                                     bool centered)
  : BaseOperator(width, height,  0, coils, frames), centered(centered),
    mask(ZeroMask)
{
  Init();
}

CartesianOperator::CartesianOperator(unsigned width, unsigned height,
                                     unsigned coils, unsigned frames)
  : BaseOperator(width, height,  0, coils, frames), centered(true), mask(ZeroMask)
{
  Init();
}

CartesianOperator::~CartesianOperator()
{
  delete fftOp;
}

void CartesianOperator::Init()
{
  fftOp = new agile::FFT<CType>(height, width);
}

RType CartesianOperator::AdaptLambda(RType k, RType d)
{
  RType lambda = 0.0;
  RType subfac = (width * height * frames);
  if (!mask.empty())
  {
    subfac /= std::pow(agile::norm2(mask), 2);
  }
  else
  {
    subfac /= width * height * frames;
  }
  std::cout << "Acceleration factor:" << subfac << std::endl;
  lambda = subfac * k + d;
  return lambda;
}

void CartesianOperator::ForwardOperation(CVector &x_gpu, CVector &sum,
                                         CVector &b1_gpu)
{
  unsigned N = width * height * frames;
  CVector z_gpu(width * height);

  // Set sum vector to zero
  sum.assign(N, 0.0);

  // perform forward operation
  for (unsigned frame = 0; frame < frames; frame++)
  {
    unsigned offset = width * height * coils * frame;

    for (unsigned coil = 0; coil < coils; coil++)
    {
      unsigned int x_offset = offset + coil * width * height;

      if (!mask.empty())
      {
        agile::lowlevel::multiplyElementwise(
            x_gpu.data() + x_offset, mask.data() + width * height * frame,
            x_gpu.data() + x_offset, width * height);
      }

      if (centered)
        fftOp->CenteredForward(x_gpu, z_gpu, x_offset, 0);
      else
        fftOp->Forward(x_gpu, z_gpu, x_offset, 0);

      // apply adjoint b1 map
      agile::lowlevel::multiplyConjElementwise(
          b1_gpu.data() + coil * width * height, z_gpu.data(), z_gpu.data(),
          width * height);

      agile::lowlevel::addVector(
          z_gpu.data(), sum.data() + width * height * frame,
          sum.data() + width * height * frame, width * height);
    }
  }
}

CVector CartesianOperator::ForwardOperation(CVector &x_gpu, CVector &b1_gpu)
{
  unsigned N = width * height * frames;
  CVector sum_gpu(N);
  ForwardOperation(x_gpu, sum_gpu, b1_gpu);
  return sum_gpu;
}

void CartesianOperator::BackwardOperation(CVector &x_gpu, CVector &z_gpu,
                                          CVector &b1_gpu)
{
  CVector x_hat_gpu(width * height);

  // perform backward operation
  for (unsigned frame = 0; frame < frames; frame++)
  {
    unsigned offset = width * height * frame;

    for (unsigned coil = 0; coil < coils; coil++)
    {
      // apply b1 map
      agile::lowlevel::multiplyElementwise(
          x_gpu.data() + offset, b1_gpu.data() + coil * width * height,
          x_hat_gpu.data(), width * height);

      unsigned z_offset =
          frame * coils * width * height + coil * width * height;

      if (centered)
        fftOp->CenteredInverse(x_hat_gpu, z_gpu, 0, z_offset);
      else
        fftOp->Inverse(x_hat_gpu, z_gpu, 0, z_offset);

      if (!mask.empty())
      {
        agile::lowlevel::multiplyElementwise(
            z_gpu.data() + z_offset, mask.data() + width * height * frame,
            z_gpu.data() + z_offset, width * height);
      }
    }
  }
}

CVector CartesianOperator::BackwardOperation(CVector &x_gpu, CVector &b1_gpu)
{
  unsigned int N = width * height * frames;
  CVector z_gpu(N * coils);
  this->BackwardOperation(x_gpu, z_gpu, b1_gpu);
  return z_gpu;
}


