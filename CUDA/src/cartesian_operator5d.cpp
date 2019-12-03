#include "../include/cartesian_operator5d.h"
CartesianOperator5D::CartesianOperator5D(unsigned width, unsigned height,
                                     unsigned depth, unsigned coils, 
                                     RVector &mask, bool centered, int cufft_mode)
  : BaseOperator(width, height, depth, coils, 0), centered(centered), mask(mask), cufft_mode(cufft_mode)
{
  Init();
}

CartesianOperator5D::CartesianOperator5D(unsigned width, unsigned height,
                                     unsigned depth, unsigned coils,
                                     RVector &mask, int cufft_mode)
  : BaseOperator(width, height, depth, coils, 0), centered(true), mask(mask), cufft_mode(cufft_mode)
{
  Init();
}

RVector ZeroMask5d(0);

CartesianOperator5D::CartesianOperator5D(unsigned width, unsigned height,
                                     unsigned depth, unsigned coils,
                                     bool centered, int cufft_mode)
    : BaseOperator(width, height, depth, coils, 0), centered(centered), mask(ZeroMask5d), cufft_mode(cufft_mode)
{
  Init();
}
CartesianOperator5D::CartesianOperator5D(unsigned width, unsigned height,
                                     unsigned depth, unsigned coils, int cufft_mode)
    : BaseOperator(width, height, depth, coils, 0), centered(true), mask(ZeroMask5d), cufft_mode(cufft_mode)
{
  Init();
}

//TODO: put cufftplan initialization and destroy in Init and Destructor
CartesianOperator5D::~CartesianOperator5D()
{
//  cufftDestroy(fftplan3d);
}

void CartesianOperator5D::Init()
{
  //cufftResult cres;
  //cufftHandle fftplan3d;
  //cres = cufftPlan3d(&fftplan3d, width, height, depth, CUFFT_C2C);
}

RType CartesianOperator5D::AdaptLambda(RType k, RType d)
{
  RType lambda = 0.0;
  RType subfac = (width * height * depth);
  if (!mask.empty())
  {
    subfac /= std::pow(agile::norm2(mask), 2);
  }
  else
  {
    subfac /= width * height * depth;
  }
  std::cout << "Acceleration factor:" << subfac << std::endl;
  lambda = subfac * k + d;
  return lambda;
}

void CartesianOperator5D::ForwardOperation(CVector &x_gpu, CVector &sum,
                                         CVector &b1_gpu)
{
  unsigned N = width * height * depth; 
  CVector z_gpu(N);

  cufftResult cres;
  cufftHandle fftplan3d;
  cres = cufftPlan3d(&fftplan3d, depth, height, width, CUFFT_C2C);

  // Set sum vector to zero
  sum.assign(N, 0.0);

  // perform forward operation
  for (unsigned coil = 0; coil < coils; coil++)
  {
    unsigned int offset = coil * N;
 
    if (!mask.empty())
    {
      agile::lowlevel::multiplyElementwise(
      x_gpu.data() + offset, mask.data(),
      x_gpu.data() + offset, N);
    }

    if (centered)
    {
      //fftOp->CenteredForward(x_gpu, z_gpu, x_offset, 0);
      const CType* in_data = x_gpu.data()+offset;
      CType* out_data = z_gpu.data();   

      if (cufft_mode == NORMAL)
    	  cres = cufftExecC2C(fftplan3d,(cufftComplex*)in_data,(cufftComplex*)out_data, CUFFT_FORWARD);
      else if (cufft_mode == FOR_BS)
    	  cres = cufftExecC2C(fftplan3d,(cufftComplex*)in_data,(cufftComplex*)out_data, CUFFT_INVERSE);

      AGILE_ASSERT(cres == CUFFT_SUCCESS,
                       StandardException::ExceptionMessage(
                         "Error during FFT procedure"));
    }
    else
    {
      //fftOp->Forward(x_gpu, z_gpu, x_offset, 0);
      const CType*  in_data = x_gpu.data()+offset;
      CType* out_data = z_gpu.data();

      if (cufft_mode == NORMAL)
    	  cufftExecC2C(fftplan3d,(cufftComplex*)in_data,(cufftComplex*)out_data, CUFFT_FORWARD);
      else if (cufft_mode == FOR_BS)
    	  cufftExecC2C(fftplan3d,(cufftComplex*)in_data,(cufftComplex*)out_data, CUFFT_INVERSE);

      AGILE_ASSERT(cres == CUFFT_SUCCESS,
                        StandardException::ExceptionMessage(
                          "Error during FFT procedure"));

   }

    agile::scale((CType)(1.0 / std::sqrt(N)), z_gpu, z_gpu);


    // apply adjoint b1 map
    agile::lowlevel::multiplyConjElementwise(
        b1_gpu.data() + coil * N, z_gpu.data(), z_gpu.data(), N);

    agile::lowlevel::addVector(
        z_gpu.data(), sum.data() ,
        sum.data() , N); 
  }
cufftDestroy(fftplan3d);
}


CVector CartesianOperator5D::ForwardOperation(CVector &x_gpu, CVector &b1_gpu)
{
  unsigned N = width * height * depth;
  CVector sum_gpu(N);
  ForwardOperation(x_gpu, sum_gpu, b1_gpu);
  return sum_gpu;
}


void CartesianOperator5D::BackwardOperation(CVector &x_gpu, CVector &z_gpu,
                                          CVector &b1_gpu)
{

  unsigned N = width * height * depth;
  CVector x_hat_gpu(N);
  const CType* in_data = x_hat_gpu.data();
  
  //TODO: put in init
  cufftResult cres;
  cufftHandle fftplan3d;
  cres = cufftPlan3d(&fftplan3d, depth, height, width, CUFFT_C2C);
 
  // perform backward operation
  for (unsigned coil = 0; coil < coils; coil++)
  {
    unsigned offset = coil * N;
    // apply b1 map
    agile::lowlevel::multiplyElementwise(
        x_gpu.data() , b1_gpu.data() + offset,
        x_hat_gpu.data(), N);


    if (centered)
    {
      //fftOp->CenteredInverse(x_hat_gpu, z_gpu, 0, z_offset);

      CType* out_data = z_gpu.data() + offset;

      if (cufft_mode == NORMAL)
    	  cres = cufftExecC2C(fftplan3d,(cufftComplex*)in_data ,(cufftComplex*)out_data, CUFFT_INVERSE);
      else if (cufft_mode == FOR_BS)
    	  cres = cufftExecC2C(fftplan3d,(cufftComplex*)in_data ,(cufftComplex*)out_data, CUFFT_FORWARD);

      AGILE_ASSERT(cres == CUFFT_SUCCESS,
                       StandardException::ExceptionMessage(
                         "Error during FFT procedure"));
    }
    else
    {
     //fftOp->Inverse(x_hat_gpu, z_gpu, 0, z_offset);
      CType* out_data = z_gpu.data() + offset;

      if (cufft_mode == NORMAL)
    	  cres = cufftExecC2C(fftplan3d,(cufftComplex*)in_data ,(cufftComplex*)out_data, CUFFT_INVERSE);
      else if (cufft_mode == FOR_BS)
    	  cres = cufftExecC2C(fftplan3d,(cufftComplex*)in_data ,(cufftComplex*)out_data, CUFFT_FORWARD);

      AGILE_ASSERT(cres == CUFFT_SUCCESS,
                       StandardException::ExceptionMessage(
                         "Error during FFT procedure"));
   }
 
    if (!mask.empty())
    {
      agile::lowlevel::multiplyElementwise(
      z_gpu.data() + offset, mask.data() ,
      z_gpu.data() + offset, N);
    }
    
  }
 agile::scale((CType)(1.0 / std::sqrt(N)), z_gpu, z_gpu);
    
cufftDestroy(fftplan3d);
}

CVector CartesianOperator5D::BackwardOperation(CVector &x_gpu, CVector &b1_gpu)
{
  unsigned int N = width * height * depth;
  CVector z_gpu(N * coils);
  this->BackwardOperation(x_gpu, z_gpu, b1_gpu);
  return z_gpu;
}

