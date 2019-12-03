#include "../include/cartesian_operator4d.h"

CartesianOperator4D::CartesianOperator4D(unsigned width, unsigned height, unsigned depth,
                    unsigned coils, unsigned frames, RVector &mask)
  : BaseOperator(width, height, depth, coils, frames), centered(true), mask(mask)
{
  Init();
}

CartesianOperator4D::CartesianOperator4D(unsigned width, unsigned height, unsigned depth,
                    unsigned coils, unsigned frames, RVector &mask, bool centered)
    : BaseOperator(width, height, depth, coils, frames), centered(centered), mask(mask)
{
  std::cout << " CartesianOperator4D " << std::endl;
  Init();
}

RVector ZeroMaskASL4D(0);

CartesianOperator4D::CartesianOperator4D(unsigned width, unsigned height, unsigned depth, 
                    unsigned coils, unsigned frames, bool centered)
  : BaseOperator(width, height, depth, coils, frames), centered(centered), mask(ZeroMaskASL4D)
{
  Init();
}


CartesianOperator4D::CartesianOperator4D(unsigned width, unsigned height,
                                     unsigned depth, unsigned coils, unsigned frames)
    : BaseOperator(width, height, depth, coils, frames), centered(true), mask(ZeroMaskASL4D)
{
  Init();
}

//TODO: put cufftplan initialization and destroy in Init and Destructor
CartesianOperator4D::~CartesianOperator4D()
{
//  cufftDestroy(fftplan3d);
}

void CartesianOperator4D::Init()
{
//  cufftResult cres;
//  cufftHandle fftplan3d;
//  cres = cufftPlan3d(&fftplan3d, width, height, depth, CUFFT_C2C);
}

RType CartesianOperator4D::AdaptLambda(RType k, RType d)
{
  RType lambda = 0.0;
//  RType subfac = (width * height * depth);
//  if (!mask.empty())
//  {
//    subfac /= std::pow(agile::norm2(mask), 2);
//  }
//  else
//  {
//    subfac /= width * height * depth;
//  }
//  std::cout << "Acceleration factor:" << subfac << std::endl;
//  lambda = subfac * k + d;
  return lambda;
}

//Image to k data
void CartesianOperator4D::BackwardOperation(CVector &x_gpu, CVector &z_gpu,
                                          CVector &b1_gpu)
{
 
  unsigned N = width * height * depth;
  CVector x_hat_gpu(N);
  //CVector x_out_data(N*coils*frames);
  const CType* in_data = x_hat_gpu.data();
  
  //TODO: put in init
  cufftResult cres;
  cufftHandle fftplan3d;
  cres = cufftPlan3d(&fftplan3d, depth, height, width, CUFFT_C2C);
 
  for (unsigned frame = 0; frame < frames; frame++)
  {
    unsigned offset_ = N*frame;
    //CType* in_data = x_gpu.data() + offset_;
    // perform backward operation
    for (unsigned coil = 0; coil < coils; coil++)
    {
      unsigned offset = coil * N;
      // apply b1 map
      agile::lowlevel::multiplyElementwise(
        x_gpu.data() + offset_, b1_gpu.data() + offset,
        x_hat_gpu.data(), N);
      if (centered)
      {      
        CType* out_data = z_gpu.data() + offset + offset_*coils;
        cres = cufftExecC2C(fftplan3d,(cufftComplex*)in_data ,(cufftComplex*)out_data, CUFFT_FORWARD);
        AGILE_ASSERT(cres == CUFFT_SUCCESS,
                       StandardException::ExceptionMessage(
                         "Error during FFT procedure"));
      }
      else
      {
        CType* out_data = z_gpu.data() + offset + offset_*coils;
        cres = cufftExecC2C(fftplan3d,(cufftComplex*)in_data ,(cufftComplex*)out_data, CUFFT_FORWARD);
        AGILE_ASSERT(cres == CUFFT_SUCCESS,
                       StandardException::ExceptionMessage(
                         "Error during FFT procedure"));
      }

      if (!mask.empty())
      {
        agile::lowlevel::multiplyElementwise(
        z_gpu.data() + offset + offset_*coils, mask.data() + offset_,
        z_gpu.data() + offset + offset_*coils, N);
      }
    }   
  }
  //if (!mask.empty())
  //{
  //  agile::multiplyElementwise(z_gpu, mask, z_gpu);
  //}
  agile::scale((CType)(1.0 / std::sqrt(N)), z_gpu, z_gpu);    
  cufftDestroy(fftplan3d);
}

CVector CartesianOperator4D::BackwardOperation(CVector &x_gpu, CVector &b1_gpu)
{
  unsigned int N = width * height * depth;
  CVector z_gpu(N * coils * frames);
  z_gpu.assign(N * coils * frames, 0.0);
  this->BackwardOperation(x_gpu, z_gpu, b1_gpu);
  return z_gpu;
}

//kspace to image
void CartesianOperator4D::ForwardOperation(CVector &x_gpu, CVector &sum,
                                         CVector &b1_gpu)
{
  unsigned N = width * height * depth * frames; 
  CVector z_gpu(N/frames);
  z_gpu.assign(N/frames,0.0);

  cufftResult cres;
  cufftHandle fftplan3d;
  cres = cufftPlan3d(&fftplan3d, depth, height, width, CUFFT_C2C);

  // Set sum vector to zero
  sum.assign(N, 0.0);
  
  //if (!mask.empty())
  //{
  //  agile::multiplyElementwise(x_gpu, mask, x_gpu);
  //}

  for (unsigned frame = 0; frame < frames; frame++)
  {
    //unsigned offset_ = N/frames*frame;
    unsigned offset_ = width*height*depth*frame;
    // perform forward operation
    for (unsigned coil = 0; coil < coils; coil++)
    {
      unsigned int offset = coil * N/frames;
      if (!mask.empty())
      {
        agile::lowlevel::multiplyElementwise(
        x_gpu.data() + offset + offset_*coils, mask.data() + offset_,
        x_gpu.data() + offset + offset_*coils, N/frames);
      }
      if (centered)
      {
        const CType* in_data = x_gpu.data()+offset+offset_*coils;
        CType* out_data = z_gpu.data();   
        cres = cufftExecC2C(fftplan3d,(cufftComplex*)in_data,(cufftComplex*)out_data, CUFFT_INVERSE);
        AGILE_ASSERT(cres == CUFFT_SUCCESS,
                       StandardException::ExceptionMessage(
                         "Error during FFT procedure"));
      }
      else
      {
        const CType*  in_data = x_gpu.data()+offset+offset_*coils;
        CType* out_data = z_gpu.data();
        cufftExecC2C(fftplan3d,(cufftComplex*)in_data,(cufftComplex*)out_data, CUFFT_INVERSE);
        AGILE_ASSERT(cres == CUFFT_SUCCESS,
                        StandardException::ExceptionMessage(
                          "Error during FFT procedure"));
       }
       agile::scale((CType)(1.0 / std::sqrt(N/frames)), z_gpu, z_gpu);
       // apply adjoint b1 map
       agile::lowlevel::multiplyConjElementwise(
        b1_gpu.data() + coil * N/frames, z_gpu.data(), z_gpu.data(), N/frames);

       agile::lowlevel::addVector(
        z_gpu.data(), sum.data() + offset_,
        sum.data() + offset_, N/frames);
    } 
  }
cufftDestroy(fftplan3d);
}

CVector CartesianOperator4D::ForwardOperation(CVector &x_gpu, CVector &b1_gpu)
{
  unsigned N = width * height * depth * frames;
  CVector sum_gpu(N);
  ForwardOperation(x_gpu, sum_gpu, b1_gpu);
  return sum_gpu;
}
