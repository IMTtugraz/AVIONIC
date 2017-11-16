#include "../include/tv_temp.h"
#include <stdexcept>
#include <math.h>

TVTEMP::TVTEMP(unsigned width, unsigned height, unsigned coils, unsigned frames,
       BaseOperator *mrOp)
  : PDRecon(width, height, 0, coils, frames, mrOp)
{
  InitParams();
  InitTempVectors();
}

TVTEMP::TVTEMP(unsigned width, unsigned height, unsigned coils, unsigned frames,
       TVtempParams &params, BaseOperator *mrOp)
  : PDRecon(width, height, 0, coils, frames, mrOp), params(params)
{
  InitLambda(params.adaptLambdaParams.adaptLambda);
  InitTempVectors();
}

TVTEMP::~TVTEMP()
{
}

void TVTEMP::InitLambda(bool adaptLambda)
{
  if (adaptLambda)
  {
    params.lambda =
        AdaptLambda(params.adaptLambdaParams.k, params.adaptLambdaParams.d);
    std::cout << "Adapted lambda with (k,d) (" << params.adaptLambdaParams.k
              << ", " << params.adaptLambdaParams.d << ") to " << params.lambda
              << std::endl;
  }
  else
    std::cout << "Using lambda: " << params.lambda << std::endl;
}

void TVTEMP::InitParams()
{
  params.maxIt = 500;
  params.stopPDGap = 0;

  params.sigma = 1.0 / 3.0;
  params.tau = 1.0 / 3.0;

  params.sigmaTauRatio = 1.0;

  params.dt = 1.0;

  params.adaptLambdaParams.k = 0.4 * 0.2991;
  params.adaptLambdaParams.d = 10.0 * 0.2991;
  InitLambda(true);
}

void TVTEMP::InitTempVectors()
{
  unsigned N = width * height * frames;

  imgTemp = CVector(N);
  divTemp = CVector(N);
  zTemp = CVector(0);  //< resized at runtime
}

PDParams &TVTEMP::GetParams()
{
  return params;
}

void TVTEMP::TestAdjointness(CVector &b1)
{
  //TODO: Implement
} 


void TVTEMP::AdaptStepSize(CVector &extDiff, CVector &b1)
{
  std::vector<CVector> gradient =
      utils::Gradient_temp(extDiff, width, height, params.dt);

  zTemp = mrOp->BackwardOperation(extDiff, b1);

  CType sum = agile::getScalarProduct(gradient[0], gradient[0]);
  sum += agile::getScalarProduct(zTemp, zTemp);
  RType nKx = std::sqrt(std::abs(sum));
  RType nx = agile::norm2(extDiff);

  Log("nKx: %.4e nx: %.4e\n", nKx, nx);

  PDRecon::AdaptStepSize(nKx, nx);
  Log("new sigma: %.4e new tau: %.4e\n", params.sigma, params.tau);
}

void TVTEMP::IterativeReconstruction(CVector &data_gpu, CVector &x, CVector &b1_gpu)
{
  unsigned N = width * height * frames;

  Log("Setting dt: %.3e\n", params.dt);
  Log("Setting Primal-Dual Gap of %.3e  as stopping criterion \n", params.stopPDGap);

  // primal
  CVector x_old(N);
  CVector ext(N);

  agile::copy(x, ext);

  // dual
  std::vector<CVector> y;
  y.push_back(CVector(N));
  y[0].assign(N, 0);

  std::vector<CVector> tempGradient;
  tempGradient.push_back(CVector(N));

  CVector z(data_gpu.size());
  zTemp.resize(data_gpu.size(), 0.0);
  z.assign(z.size(), 0.0);

  CVector norm(N);

  unsigned loopCnt = 0; 
  // loop 
  Log("Starting iteration\n"); 
  while ( loopCnt < params.maxIt )
  {
    // dual ascent step
    utils::Gradient_temp(ext, tempGradient, width, height, params.dt);
    agile::addScaledVector(y[0], params.sigma, tempGradient[0], y[0]);

    mrOp->BackwardOperation(ext, zTemp, b1_gpu);
    agile::addScaledVector(z, params.sigma, zTemp, z);

    Log("Still everything ok\n");
    // Proximal mapping
    utils::ProximalMap1D(y, (DType)1.0);

    agile::subScaledVector(z, params.sigma, data_gpu, z);
    agile::scale((DType)(1.0 / (1.0 + params.sigma / params.lambda)), z, z);

    // primal descent
    mrOp->ForwardOperation(z, imgTemp, b1_gpu);
    utils::Divergence_temp (y, divTemp, width, height, frames, params.dt);
    agile::subVector(imgTemp, divTemp, divTemp);
    agile::subScaledVector(x, params.tau, divTemp, ext);

    // save x_n+1
    agile::copy(ext, x_old);

    // extra gradient
    agile::scale((DType)2.0, ext, ext);
    agile::subVector(ext, x, ext);

    // x_n = x_n+1
    agile::copy(x_old, x);

    // adapt step size
    if (loopCnt < 10 || (loopCnt % 50 == 0))
    {
      CVector temp(N);
      agile::subVector(ext, x, temp);
      AdaptStepSize(temp, b1_gpu);
    }
    
    // compute PD Gap (export,verbose,stopping)
    if ( (verbose && (loopCnt < 10 || (loopCnt % 50 == 0)) ) ||
         ((debug) && (loopCnt % debugstep == 0)) || 
         ((params.stopPDGap > 0) && (loopCnt % 20 == 0)) )
    {
      RType pdGap =
            ComputePDGap(x, y, z, data_gpu, b1_gpu);
      pdGap=pdGap/N;
 
      pdGapExport.push_back( pdGap );
      Log("Normalized Primal-Dual Gap after %d iterations: %.4e\n", loopCnt, pdGap);     
      
      if ( pdGap < params.stopPDGap )
        return;
    }

    loopCnt++;
    if (loopCnt % 10 == 0)
      std::cout << "." << std::flush;
  }
  std::cout << std::endl;
}

RType TVTEMP::ComputeGStar(CVector &x, std::vector<CVector> &y, CVector &z,
                       CVector &data_gpu, CVector &b1_gpu)
{
  // F(Kx)
  zTemp.resize(data_gpu.size(), 0.0);
  mrOp->BackwardOperation(x, zTemp, b1_gpu);
  agile::subVector(zTemp, data_gpu, zTemp);

  RType g1 = 0.5 * params.lambda * std::real(agile::getScalarProduct(zTemp, zTemp));

  // F*(z)
  RType g2 = std::real(agile::getScalarProduct(data_gpu, z));
  g2 += (RType)1.0 / ((RType)2.0 * params.lambda) * std::real(agile::getScalarProduct(z, z));

  // G*(-Kx)
  mrOp->ForwardOperation(z, imgTemp, b1_gpu);
  utils::Divergence_temp(y, divTemp, width, height, frames, params.dt);
  agile::subVector(imgTemp, divTemp, divTemp);
  RType g3 = agile::norm1(divTemp);

  RType gstar = (RType)g1 + (RType)g2 + (RType)g3;
  return gstar;
}

RType TVTEMP::ComputePDGap(CVector &x, std::vector<CVector> &y, CVector &z,
                       CVector &data_gpu, CVector &b1_gpu)
{
  RType gstar = this->ComputeGStar(x, y, z, data_gpu, b1_gpu);
  RType tvNorm =
      utils::TVNorm(x, width, height, params.dx, params.dy, params.dt);
  RType PDGap = std::abs(gstar + tvNorm);
  return PDGap;
}

void TVTEMP::ExportAdditionalResults(const char *outputDir,
                                 ResultExportCallback callback)
{
    if (debug)
    {
        CVector pdGapExportGPU( pdGapExport.size() );
        pdGapExportGPU.assignFromHost(pdGapExport.begin(),pdGapExport.end());

        (*callback)(outputDir, "PDGap", pdGapExportGPU);
    }
  // TODO
}

