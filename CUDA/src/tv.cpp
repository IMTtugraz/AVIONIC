#include "../include/tv.h"
#include <stdexcept>
#include <math.h>

TV::TV(unsigned width, unsigned height, unsigned coils, unsigned frames,
       BaseOperator *mrOp)
  : PDRecon(width, height, coils, frames, mrOp)
{
  InitParams();
  InitTempVectors();
}

TV::TV(unsigned width, unsigned height, unsigned coils, unsigned frames,
       TVParams &params, BaseOperator *mrOp)
  : PDRecon(width, height, coils, frames, mrOp), params(params)
{
  InitLambda(params.adaptLambdaParams.adaptLambda);
  InitTempVectors();
}

TV::~TV()
{
}

void TV::InitLambda(bool adaptLambda)
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

void TV::InitParams()
{
  params.maxIt = 500;

  params.sigma = 1.0 / 3.0;
  params.tau = 1.0 / 3.0;

  params.sigmaTauRatio = 1.0;
  params.timeSpaceWeight = 5.0;

  params.ds = 1.0;
  params.dt = 1.0;
  params.adaptLambdaParams.k = 0.4 * 0.2991;
  params.adaptLambdaParams.d = 10.0 * 0.2991;
  InitLambda(true);
}

void TV::InitTempVectors()
{
  unsigned N = width * height * frames;

  imgTemp = CVector(N);
  divTemp = CVector(N);
  zTemp = CVector(0);  //< resized at runtime
}

PDParams &TV::GetParams()
{
  return params;
}

void TV::AdaptStepSize(CVector &extDiff, CVector &b1)
{
  std::vector<CVector> gradient =
      utils::Gradient(extDiff, width, height, params.ds, params.ds, params.dt);

  zTemp = mrOp->BackwardOperation(extDiff, b1);

  CType sum = agile::getScalarProduct(gradient[0], gradient[0]);
  sum += agile::getScalarProduct(gradient[1], gradient[1]);
  sum += agile::getScalarProduct(gradient[2], gradient[2]);
  sum += agile::getScalarProduct(zTemp, zTemp);
  RType nKx = std::sqrt(std::abs(sum));
  RType nx = agile::norm2(extDiff);

  Log("nKx: %.4e nx: %.4e\n", nKx, nx);

  PDRecon::AdaptStepSize(nKx, nx);
  Log("new sigma: %.4e new tau: %.4e\n", params.sigma, params.tau);
}

void TV::IterativeReconstruction(CVector &data_gpu, CVector &x, CVector &b1_gpu)
{
  unsigned N = width * height * frames;

  ComputeTimeSpaceWeights(params.timeSpaceWeight, params.ds, params.dt);
  Log("Setting ds: %.3e, dt: %.3e\n", params.ds, params.dt);

  // primal
  CVector x_old(N);
  CVector ext(N);

  agile::copy(x, ext);

  // dual
  std::vector<CVector> y;
  y.push_back(CVector(N));
  y.push_back(CVector(N));
  y.push_back(CVector(N));
  y[0].assign(N, 0);
  y[1].assign(N, 0);
  y[2].assign(N, 0);

  std::vector<CVector> tempGradient;
  tempGradient.push_back(CVector(N));
  tempGradient.push_back(CVector(N));
  tempGradient.push_back(CVector(N));

  CVector z(data_gpu.size());
  zTemp.resize(data_gpu.size(), 0.0);
  z.assign(z.size(), 0.0);

  CVector norm(N);

  unsigned loopCnt = 0;
  // loop
  Log("Starting iteration\n");
  while (loopCnt < params.maxIt)
  {
    // dual ascent step
    utils::Gradient(ext, tempGradient, width, height, params.ds, params.ds,
                    params.dt);
    agile::addScaledVector(y[0], params.sigma, tempGradient[0], y[0]);
    agile::addScaledVector(y[1], params.sigma, tempGradient[1], y[1]);
    agile::addScaledVector(y[2], params.sigma, tempGradient[2], y[2]);

    mrOp->BackwardOperation(ext, zTemp, b1_gpu);
    agile::addScaledVector(z, params.sigma, zTemp, z);

    // Proximal mapping
    utils::ProximalMap3(y, 1.0);

    agile::subScaledVector(z, params.sigma, data_gpu, z);
    agile::scale((float)(1.0 / (1.0 + params.sigma / params.lambda)), z, z);
    // primal descent
    mrOp->ForwardOperation(z, imgTemp, b1_gpu);
    utils::Divergence(y, divTemp, width, height, frames, params.ds, params.ds,
                      params.dt);
    agile::subVector(imgTemp, divTemp, divTemp);
    agile::subScaledVector(x, params.tau, divTemp, ext);

    // save x_n+1
    agile::copy(ext, x_old);

    // extra gradient
    agile::scale(2.0f, ext, ext);
    agile::subVector(ext, x, ext);

    // x_n = x_n+1
    agile::copy(x_old, x);

    // adapt step size
    if (loopCnt < 10 || (loopCnt % 50 == 0))
    {
      CVector temp(N);
      agile::subVector(ext, x, temp);
      AdaptStepSize(temp, b1_gpu);

      if (verbose)
      {
        RType pdGap = ComputePDGap(x, y, z, data_gpu, b1_gpu);
        Log("Normalized Primal-Dual Gap after %d iterations: %.4e\n", loopCnt, pdGap/N);
      }
    }
    
    // compute PD Gap for export
    if ((debug) && (loopCnt % debugstep == 0))
    {
        RType pdGap = ComputePDGap(x, y, z, data_gpu, b1_gpu);
        pdGapExport.push_back( pdGap/N );
    }

    loopCnt++;
    if (loopCnt % 10 == 0)
      std::cout << "." << std::flush;
  }
  std::cout << std::endl;
}

RType TV::ComputeGStar(CVector &x, std::vector<CVector> &y, CVector &z,
                       CVector &data_gpu, CVector &b1_gpu)
{
  // F(Kx)
  zTemp.resize(data_gpu.size(), 0.0);
  mrOp->BackwardOperation(x, zTemp, b1_gpu);
  agile::subVector(zTemp, data_gpu, zTemp);

  RType g1 = 0.5 * params.lambda * std::real(agile::getScalarProduct(zTemp, zTemp));

  // F*(z)
  RType g2 = std::real(agile::getScalarProduct(data_gpu, z));
  g2 += 1.0 / (2.0 * params.lambda) * std::real(agile::getScalarProduct(z, z));

  // G*(-Kx)
  mrOp->ForwardOperation(z, imgTemp, b1_gpu);
  utils::Divergence(y, divTemp, width, height, frames, params.ds, params.ds,
                    params.dt);
  agile::subVector(imgTemp, divTemp, divTemp);
  RType g3 = agile::norm1(divTemp);

  RType gstar = (RType)g1 + (RType)g2 + (RType)g3;
  return gstar;
}

RType TV::ComputePDGap(CVector &x, std::vector<CVector> &y, CVector &z,
                       CVector &data_gpu, CVector &b1_gpu)
{
  RType gstar = this->ComputeGStar(x, y, z, data_gpu, b1_gpu);
  RType tvNorm =
      utils::TVNorm(x, width, height, params.ds, params.ds, params.dt);
  RType PDGap = std::abs(gstar + tvNorm);
  return PDGap;
}

void TV::ExportAdditionalResults(const char *outputDir,
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

