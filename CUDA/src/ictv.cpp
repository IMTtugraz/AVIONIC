#include "../include/ictv.h"

ICTV::ICTV(unsigned width, unsigned height, unsigned coils, unsigned frames,
               BaseOperator *mrOp)
  : PDRecon(width, height, 0, coils, frames, mrOp)
{
  InitParams();
  InitTempVectors();
}

ICTV::ICTV(unsigned width, unsigned height, unsigned coils, unsigned frames,
               ICTVParams &params, BaseOperator *mrOp)
  : PDRecon(width, height, 0, coils, frames, mrOp), params(params)
{
  InitLambda(params.adaptLambdaParams.adaptLambda);
  InitTempVectors();
}

ICTV::~ICTV()
{
}

void ICTV::InitLambda(bool adaptLambda)
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

void ICTV::InitParams()
{
  params.maxIt = 500;
  params.stopPDGap = 0;

  params.sigma = 1.0 / 3.0;
  params.tau = 1.0 / 3.0;

  params.sigmaTauRatio = 1.0;

  params.timeSpaceWeight = 4.0;
  params.dx = 1.0;
  params.dy = 1.0; 
  params.dt = 1.0;

  params.timeSpaceWeight2 = 0.5;
  params.dx2 = 1.0;
  params.dy2 = 1.0; 
  params.dt2 = 1.0;

  params.alpha = 0.5;
  params.alpha1 = 1.0;

  params.adaptLambdaParams.k = 0.4 / (5.1191 * 0.35);
  params.adaptLambdaParams.d = 10.0 / (5.1191 * 0.35);
  InitLambda(true);
}

void ICTV::InitTempVectors()
{
  unsigned N = width * height * frames;

  imgTemp = CVector(N);
  zTemp = CVector(0);  //< resized at runtime
  div1Temp = CVector(N);
  div3Temp = CVector(N);

  for (unsigned cnt = 0; cnt < 3; cnt++)
  {
    div2Temp.push_back(CVector(N));
  }

  for (int cnt = 0; cnt < 6; cnt++)
  {
    y2Temp.push_back(CVector(N));
    y4Temp.push_back(CVector(N));
  }
}

PDParams &ICTV::GetParams()
{
  return params;
}

void ICTV::TestAdjointness(CVector &b1)
{
  //TODO: Implement
} 

void ICTV::AdaptStepSize(  CVector &extDiff1, 
                           CVector &extDiff3, 
                           CVector &b1)
{
  //TODO: check
  unsigned N = extDiff1.size();
  CVector tempSum(N);
  tempSum.assign(N, 0);

  // compute gradients
  agile::subVector(extDiff1, extDiff3, imgTemp);

  // y1
  utils::Gradient(imgTemp, y2Temp, width, height, params.dx, params.dy,
                  params.dt);

  // abs(x).^2
  utils::SumOfSquares3(y2Temp, tempSum);

  // y3
  utils::Gradient(extDiff3, y2Temp, width, height, params.dx2, params.dy2,
                  params.dt2);

  utils::SumOfSquares3(y2Temp, tempSum);

  zTemp = mrOp->BackwardOperation(extDiff1, b1);
  agile::multiplyConjElementwise(zTemp, zTemp, zTemp);

  CType sum = std::abs(agile::norm1(zTemp));
  RType nKx = std::sqrt(std::abs(sum));

  agile::multiplyConjElementwise(extDiff1, extDiff1, tempSum);
  agile::multiplyConjElementwise(extDiff3, extDiff3, imgTemp);
  agile::addVector(tempSum, imgTemp, tempSum);
  sum = agile::norm1(tempSum);

  RType nx = std::sqrt(std::abs(sum));

  Log("nKx: %.4e nx: %.4e\n", nKx, nx);

  PDRecon::AdaptStepSize(nKx, nx);
  Log("new sigma: %.4e new tau: %.4e\n", params.sigma, params.tau);
}

RType ICTV::ComputeDataFidelity(CVector &x1, CVector &data_gpu, CVector &b1_gpu)
{

    CVector tmp;
    tmp.resize(data_gpu.size(), 0.0);
    mrOp->BackwardOperation(x1, tmp, b1_gpu);
    agile::subVector(tmp,data_gpu,tmp);
    RType datafidelity;
    datafidelity = agile::norm2(tmp);
    datafidelity *= params.lambda/ (RType) 2.0;

    return datafidelity;
}


RType ICTV::ComputeGStar( CVector &x1, std::vector<CVector> &y1,
                          std::vector<CVector> &y3,
                          CVector &z, CVector &data_gpu, CVector &b1_gpu)
{
  // F(Kx)
  zTemp.resize(data_gpu.size(), 0.0);
  mrOp->BackwardOperation(x1, zTemp, b1_gpu);
  agile::subVector(zTemp, data_gpu, zTemp);

  RType g1 =
      0.5 * params.lambda * std::real(agile::getScalarProduct(zTemp, zTemp));

  // F*(z)
  RType g2 = std::real(agile::getScalarProduct(data_gpu, z));
  g2 += 1.0 / (2.0 * params.lambda) * std::real(agile::getScalarProduct(z, z));

  // G*(-Kx)
  mrOp->ForwardOperation(z, imgTemp, b1_gpu);
  utils::Divergence(y1, div1Temp, width, height, frames, params.dx, params.dy,
                    params.dt);
  agile::subVector(imgTemp, div1Temp, div1Temp);
  RType g3 = agile::norm1(div1Temp);

  utils::Divergence(y1, div1Temp, width, height, frames, params.dx, params.dy,
                    params.dt);
  utils::Divergence(y3, div3Temp, width, height, frames, params.dx2, params.dy2,
                    params.dt2);
  agile::subVector(div1Temp, div3Temp, div1Temp);
  RType g5 = agile::norm1(div1Temp);

  RType gstar =
      (RType)g1 + (RType)g2 + (RType)g3 + (RType)g5;
  return gstar;
}

RType ICTV::ComputePDGap( CVector &x1, CVector &x3,
                          std::vector<CVector> &y1, std::vector<CVector> &y3,
                          CVector &z, CVector &data_gpu, CVector &b1_gpu)
{
  RType gstar = this->ComputeGStar(x1, y1, y3, z, data_gpu, b1_gpu);
  RType ictvNorm =
      utils::ICTVNorm(  x1, x3, 
                        params.alpha1, params.alpha,
                        width, height,
                        params.dx,  params.dy, params.dt,
                        params.dx2, params.dy2, params.dt2);
  RType PDGap = std::abs(gstar + ictvNorm);
  return PDGap;
}

void ICTV::InitPrimalVectors(unsigned N)
{
  x3 = CVector(N);
  x3.assign(x3.size(), 0.0);
 
  ext1 = CVector(N);
  x1_old = CVector(N);
  ext3 = CVector(N);
  x3_old = CVector(N);
}

void ICTV::InitDualVectors(unsigned N)
{
  for (int cnt = 0; cnt < 3; cnt++)
  {
    y1.push_back(CVector(N));
    y3.push_back(CVector(N));
    y1[cnt].assign(N, 0.0);
    y3[cnt].assign(N, 0.0);
  }
}

void ICTV::IterativeReconstruction(CVector &data_gpu, CVector &x1,
                                     CVector &b1_gpu)
{
  unsigned N = width * height * frames;
  Log("Initial dx: %.3e, dy: %.3e, dt: %.3e\n", params.dx, params.dy, params.dt); 
  ComputeTimeSpaceWeights(params.timeSpaceWeight, params.dx, params.dt);
  ComputeTimeSpaceWeights(params.timeSpaceWeight, params.dy, params.dt); 
  Log("Setting dx: %.3e, dy: %.3e, dt: %.3e\n", params.dx, params.dy, params.dt);
  Log("Initial dx2: %.3e, dy2: %.3e, dt2: %.3e\n", params.dx2, params.dy2, params.dt2);  
  ComputeTimeSpaceWeights(params.timeSpaceWeight2, params.dx2, params.dt2);
  ComputeTimeSpaceWeights(params.timeSpaceWeight2, params.dy2, params.dt2); 
  Log("Setting dx2: %.3e, dy2: %.3e, dt2: %.3e\n", params.dx2, params.dy2, params.dt2);
  Log("Setting Primal-Dual Gap of %.3e  as stopping criterion \n", params.stopPDGap);

  // primal
  InitPrimalVectors(N);
  agile::copy(x1, ext1);

  // dual
  InitDualVectors(N);

  CVector z(data_gpu.size());
  zTemp.resize(data_gpu.size(), 0.0);
  z.assign(z.size(), 0.0);

  RType datafidelity;
  
  // used for proximal mapping
  RType denom = std::min(params.alpha, (RType)1.0 - params.alpha);
 
  unsigned loopCnt = 0;
  // loop
  Log("Starting iteration\n");
  while ( loopCnt < params.maxIt )
  {

    //---------------------------------------------------------------------
    // dual ascent step
    //---------------------------------------------------------------------
    // p, r
    agile::subVector(ext1, ext3, imgTemp);
    utils::Gradient(imgTemp, y2Temp, width, height, params.dx, params.dy,
                    params.dt);
    utils::Gradient(ext3, y4Temp, width, height, params.dx2, params.dy2,
                    params.dt2);
    for (unsigned cnt = 0; cnt < 3; cnt++)
    {
      agile::addScaledVector(y1[cnt], params.sigma, y2Temp[cnt], y1[cnt]);

      agile::addScaledVector(y3[cnt], params.sigma, y4Temp[cnt], y3[cnt]);
    }

    mrOp->BackwardOperation(ext1, zTemp, b1_gpu);
    agile::addScaledVector(z, params.sigma, zTemp, z);

    // Proximal mapping
    //---------------------------------------------------------------------
    // prox operator y1 
    RType scale = params.alpha1 * (params.alpha / denom);
    utils::ProximalMap3(y1, 1.0 / scale);

    // prox operator y3
    scale = params.alpha1 * ((1.0 - params.alpha) / denom);
    utils::ProximalMap3(y3, 1.0 / scale);

    // prox operator z
    agile::subScaledVector(z, params.sigma, data_gpu, z);
    agile::scale((float)(1.0 / (1.0 + params.sigma / params.lambda)), z, z);

    //---------------------------------------------------------------------
    // primal descent
    //---------------------------------------------------------------------
    // ext1
    mrOp->ForwardOperation(z, imgTemp, b1_gpu);
    utils::Divergence(y1, div1Temp, width, height, frames, params.dx, params.dy,
                      params.dt);
    agile::subVector(imgTemp, div1Temp, imgTemp);
    agile::subScaledVector(x1, params.tau, imgTemp, ext1);

    // ext3
    utils::Divergence(y3, div3Temp, width, height, frames, params.dx2,
                      params.dy2, params.dt2);
    agile::subVector(div1Temp, div3Temp, div3Temp);
    agile::subScaledVector(x3, params.tau, div3Temp, ext3);

    // save x_n+1
    agile::copy(ext1, x1_old);
    agile::copy(ext3, x3_old);

    // extra gradient
    agile::scale(2.0f, ext1, ext1);
    agile::scale(2.0f, ext3, ext3);
    agile::subVector(ext1, x1, ext1);
    agile::subVector(ext3, x3, ext3);

    // x_n = x_n+1
    agile::copy(x1_old, x1);
    agile::copy(x3_old, x3);

    // adapt step size
    if (loopCnt < 10 || (loopCnt % 50 == 0))
    {
      agile::subVector(ext1, x1, div1Temp);
      agile::subVector(ext3, x3, div3Temp);

      AdaptStepSize(div1Temp, div3Temp, b1_gpu);
    }

    // compute PD Gap (export,verbose,stopping)
    if ( (verbose && (loopCnt < 10 || (loopCnt % 50 == 0)) ) ||
         ((debug) && (loopCnt % debugstep == 0)) || 
         ((params.stopPDGap > 0) && (loopCnt % 20 == 0)) )
    {
      RType pdGap =
          ComputePDGap(x1, x3, y1, y3, z, data_gpu, b1_gpu);
      pdGap=pdGap/N;
      pdGapExport.push_back( pdGap );
      Log("Normalized Primal-Dual Gap after %d iterations: %.4e\n", loopCnt, pdGap);     
 
      RType ictvNorm =
              utils::ICTVNorm(  x1, x3,
                                params.alpha1, params.alpha,
                                width, height,
                                params.dx,  params.dy,  params.dt,
                                params.dx2, params.dy2, params.dt2);
      ictvNormExport.push_back(ictvNorm);
     
      datafidelity = ComputeDataFidelity(x1,data_gpu,b1_gpu);
      dataFidelityExport.push_back(datafidelity);

      Log("Data-Fidelity: %.3e | ICTV norm: %.3e\n", datafidelity,ictvNorm);

      if ( pdGap < params.stopPDGap )
        return;
    }

     loopCnt++;
    if (loopCnt % 10 == 0)
      std::cout << "." << std::flush;
  }
  std::cout << std::endl;
}

void ICTV::ExportAdditionalResults(const char *outputDir,
                                     ResultExportCallback callback)
{
  (*callback)(outputDir, "x3_component", x3);

  if (debug)
  {
    CVector pdGapExportGPU(pdGapExport.size());
    pdGapExportGPU.assignFromHost(pdGapExport.begin(), pdGapExport.end());
    (*callback)(outputDir, "PDGap", pdGapExportGPU);
   
    CVector ictvNormExportGPU(ictvNormExport.size());
    ictvNormExportGPU.assignFromHost(ictvNormExport.begin(), ictvNormExport.end());
    (*callback)(outputDir, "ictvnorm", ictvNormExportGPU);

    CVector dataFidelityExportGPU(dataFidelityExport.size());
    dataFidelityExportGPU.assignFromHost(dataFidelityExport.begin(), dataFidelityExport.end());
    (*callback)(outputDir, "DATAfidelity", dataFidelityExportGPU);
 
  }
}

