#include "../include/ictgv2.h"

ICTGV2::ICTGV2(unsigned width, unsigned height, unsigned coils, unsigned frames,
               BaseOperator *mrOp)
  : PDRecon(width, height, 0, coils, frames, mrOp)
{
  InitParams();
  InitTempVectors();
}

ICTGV2::ICTGV2(unsigned width, unsigned height, unsigned coils, unsigned frames,
               ICTGV2Params &params, BaseOperator *mrOp)
  : PDRecon(width, height, 0, coils, frames, mrOp), params(params)
{
  InitLambda(params.adaptLambdaParams.adaptLambda);
  InitTempVectors();
}

ICTGV2::~ICTGV2()
{
}

void ICTGV2::InitLambda(bool adaptLambda)
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

void ICTGV2::InitParams()
{
  params.maxIt = 500;
  params.stopPDGap = 0;

  params.sigma = 1.0 / 3.0;
  params.tau = 1.0 / 3.0;

  params.sigmaTauRatio = 1.0;

  params.timeSpaceWeight = 6.5;
  params.ds = 1.0;
  params.dt = 1.0;

  params.timeSpaceWeight2 = 2.5;
  params.ds2 = 1.0;
  params.dt2 = 1.0;

  params.alpha = 0.6423;
  params.alpha0 = std::sqrt(2);
  params.alpha1 = 1.0;

  params.adaptLambdaParams.k = 0.4 / (5.1191 * 0.35);
  params.adaptLambdaParams.d = 10.0 / (5.1191 * 0.35);
  InitLambda(true);
}

void ICTGV2::InitTempVectors()
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

PDParams &ICTGV2::GetParams()
{
  return params;
}

void ICTGV2::TestAdjointness(CVector &b1)
{
  //TODO: Implement
} 

void ICTGV2::AdaptStepSize(CVector &extDiff1, std::vector<CVector> &extDiff2,
                           CVector &extDiff3, std::vector<CVector> &extDiff4,
                           CVector &b1)
{
  unsigned N = extDiff1.size();
  CVector tempSum(N);
  tempSum.assign(N, 0);

  // compute gradients
  agile::subVector(extDiff1, extDiff3, imgTemp);

  // y1
  utils::Gradient(imgTemp, y2Temp, width, height, params.ds, params.ds,
                  params.dt);

  agile::subVector(y2Temp[0], extDiff2[0], y2Temp[0]);
  agile::subVector(y2Temp[1], extDiff2[1], y2Temp[1]);
  agile::subVector(y2Temp[2], extDiff2[2], y2Temp[2]);

  // abs(x).^2
  utils::SumOfSquares3(y2Temp, tempSum);

  // y2
  utils::SymmetricGradient(extDiff2, y2Temp, width, height, params.ds,
                           params.ds, params.dt);
  utils::SumOfSquares6(y2Temp, tempSum);

  // y3
  utils::Gradient(extDiff3, y2Temp, width, height, params.ds2, params.ds2,
                  params.dt2);

  agile::subVector(y2Temp[0], extDiff4[0], y2Temp[0]);
  agile::subVector(y2Temp[1], extDiff4[1], y2Temp[1]);
  agile::subVector(y2Temp[2], extDiff4[2], y2Temp[2]);

  utils::SumOfSquares3(y2Temp, tempSum);

  utils::SymmetricGradient(extDiff4, y2Temp, width, height, params.ds2,
                           params.ds2, params.dt2);

  zTemp = mrOp->BackwardOperation(extDiff1, b1);

  utils::SumOfSquares6(y2Temp, tempSum);

  CType sum = agile::norm1(tempSum);
  agile::multiplyConjElementwise(zTemp, zTemp, zTemp);

  sum += std::abs(agile::norm1(zTemp));
  RType nKx = std::sqrt(std::abs(sum));

  agile::multiplyConjElementwise(extDiff1, extDiff1, tempSum);
  utils::SumOfSquares3(extDiff2, tempSum);
  agile::multiplyConjElementwise(extDiff3, extDiff3, imgTemp);
  agile::addVector(tempSum, imgTemp, tempSum);
  utils::SumOfSquares3(extDiff4, tempSum);

  sum = agile::norm1(tempSum);

  RType nx = std::sqrt(std::abs(sum));

  Log("nKx: %.4e nx: %.4e\n", nKx, nx);

  PDRecon::AdaptStepSize(nKx, nx);
  Log("new sigma: %.4e new tau: %.4e\n", params.sigma, params.tau);
}

RType ICTGV2::ComputeGStar(CVector &x1, std::vector<CVector> &y1,
                           std::vector<CVector> &y2, std::vector<CVector> &y3,
                           std::vector<CVector> &y4, CVector &z,
                           CVector &data_gpu, CVector &b1_gpu)
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
  utils::Divergence(y1, div1Temp, width, height, frames, params.ds, params.ds,
                    params.dt);
  agile::subVector(imgTemp, div1Temp, div1Temp);
  RType g3 = agile::norm1(div1Temp);

  utils::SymmetricDivergence(y2, div2Temp, width, height, frames, params.ds,
                             params.ds, params.dt);
  RType g4 = 0;
  for (unsigned cnt = 0; cnt < 3; cnt++)
  {
    // -y(:,:,:,1:3) - div3_6
    agile::addVector(div2Temp[cnt], y1[cnt], div2Temp[cnt]);
    g4 += agile::norm1(div2Temp[cnt]);
  }

  utils::Divergence(y1, div1Temp, width, height, frames, params.ds, params.ds,
                    params.dt);
  utils::Divergence(y3, div3Temp, width, height, frames, params.ds2, params.ds2,
                    params.dt2);
  agile::subVector(div1Temp, div3Temp, div1Temp);
  RType g5 = agile::norm1(div1Temp);

  utils::SymmetricDivergence(y4, div2Temp, width, height, frames, params.ds2,
                             params.ds2, params.dt2);
  RType g6 = 0;
  for (unsigned cnt = 0; cnt < 3; cnt++)
  {
    // -y(:,:,:,10:12) - div3_6
    agile::addVector(div2Temp[cnt], y3[cnt], div2Temp[cnt]);
    g6 += agile::norm1(div2Temp[cnt]);
  }

  RType gstar =
      (RType)g1 + (RType)g2 + (RType)g3 + (RType)g4 + (RType)g5 + (RType)g6;
  return gstar;
}

RType ICTGV2::ComputePDGap(CVector &x1, std::vector<CVector> &x2, CVector &x3,
                           std::vector<CVector> &x4, std::vector<CVector> &y1,
                           std::vector<CVector> &y2, std::vector<CVector> &y3,
                           std::vector<CVector> &y4, CVector &z,
                           CVector &data_gpu, CVector &b1_gpu)
{
  RType gstar = this->ComputeGStar(x1, y1, y2, y3, y4, z, data_gpu, b1_gpu);
  RType ictgv2Norm =
      utils::ICTGV2Norm(x1, x2, x3, x4, div2Temp, y2Temp, params.alpha0,
                        params.alpha1, params.alpha, width, height, params.ds,
                        params.dt, params.ds2, params.dt2);
  RType PDGap = std::abs(gstar + ictgv2Norm);
  return PDGap;
}

void ICTGV2::InitPrimalVectors(unsigned N)
{
  x3 = CVector(N);
  x3.assign(x3.size(), 0.0);
  for (unsigned cnt = 0; cnt < 3; cnt++)
  {
    x2.push_back(CVector(N));
    x2[cnt].assign(N, 0.0);
    x4.push_back(CVector(N));
    x4[cnt].assign(N, 0.0);
  }

  ext1 = CVector(N);
  x1_old = CVector(N);
  ext3 = CVector(N);
  x3_old = CVector(N);
  for (unsigned cnt = 0; cnt < 3; cnt++)
  {
    ext2.push_back(CVector(N));
    x2_old.push_back(CVector(N));
    ext4.push_back(CVector(N));
    x4_old.push_back(CVector(N));
  }
}

void ICTGV2::InitDualVectors(unsigned N)
{
  for (int cnt = 0; cnt < 3; cnt++)
  {
    y1.push_back(CVector(N));
    y3.push_back(CVector(N));
    y1[cnt].assign(N, 0.0);
    y3[cnt].assign(N, 0.0);
  }
  for (int cnt = 0; cnt < 6; cnt++)
  {
    y2.push_back(CVector(N));
    y4.push_back(CVector(N));
    y2[cnt].assign(N, 0);
    y4[cnt].assign(N, 0);
  }
}

void ICTGV2::IterativeReconstruction(CVector &data_gpu, CVector &x1,
                                     CVector &b1_gpu)
{
  unsigned N = width * height * frames;
  ComputeTimeSpaceWeights(params.timeSpaceWeight, params.ds, params.dt);
  Log("Setting ds: %.3e, dt: %.3e\n", params.ds, params.dt);
  ComputeTimeSpaceWeights(params.timeSpaceWeight2, params.ds2, params.dt2);
  Log("Setting ds2: %.3e, dt2: %.3e\n", params.ds2, params.dt2);
  Log("Setting Primal-Dual Gap of %.3e  as stopping criterion \n", params.stopPDGap);

  // primal
  InitPrimalVectors(N);
  agile::copy(x1, ext1);

  // dual
  InitDualVectors(N);

  CVector z(data_gpu.size());
  zTemp.resize(data_gpu.size(), 0.0);
  z.assign(z.size(), 0.0);

  unsigned loopCnt = 0;
  // loop
  Log("Starting iteration\n");
  while ( loopCnt < params.maxIt )
  {
    // dual ascent step
    // p, r
    agile::subVector(ext1, ext3, imgTemp);
    utils::Gradient(imgTemp, y2Temp, width, height, params.ds, params.ds,
                    params.dt);
    utils::Gradient(ext3, y4Temp, width, height, params.ds2, params.ds2,
                    params.dt2);
    for (unsigned cnt = 0; cnt < 3; cnt++)
    {
      agile::subVector(y2Temp[cnt], ext2[cnt], y2Temp[cnt]);
      agile::addScaledVector(y1[cnt], params.sigma, y2Temp[cnt], y1[cnt]);

      agile::subVector(y4Temp[cnt], ext4[cnt], y4Temp[cnt]);
      agile::addScaledVector(y3[cnt], params.sigma, y4Temp[cnt], y3[cnt]);
    }

    // q, s
    utils::SymmetricGradient(ext2, y2Temp, width, height, params.ds, params.ds,
                             params.dt);
    utils::SymmetricGradient(ext4, y4Temp, width, height, params.ds2,
                             params.ds2, params.dt2);
    for (unsigned cnt = 0; cnt < 6; cnt++)
    {
      agile::addScaledVector(y2[cnt], params.sigma, y2Temp[cnt], y2[cnt]);
      agile::addScaledVector(y4[cnt], params.sigma, y4Temp[cnt], y4[cnt]);
    }

    mrOp->BackwardOperation(ext1, zTemp, b1_gpu);
    agile::addScaledVector(z, params.sigma, zTemp, z);

    // Proximal mapping
    RType denom = std::min(params.alpha, (RType)1.0 - params.alpha);
    RType scale = params.alpha1 * (params.alpha / denom);
    utils::ProximalMap3(y1, 1.0 / scale);

    // prox operator y2
    scale = params.alpha0 * (params.alpha / denom);
    utils::ProximalMap6(y2, 1.0 / scale);

    // prox operator y3
    scale = params.alpha1 * ((1.0 - params.alpha) / denom);
    utils::ProximalMap3(y3, 1.0 / scale);

    // prox operator y4
    scale = params.alpha0 * ((1.0 - params.alpha) / denom);
    utils::ProximalMap6(y4, 1.0 / scale);

    agile::subScaledVector(z, params.sigma, data_gpu, z);
    agile::scale((float)(1.0 / (1.0 + params.sigma / params.lambda)), z, z);

    // primal descent
    // ext1
    mrOp->ForwardOperation(z, imgTemp, b1_gpu);
    utils::Divergence(y1, div1Temp, width, height, frames, params.ds, params.ds,
                      params.dt);
    agile::subVector(imgTemp, div1Temp, imgTemp);
    agile::subScaledVector(x1, params.tau, imgTemp, ext1);

    // ext2
    utils::SymmetricDivergence(y2, div2Temp, width, height, frames, params.ds,
                               params.ds, params.dt);
    for (unsigned cnt = 0; cnt < 3; cnt++)
    {
      agile::addVector(y1[cnt], div2Temp[cnt], div2Temp[cnt]);
      agile::addScaledVector(x2[cnt], params.tau, div2Temp[cnt], ext2[cnt]);
    }

    // ext3
    utils::Divergence(y3, div3Temp, width, height, frames, params.ds2,
                      params.ds2, params.dt2);
    agile::subVector(div1Temp, div3Temp, div3Temp);
    agile::subScaledVector(x3, params.tau, div3Temp, ext3);

    // ext4
    utils::SymmetricDivergence(y4, div2Temp, width, height, frames, params.ds2,
                               params.ds2, params.dt2);
    for (unsigned cnt = 0; cnt < 3; cnt++)
    {
      agile::addVector(y3[cnt], div2Temp[cnt], div2Temp[cnt]);
      agile::addScaledVector(x4[cnt], params.tau, div2Temp[cnt], ext4[cnt]);
    }

    // save x_n+1
    agile::copy(ext1, x1_old);
    agile::copy(ext3, x3_old);
    for (unsigned cnt = 0; cnt < 3; cnt++)
    {
      agile::copy(ext2[cnt], x2_old[cnt]);
      agile::copy(ext4[cnt], x4_old[cnt]);
    }

    // extra gradient
    agile::scale(2.0f, ext1, ext1);
    agile::scale(2.0f, ext3, ext3);
    agile::subVector(ext1, x1, ext1);
    agile::subVector(ext3, x3, ext3);
    // x_n = x_n+1
    agile::copy(x1_old, x1);
    agile::copy(x3_old, x3);

    for (unsigned cnt = 0; cnt < 3; cnt++)
    {
      agile::scale((DType)2.0, ext2[cnt], ext2[cnt]);
      agile::subVector(ext2[cnt], x2[cnt], ext2[cnt]);
      agile::copy(x2_old[cnt], x2[cnt]);

      agile::scale((DType)2.0, ext4[cnt], ext4[cnt]);
      agile::subVector(ext4[cnt], x4[cnt], ext4[cnt]);
      agile::copy(x4_old[cnt], x4[cnt]);
    }

    // adapt step size
    if (loopCnt < 10 || (loopCnt % 50 == 0))
    {
      agile::subVector(ext1, x1, div1Temp);
      agile::subVector(ext3, x3, div3Temp);

      for (unsigned cnt = 0; cnt < 3; cnt++)
      {
        agile::subVector(ext2[cnt], x2[cnt], div2Temp[cnt]);
        agile::subVector(ext4[cnt], x4[cnt], y4Temp[cnt]);
      }
      AdaptStepSize(div1Temp, div2Temp, div3Temp, y4Temp, b1_gpu);
    }

    // compute PD Gap (export,verbose,stopping)
    if ( (verbose && (loopCnt < 10 || (loopCnt % 50 == 0)) ) ||
         ((debug) && (loopCnt % debugstep == 0)) || 
         ((params.stopPDGap > 0) && (loopCnt % 20 == 0)) )
    {
      RType pdGap =
          ComputePDGap(x1, x2, x3, x4, y1, y2, y3, y4, z, data_gpu, b1_gpu);
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

void ICTGV2::ExportAdditionalResults(const char *outputDir,
                                     ResultExportCallback callback)
{
  (*callback)(outputDir, "x3_component", x3);

  if (debug)
  {
    CVector pdGapExportGPU(pdGapExport.size());
    pdGapExportGPU.assignFromHost(pdGapExport.begin(), pdGapExport.end());

    (*callback)(outputDir, "PDGap", pdGapExportGPU);
  }
  // TODO
  // other components?
}

