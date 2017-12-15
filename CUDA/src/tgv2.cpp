#include "../include/tgv2.h"
TGV2::TGV2(unsigned width, unsigned height, unsigned coils, unsigned frames,
           BaseOperator *mrOp)
  : PDRecon(width, height, 0, coils, frames, mrOp)
{
  InitParams();
  InitTempVectors();
}

TGV2::TGV2(unsigned width, unsigned height, unsigned coils, unsigned frames,
           TGV2Params &params, BaseOperator *mrOp)
  : PDRecon(width, height, 0, coils, frames, mrOp), params(params)
{
  InitLambda(params.adaptLambdaParams.adaptLambda);
  InitTempVectors();
}

TGV2::~TGV2()
{
}

void TGV2::InitLambda(bool adaptLambda)
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

void TGV2::InitParams()
{
  params.maxIt = 500;
  params.stopPDGap = 0;

  params.sigma = 1.0 / 3.0;
  params.tau = 1.0 / 3.0;

  params.sigmaTauRatio = 1.0;
  params.timeSpaceWeight = 5.0;

  params.dx = 1.0;
  params.dy = 1.0; 
  params.dt = 1.0;

  params.alpha0 = std::sqrt(2);
  params.alpha1 = 1.0;
  params.adaptLambdaParams.k = 0.4 * 0.2991;
  params.adaptLambdaParams.d = 10.0 * 0.2991;
  InitLambda(true);
}

void TGV2::InitTempVectors()
{
  unsigned N = width * height * frames;
  imgTemp = CVector(N);
  zTemp = CVector(0);  //< resized at runtime

  div1Temp = CVector(N);
  for (unsigned cnt = 0; cnt < 3; cnt++)
  {
    div2Temp.push_back(CVector(N));
    y1Temp.push_back(CVector(N));
  }

  for (int cnt = 0; cnt < 6; cnt++)
  {
    y2Temp.push_back(CVector(N));
  }
}


PDParams &TGV2::GetParams()
{
  return params;
}

void TGV2::TestAdjointness(CVector &b1)
{
  //TODO: Implement
} 


void TGV2::AdaptStepSize(CVector &extDiff1, std::vector<CVector> &extDiff2,
                         CVector &b1)
{
  std::vector<CVector> gradient1 =
      utils::Gradient(extDiff1, width, height, params.dx, params.dy, params.dt);
  for (unsigned cnt = 0; cnt < 3; cnt++)
  {
    agile::subVector(gradient1[cnt], extDiff2[cnt], gradient1[cnt]);
  }

  std::vector<CVector> gradient2 = utils::SymmetricGradient(
      extDiff2, width, height, params.dx, params.dy, params.dt);

  zTemp = mrOp->BackwardOperation(extDiff1, b1);

  unsigned N = width * height * frames;
  CVector tempSum(N);
  tempSum.assign(N, 0.0);
  // abs(x).^2
  utils::SumOfSquares3(gradient1, tempSum);

  utils::SumOfSquares6(gradient2, tempSum);

  CType sum = agile::norm1(tempSum);
  agile::multiplyConjElementwise(zTemp, zTemp, zTemp);

  sum += std::abs(agile::norm1(zTemp));
  RType nKx = std::sqrt(std::abs(sum));

  agile::multiplyConjElementwise(extDiff1, extDiff1, tempSum);
  utils::SumOfSquares3(extDiff2, tempSum);

  sum = agile::norm1(tempSum);
  RType nx = std::sqrt(std::abs(sum));

  Log("nKx: %.4e nx: %.4e\n", nKx, nx);

  PDRecon::AdaptStepSize(nKx, nx);
  Log("new sigma: %.4e new tau: %.4e\n", params.sigma, params.tau);
}

RType TGV2::ComputeGStar(CVector &x, std::vector<CVector> &y1,
                         std::vector<CVector> &y2, CVector &z,
                         CVector &data_gpu, CVector &b1_gpu)
{
  // F(Kx)
  zTemp.resize(data_gpu.size(), 0.0);
  mrOp->BackwardOperation(x, zTemp, b1_gpu);
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

  utils::SymmetricDivergence(y2, div2Temp, width, height, frames, params.dx,
                             params.dy, params.dt);
  RType g4 = 0;
  for (unsigned cnt = 0; cnt < 3; cnt++)
  {
    // -y(:,:,:,1:3) - div3_6
    agile::addVector(div2Temp[cnt], y1[cnt], div2Temp[cnt]);
    g4 += agile::norm1(div2Temp[cnt]);
  }

  RType gstar = (RType)g1 + (RType)g2 + (RType)g3 + (RType)g4;
  return gstar;
}

RType TGV2::ComputePDGap(CVector &x1, std::vector<CVector> &x2,
                         std::vector<CVector> &y1, std::vector<CVector> &y2,
                         CVector &z, CVector &data_gpu, CVector &b1_gpu)
{
  RType gstar = this->ComputeGStar(x1, y1, y2, z, data_gpu, b1_gpu);
  RType tgv2Norm = utils::TGV2Norm(x1, x2, params.alpha0, params.alpha1, width,
                                   height, params.dx, params.dy, params.dt);
  RType PDGap = std::abs(gstar + tgv2Norm);
  return PDGap;
}



void TGV2::IterativeReconstruction(CVector &data_gpu, CVector &x1,
                                   CVector &b1_gpu)
{
  unsigned N = width * height * frames;
  Log("Init width: %.3e, heigth: %.3e nframes: %.3e\n",  (RType) width, (RType) height,(RType)  frames);

  Log("Setting dx old: %.3e, dy: %.3e, dt: %.3e tsw1: %.3e\n", params.dx, params.dy, params.dt, params.timeSpaceWeight);

  ComputeTimeSpaceWeights(params.timeSpaceWeight, params.dx, params.dt);
  ComputeTimeSpaceWeights(params.timeSpaceWeight, params.dy, params.dt);
  Log("Setting dx: %.3e, dy: %.3e, dt: %.3e tsw1: %.3e\n", params.dx, params.dy, params.dt, params.timeSpaceWeight);
  Log("Setting Primal-Dual Gap of %.3e  as stopping criterion \n", params.stopPDGap);

  // primal
  CVector ext1(N), x1_old(N);
  std::vector<CVector> ext2, x2_old;
  for (unsigned cnt = 0; cnt < 3; cnt++)
  {
    ext2.push_back(CVector(N));
    x2_old.push_back(CVector(N));
  }

   std::vector<CVector> x2;
   for (unsigned cnt = 0; cnt < 3; cnt++)
   {
     x2.push_back(CVector(N));
     x2[cnt].assign(N, 0.0);
   }
  agile::copy(x1, ext1);

  // dual
  std::vector<CVector> y1;
  for (int cnt = 0; cnt < 3; cnt++)
  {
    y1.push_back(CVector(N));
    y1[cnt].assign(N, 0.0);
  }

  std::vector<CVector> y2;
  for (int cnt = 0; cnt < 6; cnt++)
  {
    y2.push_back(CVector(N));
    y2[cnt].assign(N, 0.0);
  }

  CVector z(data_gpu.size());
  zTemp.resize(data_gpu.size(), 0.0);
  z.assign(z.size(), 0.0);



  unsigned loopCnt = 0;
  // loop ---------------------------------------------------------------------------------------
  Log("Starting iteration\n");
  while ( loopCnt < params.maxIt )
  {
    // dual ascent step
    // p
    utils::Gradient(ext1, y1Temp, width, height, params.dx, params.dy,
                    params.dt);

    for (unsigned cnt = 0; cnt < 3; cnt++)
    {
      agile::subVector(y1Temp[cnt], ext2[cnt], y1Temp[cnt]);
      agile::addScaledVector(y1[cnt], params.sigma, y1Temp[cnt], y1[cnt]);
    }

    // q
    utils::SymmetricGradient(ext2, y2Temp, width, height, params.dx, params.dy,
                             params.dt);
    for (unsigned cnt = 0; cnt < 6; cnt++)
    {
       agile::addScaledVector(y2[cnt], params.sigma, y2Temp[cnt], y2[cnt]);
    }
    mrOp->BackwardOperation(ext1, zTemp, b1_gpu);
    agile::addScaledVector(z, params.sigma, zTemp, z);


    utils::ProximalMap3(y1, 1.0 /  params.alpha1);
    utils::ProximalMap6(y2, 1.0 / params.alpha0);

    agile::subScaledVector(z, params.sigma, data_gpu, z);

    //agile::scale((float)((RType)1.0 / ((RType)1.0 + params.sigma / params.lambda)), z, z);
    agile::scale((float)(1.0 / (1.0 + params.sigma / params.lambda)), z, z);

    // primal descent
    // ext1
    mrOp->ForwardOperation(z, imgTemp, b1_gpu);


    utils::Divergence(y1, div1Temp, width, height, frames, params.dx, params.dy,
                      params.dt);


    agile::subVector(imgTemp, div1Temp, div1Temp);
    agile::subScaledVector(x1, params.tau, div1Temp, ext1);

    // ext2
    utils::SymmetricDivergence(y2, div2Temp, width, height, frames, params.dx,
                               params.dy, params.dt);
    for (unsigned cnt = 0; cnt < 3; cnt++)
    {
      agile::addVector(y1[cnt], div2Temp[cnt], div2Temp[cnt]);
      agile::addScaledVector(x2[cnt], params.tau, div2Temp[cnt], ext2[cnt]);
    }

    // save x_n+1
    agile::copy(ext1, x1_old);
    for (unsigned cnt = 0; cnt < 3; cnt++)
      agile::copy(ext2[cnt], x2_old[cnt]);

    // extra gradient
    agile::scale(2.0f, ext1, ext1);
    agile::subVector(ext1, x1, ext1);
    // x_n = x_n+1
    agile::copy(x1_old, x1);

    for (unsigned cnt = 0; cnt < 3; cnt++)
    {
      agile::scale((DType)2.0, ext2[cnt], ext2[cnt]);
      agile::subVector(ext2[cnt], x2[cnt], ext2[cnt]);
      agile::copy(x2_old[cnt], x2[cnt]);
    }

    // adapt step size
    if (loopCnt < 10 || (loopCnt % 50 == 0))
    {
      agile::subVector(ext1, x1, div1Temp);
      for (unsigned cnt = 0; cnt < 3; cnt++)
      {
        agile::subVector(ext2[cnt], x2[cnt], div2Temp[cnt]);
      }
      AdaptStepSize(div1Temp, div2Temp, b1_gpu);
    }
    
    // compute PD Gap (export,verbose,stopping)
    if ( (verbose && (loopCnt < 10 || (loopCnt % 50 == 0)) ) ||
         ((debug) && (loopCnt % debugstep == 0)) || 
         ((params.stopPDGap > 0) && (loopCnt % 20 == 0)) )
    {
      RType pdGap =
            ComputePDGap(x1, x2, y1, y2, z, data_gpu, b1_gpu);
      pdGap=pdGap/N;
      
      if ( pdGap < params.stopPDGap )
        return;

      pdGapExport.push_back( pdGap );
      Log("Normalized Primal-Dual Gap after %d iterations: %.4e\n", loopCnt, pdGap);     
    }

    loopCnt++;
    if (loopCnt % 10 == 0)
      std::cout << "." << std::flush;
  }
  // loop ---------------------------------------------------------------------------------------
  std::cout << std::endl;
}

void TGV2::ExportAdditionalResults(const char *outputDir,
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

