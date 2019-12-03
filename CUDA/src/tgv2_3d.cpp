#include "../include/tgv2_3d.h"

TGV2_3D::TGV2_3D(unsigned width, unsigned height, unsigned depth, unsigned coils,
           BaseOperator *mrOp)
  : PDRecon(width, height, depth, coils, 0, mrOp)
{
  InitParams();
}

TGV2_3D::TGV2_3D(unsigned width, unsigned height,  unsigned depth, unsigned coils,
           TGV2_3DParams &params, BaseOperator *mrOp)
  : PDRecon(width, height, depth, coils, 0, mrOp), params(params)
{
  InitLambda(params.adaptLambdaParams.adaptLambda);
  temp.assign(width*height*depth, 0.0);
  temp2.assign(width*height*depth, 0.0);
  //zTemp1.assign(width*height*depth * coils, 0.0);
  //g.assign(width*height*depth * coils, 0.0);
  imgTemp1.assign(width*height*depth, 0.0);
  for (unsigned cnt = 0; cnt < 3; cnt++)
  {
    divTemp2.push_back(CVector(width*height*depth));
  }
}

TGV2_3D::~TGV2_3D()
{
}

void TGV2_3D::TestAdjointness(CVector &b1)
{
/*
  // TODO: implement random numbers
  unsigned N = width * height * depth;
//
  std::vector<CVector> urand_h,vrand_h;
  for(int i=0; i<N; ++i) 
    urand_h.push_back( rand()%10 );  // u
  for(int i=0; i<N*coils; ++i) 
    vrand_h.push_back( rand()%10 ); // v
///
  CVector urand(N), vrand(N);
  urand.assign(N, 1.0);
  vrand.assign(N*coils,1.0);
  //urand.assignFromHost(urand_h.begin(),urand_h.end());
  //vrand.assignFromHost(vrand_h.begin(),vrand_h.end());
 
  //CVector temp1(N);
  CVector temp1 = mrOp->ForwardOperation(vrand, b1); // KHv
  CVector temp2 = mrOp->BackwardOperation(urand, b1); //Ku
 
  CType temp3=agile::getScalarProduct(vrand,temp2); 
  CType temp4=agile::getScalarProduct(urand,temp1); 
 
  std::cout << "test adjointness:" << (temp3-temp4) << std::endl; 
*/
}

void TGV2_3D::InitLambda(bool adaptLambda)
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

void TGV2_3D::InitParams()
{
  params.maxIt = 500;
  params.stopPDGap = 0;

  params.sigma = 1.0 / 3.0;
  params.tau = 1.0 / 3.0;

  params.sigmaTauRatio = 1.0;

  params.dx = 1.0;
  params.dy = 1.0;
  params.dz = 1.0;

  params.alpha0 = std::sqrt(3);
  params.alpha1 = 1.0;
  params.adaptLambdaParams.k = 0.4 * 0.2991;
  params.adaptLambdaParams.d = 10.0 * 0.2991;
  InitLambda(true);
}

PDParams &TGV2_3D::GetParams()
{
  return params;
}

void TGV2_3D::AdaptStepSize(CVector &extDiff1, std::vector<CVector> &extDiff2,
                         CVector &b1, std::vector<CVector> &gradient1, std::vector<CVector> &gradient2, CVector &temp)
{
  utils::Gradient(extDiff1, gradient1, width, height, params.dx, params.dy, params.dz);
  for (unsigned cnt = 0; cnt < 3; cnt++)
  {
    agile::subVector(gradient1[cnt], extDiff2[cnt], gradient1[cnt]);
  }

  utils::SymmetricGradient(extDiff2, gradient2, imgTemp1, width, height, params.dx, params.dy, params.dz);


  mrOp->BackwardOperation(extDiff1, temp, b1, imgTemp1);


  unsigned N = width * height * depth;
  temp2.assign(N, 0.0);
  // abs(x).^2
  utils::SumOfSquares3(gradient1, temp2, imgTemp1);

  utils::SumOfSquares6(gradient2, temp2, imgTemp1);

  CType sum = agile::norm1(temp2);
  agile::multiplyConjElementwise(temp, temp, temp);

  sum += std::abs(agile::norm1(temp));
  RType nKx = std::sqrt(std::abs(sum));

  agile::multiplyConjElementwise(extDiff1, extDiff1, temp2);
  utils::SumOfSquares3(extDiff2, temp2, imgTemp1);

  sum = agile::norm1(temp2);
  RType nx = std::sqrt(std::abs(sum));

  Log("nKx: %.4e nx: %.4e\n", nKx, nx);

  PDRecon::AdaptStepSize(nKx, nx);
  Log("new sigma: %.4e new tau: %.4e\n", params.sigma, params.tau);
}

RType TGV2_3D::ComputeGStar(CVector &x, std::vector<CVector> &y1,
                         std::vector<CVector> &y2, CVector &z,
                         CVector &data_gpu, CVector &b1_gpu)
{
  //unsigned N = width * height * depth;
  // F(Kx)
  //CVector zTemp(N * coils);
  //CVector g(N * coils);
  mrOp->BackwardOperation(x, zTemp1, b1_gpu, temp);
  agile::subVector(zTemp1, data_gpu, g);

  RType g1 = 0.5 * params.lambda * std::real(agile::getScalarProduct(g, g));

  // F*(z)
  RType g2 = std::real(agile::getScalarProduct(data_gpu, z));
  g2 += 1.0 / (2.0 * params.lambda) * std::real(agile::getScalarProduct(z, z));

  // G*(-Kx)
  //CVector imgTemp(N);
  //CVector divTemp(N);
  //std::vector<CVector> divTemp2;
  //for (unsigned cnt = 0; cnt < 3; cnt++)
  //{
  //  divTemp2.push_back(CVector(N));
  //}
  mrOp->ForwardOperation(z, imgTemp1, b1_gpu, temp);
  utils::Divergence(y1, temp2, temp, width, height, depth, params.dx, params.dy,
                    params.dz);
  agile::subVector(imgTemp1, temp2, temp2);
  RType g3 = agile::norm1(temp2);

  utils::SymmetricDivergence(y2, divTemp2, temp, width, height, depth, params.dx,
                             params.dy, params.dz);
  RType g4 = 0;
  for (unsigned cnt = 0; cnt < 3; cnt++)
  {
    // -y(:,:,:,1:3) - div3_6
    agile::addVector(divTemp2[cnt], y1[cnt], divTemp2[cnt]);
    g4 += agile::norm1(divTemp2[cnt]);
  }

  RType gstar = (RType)g1 + (RType)g2 + (RType)g3 + (RType)g4;
  return gstar;
}

RType TGV2_3D::ComputePDGap(CVector &x1, std::vector<CVector> &x2,
                         std::vector<CVector> &y1, std::vector<CVector> &y2,
                         CVector &z, CVector &data_gpu, CVector &b1_gpu)
{
  RType gstar = this->ComputeGStar(x1, y1, y2, z, data_gpu, b1_gpu);
  RType tgv2Norm = utils::TGV2Norm(x1, x2, params.alpha0, params.alpha1, width,
                                   height, params.dx, params.dy, params.dz);
  RType PDGap = std::abs(gstar + tgv2Norm);
  return PDGap;
}


void TGV2_3D::IterativeReconstruction(CVector &data_gpu, CVector &x1,
                                   CVector &b1_gpu)
{
  if (verbose)
    TestAdjointness(b1_gpu);

  unsigned N = width * height * depth;

  //TODO: compute for dx,dy,dz
  //ComputeTimeSpaceWeights(params.timeSpaceWeight, params.ds, params.dt);
  //Log("Setting ds: %.3e, dt: %.3e\n", params.ds, params.dt);
   Log("Setting Primal-Dual Gap of %.3e  as stopping criterion \n", params.stopPDGap);


  std::vector<CVector> x2;
  for (unsigned cnt = 0; cnt < 3; cnt++)
  {
    x2.push_back(CVector(N));
    x2[cnt].assign(N, 0.0);
  }

  // primal
  CVector ext1(N), x1_old(N);
  std::vector<CVector> ext2, x2_old;
  for (unsigned cnt = 0; cnt < 3; cnt++)
  {
    ext2.push_back(CVector(N));
    ext2[cnt].assign(N, 0.0);
    x2_old.push_back(CVector(N));
  }
  agile::copy(x1, ext1);
  // dual
  std::vector<CVector> y1;
  std::vector<CVector> y1Temp;
  for (int cnt = 0; cnt < 3; cnt++)
  {
    y1.push_back(CVector(N));
    y1[cnt].assign(N, 0.0);
    y1Temp.push_back(CVector(N));
  }
  std::vector<CVector> y2;
  std::vector<CVector> y2Temp;
  for (int cnt = 0; cnt < 6; cnt++)
  {
    y2.push_back(CVector(N));
    y2[cnt].assign(N, 0.0);
    y2Temp.push_back(CVector(N));
  }

  CVector z(data_gpu.size());
  CVector zTemp(data_gpu.size());
  z.assign(z.size(), 0.0);
  zTemp.assign(zTemp.size(), 0.0);
  zTemp1.assign(data_gpu.size(), 0.0);
  g.assign(data_gpu.size(), 0.0);

/*  CVector z(N * coils);
  CVector zTemp(N * coils);
  z.assign(N * coils, 0.0);
  zTemp.assign(N * coils, 0.0);*/

  CVector imgTemp(N);

  CVector div1Temp(N);
  std::vector<CVector> div2Temp;
  for (unsigned cnt = 0; cnt < 3; cnt++)
    div2Temp.push_back(CVector(N));

  unsigned loopCnt = 0; 
  // loop
  Log("Starting iteration\n");
  while (loopCnt < params.maxIt)
  {
    // dual ascent step
    // p

    utils::Gradient(ext1, y1Temp, width, height, params.dx, params.dy,
                    params.dz);
    for (unsigned cnt = 0; cnt < 3; cnt++)
    {
      agile::subVector(y1Temp[cnt], ext2[cnt], y1Temp[cnt]);
      agile::addScaledVector(y1[cnt], params.sigma, y1Temp[cnt], y1[cnt]);
    }
    // q
    utils::SymmetricGradient(ext2, y2Temp, temp, width, height, params.dx, params.dy,
                             params.dz);
    for (unsigned cnt = 0; cnt < 6; cnt++)
    {
      agile::addScaledVector(y2[cnt], params.sigma, y2Temp[cnt], y2[cnt]);
    }

    mrOp->BackwardOperation(ext1, zTemp, b1_gpu, temp);
    agile::addScaledVector(z, params.sigma, zTemp, z);
   
    // Proximal mapping
    utils::ProximalMap3(y1, temp, temp2, (DType)1.0 / params.alpha1);
    utils::ProximalMap6(y2, temp, temp2, (DType)1.0 / params.alpha0);

    agile::subScaledVector(z, params.sigma, data_gpu, z);
    agile::scale((DType)(1.0 / (1.0 + params.sigma / params.lambda)), z, z);
  
    // primal descent
    // ext1
    mrOp->ForwardOperation(z, imgTemp, b1_gpu, temp);
    utils::Divergence(y1, div1Temp, temp, width, height, depth, params.dx, params.dy,
                      params.dz);
    agile::subVector(imgTemp, div1Temp, div1Temp);
    agile::subScaledVector(x1, params.tau, div1Temp, ext1);
    // ext2
    utils::SymmetricDivergence(y2, div2Temp, temp, width, height, depth, params.dx,
                               params.dy, params.dz);
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
      AdaptStepSize(div1Temp, div2Temp, b1_gpu, y1Temp, y2Temp, zTemp);
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


/*
    // adapt step size
    if (loopCnt < 10 || (loopCnt % 50 == 0))
    {
      CVector temp1(N);
   
      agile::subVector(ext1, x1, temp1);
      std::vector<CVector> temp2;
      for (unsigned cnt = 0; cnt < 3; cnt++)
      {
        temp2.push_back(CVector(N));
        agile::subVector(ext2[cnt], x2[cnt], temp2[cnt]);
      }
      AdaptStepSize(temp1, temp2, b1_gpu);

      if (verbose)
      {
        //RType pdGap = 1.0;
        RType pdGap = ComputePDGap(x1, x2, y1, y2, z, data_gpu, b1_gpu);
        Log("Normalized Primal-Dual Gap after %d iterations: %.4e\n", loopCnt, pdGap/N);
      }  
    }
    
    // compute PD Gap for export
    if ((debug) && (loopCnt % debugstep == 0))
    {
        RType pdGap = ComputePDGap(x1, x2, y1, y2, z, data_gpu, b1_gpu);
        pdGapExport.push_back( pdGap/N );
    }
*/

    loopCnt++;
    if (loopCnt % 10 == 0)
      std::cout << "." << std::flush;
  }
  std::cout << std::endl;
}

void TGV2_3D::ExportAdditionalResults(const char *outputDir,
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

