#include "../include/asltgv2recon4d.h"
#include "agile/agile.hpp"
#include "agile/io/file.hpp"
#include "agile/gpu_timer.hpp"
#include "agile/io/dicom.hpp"

/*CType RandomNumber () 
{ 
  CType x;
  x = CType(std::rand()/(RAND_MAX+0.0f)*2.0f,std::rand()/(RAND_MAX+0.0f)*2.0f); 
  //std::cout << "Test function:" << x << std::endl;
  return x; //CType(std::rand()/(RAND_MAX+0.0f)*255.0f,std::rand()/(RAND_MAX+0.0f)*255.0f); 
}*/

ASLTGV2RECON4D::ASLTGV2RECON4D(unsigned width, unsigned height, unsigned depth, unsigned coils,
           unsigned frames, BaseOperator *mrOp)
  : PDRecon(width, height, depth, coils, frames, mrOp)
{
  InitParams();
  InitTempVectors();
}

ASLTGV2RECON4D::ASLTGV2RECON4D(unsigned width, unsigned height, unsigned depth, unsigned coils,
            unsigned frames, ASLTGV2RECON4DParams &params, BaseOperator *mrOp)
  : PDRecon(width, height, depth, coils, frames, mrOp), params(params)
{
  // InitLambda(params.adaptLambdaParams.adaptLambda);
  InitTempVectors();
}

ASLTGV2RECON4D::~ASLTGV2RECON4D()
{
}

void ASLTGV2RECON4D::InitLambda(bool adaptLambda)
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

void ASLTGV2RECON4D::InitParams()
{
  params.maxIt = 1000;
  //params.stopPDGap = 0;

  params.sigma = 0.17;
  params.tau = 0.17;

  params.sigmaTauRatio = 1.0;

  params.dx = 1.0;
  params.dy = 1.0;
  params.dz = 1.0;
  params.dt = 1.0;

  params.alpha = 0.5;
  params.alpha0 = std::sqrt(2);
  params.alpha1 = 1.0;

  params.lambda_c = 1.0;
  params.lambda_l = 1.0;
  // InitLambda(true);
}

void ASLTGV2RECON4D::InitTempVectors()
{
  unsigned N = width * height * depth * frames;

  imgTemp = CVector(N);
  //zTemp = CVector(0);  //< resized at runtime
  div1Temp = CVector(N);
  div3Temp = CVector(N);

  for (unsigned cnt = 0; cnt < 4; cnt++)
  {
    div2Temp.push_back(CVector(N));
    y1Temp.push_back(CVector(N));
    y3Temp.push_back(CVector(N));
    y7Temp.push_back(CVector(N));
  }

  for (int cnt = 0; cnt < 10; cnt++)
  {
    y2Temp.push_back(CVector(N));
    //y4Temp.push_back(CVector(N));
    //y8Temp.push_back(CVector(N));
  }
}

PDParams &ASLTGV2RECON4D::GetParams()
{
  return params;
}

//<Ax,y> = <x,A*y>
void ASLTGV2RECON4D::TestAdjointness(CVector &b1)
{
  
  /* initialize random seed: */
  std::srand(time(NULL));

  unsigned N = width*height*depth*frames;

  //Forwad/Backward operator
  //A*y
  std::vector< std::complex<float> > y_fw_;
  CVector y_fw;
  CVector Ay_fw;
  Ay_fw.assign(N,0.0);

  for(int idx=0; idx < N*coils; idx++)
  {
    y_fw_.push_back(utils::RandomNumber());
  }
  y_fw.assignFromHost(y_fw_.begin(),y_fw_.end());
  mrOp->ForwardOperation(y_fw, Ay_fw, b1);
  
  //Ax
  std::vector< std::complex<float> > x_bw_;
  CVector x_bw;
  CVector Ax_bw;
  Ax_bw.assign(N*coils,0.0);

  for(int idx=0; idx < N; idx++)
  {
    x_bw_.push_back(utils::RandomNumber());
  }
  x_bw.assignFromHost(x_bw_.begin(),x_bw_.end());

  mrOp->BackwardOperation(x_bw,Ax_bw,b1);
  
  //<Ax,y> 
  RType s1 = std::real(agile::getScalarProduct(Ax_bw,y_fw));
  
  //<A*y,x>
  RType s2 = std::real(agile::getScalarProduct(Ay_fw,x_bw));
  
  std::cout << "################################################" << std::endl;
  std::cout << "Adjoint Test of Forward and Backward operator" << std::endl;
  std::cout << "S1:" << s1 << std::endl;
  std::cout << "S2:" << s2 << std::endl;
  std::cout << "Dif:" << std::abs(s1-s2) << std::endl;
  std::cout << "Rel-Dif:" << std::abs(s1-s2)/N << std::endl;


  //Gradient/Divergence
  std::vector<CVector> gx;
  std::vector<CVector> y;
  for (unsigned cnt = 0; cnt < 4; cnt++)
  {
    gx.push_back(CVector(N));
    gx[cnt].assign(N, 0.0);
    std::vector< std::complex<float> > temp_;
    CVector temp;
    for(int idx=0; idx < N; idx++)
    {
      temp_.push_back(utils::RandomNumber());
    }
    temp.assignFromHost(temp_.begin(),temp_.end());
    y.push_back(temp);
  }

  CVector x;
  CVector divy;
  divy.assign(N,0.0);

  std::vector< std::complex<float> > x_;

  for(int idx=0; idx < N; idx++)
  {
    x_.push_back(utils::RandomNumber());
  }
  x.assignFromHost(x_.begin(),x_.end());

  utils::Gradient4D(x, gx, width, height, depth, params.dx, params.dy, params.dz, params.dt);
  utils::Divergence4D(y, divy, temp, width, height, depth, frames, params.dx, params.dy,
                      params.dz, params.dt);
  agile::scale(-1.0f, divy, divy);
  RType s3 = std::real(agile::getScalarProduct(gx[0], y[0]));
  s3 += std::real(agile::getScalarProduct(gx[1], y[1]));
  s3 += std::real(agile::getScalarProduct(gx[2], y[2]));
  s3 += std::real(agile::getScalarProduct(gx[3], y[3]));
  RType s4 = std::real(agile::getScalarProduct(x, divy));

  std::cout << "################################################" << std::endl;
  std::cout << "Adjoint Test of Gradient and Divergence operator" << std::endl;
  std::cout << "S3:" << s3 << std::endl;
  std::cout << "S4:" << s4 << std::endl;
  std::cout << "Dif:" << std::abs(s3-s4) << std::endl;
  std::cout << "Rel-Dif:" << std::abs(s3-s4)/N << std::endl;

  //SymmetricGradient/SymmetricDivergence
  std::vector<CVector> gx_sym;
  std::vector<CVector> y_sym;
  for (unsigned cnt = 0; cnt < 10; cnt++)
  {
    gx_sym.push_back(CVector(N));
    gx_sym[cnt].assign(N, 0.0);
    std::vector< std::complex<float> > temp_;
    CVector temp;
    for(int idx=0; idx < N; idx++)
    {
      temp_.push_back(utils::RandomNumber());
    }
    temp.assignFromHost(temp_.begin(),temp_.end());
    y_sym.push_back(temp);
  }

  std::vector<CVector> Ay_sym;
  std::vector<CVector> x_sym;
  for (unsigned cnt = 0; cnt < 4; cnt++)
  {
    Ay_sym.push_back(CVector(N));
    Ay_sym[cnt].assign(N, 0.0);
    std::vector< std::complex<float> > temp_;
    CVector temp;
    for(int idx=0; idx < N; idx++)
    {
      temp_.push_back(utils::RandomNumber());
    }
    temp.assignFromHost(temp_.begin(),temp_.end());
    x_sym.push_back(temp);
  }
  
  utils::SymmetricGradient4D(x_sym, gx_sym, temp, width, height, depth, params.dx,
                           params.dy, params.dz, params.dt);
  utils::SymmetricDivergence4D(y_sym, Ay_sym, temp, width, height, depth, frames, params.dx, params.dy,
                      params.dz, params.dt);
  
  agile::scale(-1.0f, Ay_sym[0], Ay_sym[0]);
  agile::scale(-1.0f, Ay_sym[1], Ay_sym[1]);
  agile::scale(-1.0f, Ay_sym[2], Ay_sym[2]);
  agile::scale(-1.0f, Ay_sym[3], Ay_sym[3]);
  
  
  RType s5 = std::real(agile::getScalarProduct(gx_sym[0], y_sym[0]));
  s5 += std::real(agile::getScalarProduct(gx_sym[1], y_sym[1]));
  s5 += std::real(agile::getScalarProduct(gx_sym[2], y_sym[2]));
  s5 += std::real(agile::getScalarProduct(gx_sym[3], y_sym[3]));
  s5 += 2.0*std::real(agile::getScalarProduct(gx_sym[4], y_sym[4]));
  s5 += 2.0*std::real(agile::getScalarProduct(gx_sym[5], y_sym[5]));
  s5 += 2.0*std::real(agile::getScalarProduct(gx_sym[6], y_sym[6]));
  s5 += 2.0*std::real(agile::getScalarProduct(gx_sym[7], y_sym[7]));
  s5 += 2.0*std::real(agile::getScalarProduct(gx_sym[8], y_sym[8]));
  s5 += 2.0*std::real(agile::getScalarProduct(gx_sym[9], y_sym[9]));

  
  RType s6 = std::real(agile::getScalarProduct(Ay_sym[0], x_sym[0]));
  s6 += std::real(agile::getScalarProduct(Ay_sym[1], x_sym[1]));
  s6 += std::real(agile::getScalarProduct(Ay_sym[2], x_sym[2]));
  s6 += std::real(agile::getScalarProduct(Ay_sym[3], x_sym[3]));
  
  std::cout << "################################################" << std::endl;
  std::cout << "Adjoint Test of SymmetricGradient and SymmetricDivergence operator" << std::endl;
  std::cout << "S5:" << s5 << std::endl;
  std::cout << "S6:" << s6 << std::endl;
  std::cout << "Dif:" << std::abs(s5-s6) << std::endl;
  std::cout << "Rel-Dif:" << std::abs(s5-s6)/N << std::endl;
} 

void ASLTGV2RECON4D::AdaptStepSize(CVector &extDiff1, std::vector<CVector> &extDiff2,
                           CVector &extDiff3, std::vector<CVector> &extDiff4, std::vector<CVector> &extDiff5, CVector &b1_gpu) //added extDiff5
{
  unsigned N = extDiff1.size();
  CVector tempSum(N);
  tempSum.assign(N, 0.0);

  // compute gradients
  agile::subVector(extDiff1, extDiff3, imgTemp);

  // y1
  utils::Gradient4D(imgTemp, div2Temp, width, height, depth, params.dx, params.dy,
                  params.dz, params.dt);

  agile::subVector(div2Temp[0], extDiff2[0], div2Temp[0]);
  agile::subVector(div2Temp[1], extDiff2[1], div2Temp[1]);
  agile::subVector(div2Temp[2], extDiff2[2], div2Temp[2]);
  agile::subVector(div2Temp[3], extDiff2[3], div2Temp[3]);

  // abs(x).^2
  utils::SumOfSquares4(div2Temp, tempSum, temp);

  // y2
  utils::SymmetricGradient4D(extDiff2, y2Temp, temp, width, height, depth, params.dx,
                           params.dy, params.dz, params.dt);
  utils::SumOfSquares10(y2Temp, tempSum, temp);

  // y3
  utils::Gradient4D(extDiff3, div2Temp, width, height, depth, params.dx, params.dy,
                  params.dz, params.dt);

  agile::subVector(div2Temp[0], extDiff4[0], div2Temp[0]);
  agile::subVector(div2Temp[1], extDiff4[1], div2Temp[1]);
  agile::subVector(div2Temp[2], extDiff4[2], div2Temp[2]);
  agile::subVector(div2Temp[3], extDiff4[3], div2Temp[3]);

  utils::SumOfSquares4(div2Temp, tempSum, temp);

  // y4
  utils::SymmetricGradient4D(extDiff4, y2Temp, temp, width, height, depth, params.dx,
                           params.dy, params.dz, params.dt);

  utils::SumOfSquares10(y2Temp, tempSum, temp);
  
  // y7
  utils::Gradient4D(extDiff1, div2Temp, width, height, depth, params.dx, params.dy, params.dz, params.dt);
  
  agile::subVector(div2Temp[0], extDiff5[0], div2Temp[0]);
  agile::subVector(div2Temp[1], extDiff5[1], div2Temp[1]);
  agile::subVector(div2Temp[2], extDiff5[2], div2Temp[2]);
  agile::subVector(div2Temp[3], extDiff5[3], div2Temp[3]);
  
  utils::SumOfSquares4(div2Temp, tempSum, temp);
 
  // y8
  utils::SymmetricGradient4D(extDiff5, y2Temp, temp, width, height, depth, params.dx,
                           params.dy, params.dz, params.dt);

  utils::SumOfSquares10(y2Temp, tempSum, temp);
  
  // y5
  mrOp->BackwardOperation(extDiff1, y5Temp, b1_gpu);
  agile::multiplyConjElementwise(y5Temp, y5Temp, y5Temp);
  
  CType sum = agile::norm1(tempSum);

  sum += std::abs(agile::norm1(y5Temp));

  // y6
  mrOp->BackwardOperation(extDiff3, y5Temp, b1_gpu);
  agile::multiplyConjElementwise(y5Temp, y5Temp, y5Temp);

  sum += std::abs(agile::norm1(y5Temp));

  RType nKx = std::sqrt(std::abs(sum));

  agile::multiplyConjElementwise(extDiff1, extDiff1, tempSum);
  utils::SumOfSquares4(extDiff2, tempSum, temp);
  agile::multiplyConjElementwise(extDiff3, extDiff3, imgTemp);
  agile::addVector(tempSum, imgTemp, tempSum);
  utils::SumOfSquares4(extDiff4, tempSum, temp);
  utils::SumOfSquares4(extDiff5, tempSum, temp);

  sum = agile::norm1(tempSum);

  RType nx = std::sqrt(std::abs(sum));
  
  //std::cout << "nx:" << nx << " nkx: " << nKx << std::endl;
  Log("nKx: %.4e nx: %.4e\n", nKx, nx);

  PDRecon::AdaptStepSize(nKx, nx);
  Log("new sigma: %.4e new tau: %.4e\n", params.sigma, params.tau);
}


RType ASLTGV2RECON4D::ComputeDataFidelity(CVector &x1, CVector &data_gpu, CVector &b1_gpu, RType lambda)
{

    CVector tmp;
    tmp.resize(data_gpu.size(), 0.0);
    mrOp->BackwardOperation(x1, tmp, b1_gpu);
    agile::subVector(tmp,data_gpu,tmp);
    RType datafidelity;
    datafidelity = std::abs(agile::getScalarProduct(tmp,tmp));
    datafidelity *= lambda * (RType) 0.5;

    return datafidelity;
}


RType ASLTGV2RECON4D::ComputePDGap(std::vector<CVector> &y1, std::vector<CVector> &y2, std::vector<CVector> &y3, std::vector<CVector> &y4, CVector &y5, CVector &y6, std::vector<CVector> &y7, std::vector<CVector> &y8, CVector &data_gpu_c, CVector &data_gpu_l, CVector &b1_gpu)
{
    //Divergence = Divergence! nicht -Divergenz!
    unsigned N = width * height * depth * frames;
    CVector diff_tmp_pd1(N);
    diff_tmp_pd1.assign(N, 0.0);

    //F*(y)
    RType g1 = std::real(agile::getScalarProduct(data_gpu_c, y5));
    g1 += 1.0 / (2.0 * params.lambda_c) * std::real(agile::getScalarProduct(y5, y5));
    
    //F*(z)
    RType g2 = std::real(agile::getScalarProduct(data_gpu_l, y6));
    g2 += 1.0 / (2.0 * params.lambda_l) * std::real(agile::getScalarProduct(y6, y6));

    //G*(-K*y)
    RType g3 = 0;
    mrOp->ForwardOperation(y5, imgTemp, b1_gpu); //K*y
    utils::Divergence4D(y1, diff_tmp_pd1, temp, width, height, depth, frames, params.dx, params.dy, params.dz, params.dt); //div1(p2)
    agile::subVector(diff_tmp_pd1, imgTemp, imgTemp);
    utils::Divergence4D(y7, diff_tmp_pd1, temp, width, height, depth, frames, params.dx, params.dy, params.dz, params.dt); //div1(r2)
    agile::addVector(diff_tmp_pd1, imgTemp, imgTemp);
    g3 = agile::norm1(imgTemp);
  
    RType g4 = 0;
    mrOp->ForwardOperation(y6, imgTemp, b1_gpu); //K*z
    utils::Divergence4D(y3, diff_tmp_pd1, temp, width, height, depth, frames, params.dx, params.dy, params.dz, params.dt); //div1(q2)
    agile::subVector(diff_tmp_pd1, imgTemp, imgTemp);
    utils::Divergence4D(y1, diff_tmp_pd1, temp, width, height, depth, frames, params.dx, params.dy, params.dz, params.dt); //div1(p2)
    agile::subVector(imgTemp, diff_tmp_pd1, imgTemp);
    g4 = agile::norm1(imgTemp);

    //p2 + div2(p3)
    RType g5 = 0;
    utils::SymmetricDivergence4D(y2, div2Temp, temp, width, height, depth, params.dx, params.dy, params.dz, params.dt);
    for (unsigned cnt = 0; cnt < 4; cnt++)
    {
        agile::addVector(div2Temp[cnt], y1[cnt], div2Temp[cnt]);
        g5 += agile::norm1(div2Temp[cnt]);
    }

    //q2 + div2(q3)
    RType g6 = 0;
    utils::SymmetricDivergence4D(y4, div2Temp, temp, width, height, depth, params.dx, params.dy, params.dz, params.dt);
    for (unsigned cnt = 0; cnt < 4; cnt++)
    {
        agile::addVector(div2Temp[cnt], y3[cnt], div2Temp[cnt]);
        g6 += agile::norm1(div2Temp[cnt]);
    }


    //r2 + div2(r3)
    RType g7 = 0;
    utils::SymmetricDivergence4D(y8, div2Temp, temp, width, height, depth, params.dx, params.dy, params.dz, params.dt);
    for (unsigned cnt = 0; cnt < 4; cnt++)
    {
        agile::addVector(div2Temp[cnt], y7[cnt], div2Temp[cnt]);
        g7 += agile::norm1(div2Temp[cnt]);
    }

    RType gstar = (RType)g1 + (RType)g2 + (RType)g3 + (RType)g4 + (RType)g5 + (RType)g6 + (RType)g7;
    return gstar;
}

void ASLTGV2RECON4D::InitPrimalVectors(unsigned N)
{
  for (unsigned cnt = 0; cnt < 4; cnt++)
  {
    x2.push_back(CVector(N));
    x2[cnt].assign(N, 0.0);
    x4.push_back(CVector(N));
    x4[cnt].assign(N, 0.0);
    x5.push_back(CVector(N));
    x5[cnt].assign(N, 0.0);
  }

  ext1 = CVector(N);
  x1_old = CVector(N);
  ext3 = CVector(N);
  x3_old = CVector(N);
  for (unsigned cnt = 0; cnt < 4; cnt++)
  {
    ext2.push_back(CVector(N));
    x2_old.push_back(CVector(N));
    ext4.push_back(CVector(N));
    x4_old.push_back(CVector(N));
    ext5.push_back(CVector(N));
    x5_old.push_back(CVector(N));
  }
}

void ASLTGV2RECON4D::InitDualVectors(unsigned N)
{
  for (int cnt = 0; cnt < 4; cnt++)
  {
    y1.push_back(CVector(N));
    y3.push_back(CVector(N));
    y7.push_back(CVector(N));
    y1[cnt].assign(N, 0.0);
    y3[cnt].assign(N, 0.0);
    y7[cnt].assign(N, 0.0);
  }
  for (int cnt = 0; cnt < 10; cnt++)
  {
    y2.push_back(CVector(N));
    y4.push_back(CVector(N));
    y8.push_back(CVector(N));
    y2[cnt].assign(N, 0.0);
    y4[cnt].assign(N, 0.0);
    y8[cnt].assign(N, 0.0);
  }
  y5 = CVector(N*coils);
  y6 = CVector(N*coils);
  y5.assign(N*coils, 0);
  y6.assign(N*coils, 0);
}

void ASLTGV2RECON4D::IterativeReconstructionASL(CVector &data_gpu_l, CVector &data_gpu_c, CVector &x1, CVector &x3, CVector &b1_gpu)
{
  unsigned N = width * height * depth * frames;

  RType dx;
  dx = params.dx;
  ComputeTimeSpaceWeights(params.timeSpaceWeight, params.dx, params.dt);
  Log("Setting dx: %.3e, dt: %.3e\n", params.dx, params.dt);
 
  params.dy = params.dx * params.dy/dx;
  params.dz = params.dx * params.dz/dx;

  std::cout << "dx: " << params.dx << "dy: " << params.dy << "dz: " << params.dz << "dt: " << params.dt << std::endl;
   
  temp.resize(N, 0.0);
  norm_gpu.resize(N, 0.0);
  y5Temp.resize(data_gpu_l.size(),0.0);
  
  //Test adjointness
  //TestAdjointness(b1_gpu);

  // primal
  InitPrimalVectors(N);
  InitDualVectors(N);
  
  mrOp->ForwardOperation(data_gpu_c,x1,b1_gpu);
  mrOp->ForwardOperation(data_gpu_l,x3,b1_gpu);
  mrOp->BackwardOperation(x1,y5,b1_gpu);
  mrOp->BackwardOperation(x3,y6,b1_gpu);

  agile::copy(x1, ext1);
  agile::copy(x3, ext3);

  RType datafidelity_c;
  RType datafidelity_l;
  RType TGV_c_norm;
  RType TGV_l_norm;
  RType TGV_diff_norm;
  RType primal_cost;
  RType dual_cost;
  RType gap;

  unsigned loopCnt = 0;
  // loop
  Log("Starting iteration\n");
  while ( loopCnt < params.maxIt )
  {
    // dual ascent step
    // p2, q2, r2
    agile::subVector(ext1, ext3, imgTemp);
    // y2Temp = grad(c_ - l_)
    utils::Gradient4D(imgTemp, y1Temp, width, height, depth, params.dx, params.dy,
                    params.dz, params.dt);
    // y4Temp = grad(l_)
    utils::Gradient4D(ext3, y3Temp, width, height, depth, params.dx, params.dy,
                    params.dz, params.dt);
    // y7Temp = grad(c_) //added
    utils::Gradient4D(ext1, y7Temp, width, height, depth, params.dx, params.dy, params.dz, params.dt); //added
				
    for (unsigned cnt = 0; cnt < 4; cnt++)
    {
      // y1Temp = y1Temp - p1_
      agile::subVector(y1Temp[cnt], ext2[cnt], y1Temp[cnt]);
      // p2 = y1 = y1 + sigma*y1Temp
      agile::addScaledVector(y1[cnt], params.sigma, y1Temp[cnt], y1[cnt]);

      // y3Temp = y3Temp - q1_
      agile::subVector(y3Temp[cnt], ext4[cnt], y3Temp[cnt]);
      // q2 = y3 = y3 + sigma*y3Temp
      agile::addScaledVector(y3[cnt], params.sigma, y3Temp[cnt], y3[cnt]);
	  
      // y7Temp = y7Temp - r1_ //added
      agile::subVector(y7Temp[cnt], ext5[cnt], y7Temp[cnt]); //added
      // r2 = y7 = y7 + sigma*y7Temp //added
      agile::addScaledVector(y7[cnt], params.sigma, y7Temp[cnt], y7[cnt]); //added  
    }

    // p3, q3, r3
    // y2Temp = symgrad1(p1_)
    utils::SymmetricGradient4D(ext2, y2Temp, temp, width, height, depth, params.dx, params.dy,
                             params.dz, params.dt);

    for (unsigned cnt = 0; cnt < 10; cnt++)
    {
      // p3 = y2 = y2 + sigma*y2Temp
      agile::addScaledVector(y2[cnt], params.sigma, y2Temp[cnt], y2[cnt]);
    }

    // y4Temp = symgrad2(q1_)
    utils::SymmetricGradient4D(ext4, y2Temp, temp, width, height, depth, params.dx, params.dy,
                             params.dz, params.dt);

    for (unsigned cnt = 0; cnt < 10; cnt++)
    {
      // q3 = y4 = y4 + sigma*y4Temp
      agile::addScaledVector(y4[cnt], params.sigma, y2Temp[cnt], y4[cnt]);
    }


    // y8Temp = symgrad2(r1_) //added
    utils::SymmetricGradient4D(ext5, y2Temp, temp, width, height, depth, params.dx, params.dy,
							 params.dz, params.dt); //added
							 
    for (unsigned cnt = 0; cnt < 10; cnt++)
    {
      // r3 = y8 = y8 + sigma*y8Temp //added
      agile::addScaledVector(y8[cnt], params.sigma, y2Temp[cnt], y8[cnt]); //added
    }


    // c2, l2 in recon form
    // y5Temp = MFc(ext1)
    y5Temp.assign(y5.size(),0.0);
    mrOp->BackwardOperation(ext1, y5Temp, b1_gpu);
    // y5 = y5 + sigma*y5Temp;
    agile::addScaledVector(y5, params.sigma, y5Temp, y5);
	
    // y6Temp = MFc(ext3) not necessary to use a second temp variable!
    y5Temp.assign(y5.size(),0.0);
    mrOp->BackwardOperation(ext3, y5Temp, b1_gpu);
    // y6 = y6 + sigma*y6Temp;
    agile::addScaledVector(y6, params.sigma, y5Temp, y6);

    // Proximal mapping
    RType denom = std::min(params.alpha, (RType)1.0 - params.alpha);
    // prox operator y1
    RType scale = params.alpha1 * ((1.0 - params.alpha) / denom);
    utils::ProximalMap4(y1, norm_gpu, temp, 1.0 / scale);

    // prox operator y2
    scale = params.alpha0 * ((1.0 - params.alpha) / denom);
    utils::ProximalMap10(y2, norm_gpu, temp, 1.0 / scale);

    // prox operator y3
    scale = params.alpha1 * ((params.alpha) / denom);
    utils::ProximalMap4(y3, norm_gpu, temp, 1.0 / scale);

    // prox operator y4
    scale = params.alpha0 * ((params.alpha) / denom);
    utils::ProximalMap10(y4, norm_gpu, temp, 1.0 / scale);

    // prox operator y5
    agile::subScaledVector(y5, params.sigma, data_gpu_c, y5);
    agile::scale((RType)(1.0 / (1.0 + params.sigma / params.lambda_c)), y5, y5);

    // prox operator y6
    agile::subScaledVector(y6, params.sigma, data_gpu_l, y6);
    agile::scale((RType)(1.0 / (1.0 + params.sigma / params.lambda_l)), y6, y6);
	
    // prox operator y7 
    scale = params.alpha1 * ((params.alpha) / denom); 
    utils::ProximalMap4(y7, norm_gpu, temp, 1.0 / scale); 

    // prox operator y8 
    scale = params.alpha0 * ((params.alpha) / denom); 
    utils::ProximalMap10(y8, norm_gpu, temp, 1.0 / scale); 

    // primal descent
    // ext1
    // div1Temp = diff1(p2)
    utils::Divergence4D(y1, div1Temp, temp, width, height, depth, frames, params.dx, params.dy,
                      params.dz, params.dt);
    // div3Temp = diff1(r2)
    utils::Divergence4D(y7, div3Temp, temp, width, height, depth, frames, params.dx, params.dy,
                      params.dz, params.dt); 	
    agile::addVector(div3Temp,div1Temp,div1Temp); 				  
    // ext1 = K*(y5) = K*(c2)
    mrOp->ForwardOperation(y5, ext1, b1_gpu);
    // ext1 = diff1(p2) + diff1(r2) - K*(c2)
    agile::subVector(div1Temp, ext1, div1Temp);
    // c = ext1 = x1 + tau*div1Temp
    agile::addScaledVector(x1, params.tau, div1Temp, ext1);

    // ext2
    // div2Temp = symdiv1(p3)
    utils::SymmetricDivergence4D(y2, div2Temp, temp, width, height, depth, frames, params.dx,
                               params.dy, params.dz, params.dt);
    // p1 = ext2 = x2 + tau*(y1 + div2Temp)
    for (unsigned cnt = 0; cnt < 4; cnt++)
    {
      agile::addVector(y1[cnt], div2Temp[cnt], div2Temp[cnt]);
      agile::addScaledVector(x2[cnt], params.tau, div2Temp[cnt], ext2[cnt]);
    }

    // ext3
    utils::Divergence4D(y3, div3Temp, temp, width, height, depth, frames, params.dx,
                      params.dy, params.dz, params.dt);
    //div1Temp = diff1(p2)
    utils::Divergence4D(y1, div1Temp, temp, width, height, depth, frames, params.dx, params.dy,
                      params.dz, params.dt);
    // div3Temp = div3Temp - div1Temp
    agile::subVector(div3Temp, div1Temp, div3Temp);
    // ext3 = K*(y6) = K*(l2)
    mrOp->ForwardOperation(y6, ext3, b1_gpu);
    // ext3 = diff1(p2) - K*(c2)
    agile::subVector(div3Temp, ext3, div3Temp);
    // l = ext3 = x3 + tau*div3Temp
    agile::addScaledVector(x3, params.tau, div3Temp, ext3);

    // ext4
    // div2Temp = symdiv2(q3)
    utils::SymmetricDivergence4D(y4, div2Temp, temp, width, height, depth, frames, params.dx,
                               params.dy, params.dz, params.dt);
    // q1 = ext4 = x4 + tau*(y3 + div2Temp)
    for (unsigned cnt = 0; cnt < 4; cnt++)
    {
      agile::addVector(y3[cnt], div2Temp[cnt], div2Temp[cnt]);
      agile::addScaledVector(x4[cnt], params.tau, div2Temp[cnt], ext4[cnt]);
    }
	
	// ext5 
    // div2Temp = symdiv2(r3) 
    utils::SymmetricDivergence4D(y8, div2Temp, temp, width, height, depth, frames, params.dx,
                               params.dy, params.dz, params.dt); 
    // q1 = ext5 = x5 + tau*(y7 + div2Temp)
    for (unsigned cnt = 0; cnt < 4; cnt++) 
    {
      agile::addVector(y7[cnt], div2Temp[cnt], div2Temp[cnt]); 
      agile::addScaledVector(x5[cnt], params.tau, div2Temp[cnt], ext5[cnt]); 
    }

    // save x_n+1
    agile::copy(ext1, x1_old);
    agile::copy(ext3, x3_old);
    for (unsigned cnt = 0; cnt < 4; cnt++)
    {
      agile::copy(ext2[cnt], x2_old[cnt]);
      agile::copy(ext4[cnt], x4_old[cnt]);
      agile::copy(ext5[cnt], x5_old[cnt]); 
    }

    // extra gradient
    agile::scale(RType(2.0), ext1, ext1);
    agile::scale(RType(2.0), ext3, ext3);
    agile::subVector(ext1, x1, ext1);
    agile::subVector(ext3, x3, ext3);
    // x_n = x_n+1
    agile::copy(x1_old, x1);
    agile::copy(x3_old, x3);

    for (unsigned cnt = 0; cnt < 4; cnt++)
    {
      agile::scale((DType)2.0, ext2[cnt], ext2[cnt]);
      agile::subVector(ext2[cnt], x2[cnt], ext2[cnt]);
      agile::copy(x2_old[cnt], x2[cnt]);

      agile::scale((DType)2.0, ext4[cnt], ext4[cnt]);
      agile::subVector(ext4[cnt], x4[cnt], ext4[cnt]);
      agile::copy(x4_old[cnt], x4[cnt]);
	  
      agile::scale((DType)2.0, ext5[cnt], ext5[cnt]);
      agile::subVector(ext5[cnt], x5[cnt], ext5[cnt]);
      agile::copy(x5_old[cnt], x5[cnt]);
    }

    // adapt step size
    if (loopCnt < 10 || (loopCnt % 50 == 0))
    {
      agile::subVector(ext1, x1, div1Temp);
      agile::subVector(ext3, x3, div3Temp);

      for (unsigned cnt = 0; cnt < 4; cnt++)
      {
        agile::subVector(ext2[cnt], x2[cnt], y1Temp[cnt]);
        agile::subVector(ext4[cnt], x4[cnt], y3Temp[cnt]);
	    agile::subVector(ext5[cnt], x5[cnt], y7Temp[cnt]); //added
      }

      AdaptStepSize(div1Temp, y1Temp, div3Temp, y3Temp, y7Temp, b1_gpu);
      std::cout << "Sigma: " << params.sigma << std::endl;
    }
     loopCnt++;
    if (loopCnt % 10 == 0)
      std::cout << "." << std::flush;
  }
}

void ASLTGV2RECON4D::ExportAdditionalResults(const char *outputDir,
                                     ResultExportCallback callback)
{
  /*(*callback)(outputDir, "x3_component", x3);

  if (debug)
  {
    CVector pdGapExportGPU(pdGapExport.size());
    pdGapExportGPU.assignFromHost(pdGapExport.begin(), pdGapExport.end());

    (*callback)(outputDir, "PDGap", pdGapExportGPU);
  }*/
  // TODO
  // other components?
}

