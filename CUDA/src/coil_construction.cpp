#include "../include/coil_construction.h"
#include "../include/cg_forward_operation.h"

CoilConstruction::CoilConstruction(unsigned width, unsigned height,
                                   unsigned coils, unsigned frames)
  : PDRecon(width, height, coils, frames, mrOp)
{
  InitParams();
}

CoilConstruction::CoilConstruction(unsigned width, unsigned height,
                                   unsigned coils, unsigned frames,
                                   CoilConstructionParams &params)
  : PDRecon(width, height, coils, frames, mrOp), params(params)
{
}

CoilConstruction::~CoilConstruction()
{
}

void CoilConstruction::InitParams()
{
  params.uH1mu = 1E-5;

  params.uReg = 0.2;
  params.uNrIt = 100;

  params.uSigmaTauRatio = 1.0;
  params.uSigma = std::sqrt(1.0 / 12.0);
  params.uTau = std::sqrt(1.0 / 12.0);
  params.uSigma *= params.uSigmaTauRatio;
  params.uTau /= params.uSigmaTauRatio;
  params.uAlpha0 = std::sqrt(2.0);
  params.uAlpha1 = 1.0;

  params.b1Reg = 2;
  params.b1NrIt = 500;

  params.b1SigmaTauRatio = 1.0;
  params.b1Sigma = std::sqrt(1.0 / 8.0);
  params.b1Tau = std::sqrt(1.0 / 8.0);
  params.b1Sigma *= params.b1SigmaTauRatio;
  params.b1Tau /= params.b1SigmaTauRatio;

  params.b1FinalReg = 0.1;
  params.b1FinalNrIt = 1000;
}

PDParams &CoilConstruction::GetParams()
{
  return params;
}

unsigned CoilConstruction::FindMaximumSum(CVector &crec)
{
  RType sums[coils];
  RType maxSum = std::numeric_limits<RType>::min();
  unsigned maxInd = -1;

  for (unsigned cnt = 0; cnt < coils; cnt++)
  {
    unsigned cOff = cnt * width * height;
    sums[cnt] = agile::lowlevel::norm1(crec.data() + cOff, width * height);
    Log("Absolute sum of coil %d: %.4e\n", cnt, sums[cnt]);
    if (sums[cnt] > maxSum)
    {
      maxInd = cnt;
      maxSum = sums[cnt];
    }
  }
  return maxInd;
}

void CoilConstruction::UTGV2Recon(CVector &b10, CVector &crec0, CVector &x1,
                                  std::vector<unsigned> inds)
{
  RType nu = params.uReg;

  unsigned N = width * height;

  // primal
  std::vector<CVector> x2;
  CVector ext1(N), x1_old(N);
  std::vector<CVector> ext2, x2_old;
  // dual
  std::vector<CVector> y1;
  std::vector<CVector> yTemp;
  CVector div1Temp(N);

  // init gpu vectors
  for (unsigned cnt = 0; cnt < 2; cnt++)
  {
    x2.push_back(CVector(N));
    x2[cnt].assign(N, 0.0);

    ext2.push_back(CVector(N));
    x2_old.push_back(CVector(N));

    y1.push_back(CVector(N));
    y1[cnt].assign(N, 0.0);
    yTemp.push_back(CVector(N));
  }
  // set initial value of x1
  agile::copy(x1, ext1);

  std::vector<CVector> y2;
  std::vector<CVector> y2Temp;
  for (int cnt = 0; cnt < 3; cnt++)
  {
    y2.push_back(CVector(N));
    y2[cnt].assign(N, 0);
    y2Temp.push_back(CVector(N));
  }

  // prepare prox mappings
  CVector Lw(N);
  Lw.assign(N, 0);

  CVector crec(N);
  CVector b1(N);
  CVector sumB1(N);
  sumB1.assign(N, 0);
  CVector temp(N);

  // Loop over array of indices
  for (unsigned cnt = 0; cnt < inds.size(); cnt++)
  {
    utils::GetSubVector(b10, b1, inds[cnt], N);
    utils::GetSubVector(crec0, crec, inds[cnt], N);
    agile::multiplyConjElementwise(b1, crec, temp);
    agile::addVector(Lw, temp, Lw);

    agile::multiplyConjElementwise(b1, b1, temp);
    agile::addVector(sumB1, temp, sumB1);
  }

  CVector Ib(N);

  CVector onesVec(N);
  onesVec.assign(N, 1.0);

  agile::addScaledVector(onesVec, params.uTau, sumB1, sumB1);

  Ib.assign(N, 1.0);
  agile::divideElementwise(Ib, sumB1, Ib);

  unsigned maxIt = params.uNrIt;
  unsigned loopCnt = 0;

  // loop
  Log("UTGV2Recon: Starting iteration:\n");
  while (loopCnt < maxIt)
  {
    // dual ascent step
    // p
    utils::Gradient2D(ext1, yTemp, width, height);
    for (unsigned cnt = 0; cnt < 2; cnt++)
    {
      agile::subVector(yTemp[cnt], ext2[cnt], yTemp[cnt]);
      agile::addScaledVector(y1[cnt], params.uSigma, yTemp[cnt], y1[cnt]);
    }

    // q
    utils::SymmetricGradient2D(ext2, y2Temp, width, height);
    for (unsigned cnt = 0; cnt < 3; cnt++)
    {
      agile::addScaledVector(y2[cnt], params.uSigma, y2Temp[cnt], y2[cnt]);
    }

    // Proximal mapping
    utils::ProximalMap2D(y1, (DType)1.0 / (params.uAlpha1 * nu));
    utils::ProximalMap2DSym(y2, (DType)1.0 / (params.uAlpha0 * nu));

    // primal descent
    // ext1
    utils::Divergence2D(y1, div1Temp, width, height);
    agile::addScaledVector(x1, params.uTau, div1Temp, ext1);

    // ext2
    utils::SymmetricDivergence2D(y2, yTemp, width, height);
    for (unsigned cnt = 0; cnt < 2; cnt++)
    {
      agile::addVector(y1[cnt], yTemp[cnt], yTemp[cnt]);
      agile::addScaledVector(x2[cnt], params.uTau, yTemp[cnt], ext2[cnt]);
    }

    // Proximal mapping
    agile::addScaledVector(ext1, params.uTau, Lw, ext1);
    agile::multiplyElementwise(ext1, Ib, ext1);

    // save x_n+1
    agile::copy(ext1, x1_old);
    for (unsigned cnt = 0; cnt < 2; cnt++)
      agile::copy(ext2[cnt], x2_old[cnt]);

    // extra gradient
    agile::scale(2.0f, ext1, ext1);
    agile::subVector(ext1, x1, ext1);
    // x_n = x_n+1
    agile::copy(x1_old, x1);

    for (unsigned cnt = 0; cnt < 2; cnt++)
    {
      agile::scale((DType)2.0, ext2[cnt], ext2[cnt]);
      agile::subVector(ext2[cnt], x2[cnt], ext2[cnt]);
      agile::copy(x2_old[cnt], x2[cnt]);
    }

    loopCnt++;
    if (loopCnt % 10 == 0)
      Log(".");
  }
  Log("\n");
}

void CoilConstruction::B1Recon(CVector &u0, CVector &crec, CVector &x1,
                               unsigned coils, unsigned maxIt, RType mu)
{
  unsigned N = width * height * coils;
  // iterative recon

  // primal: x1
  x1.assign(N, 0);
  CVector ext1(N), x1_old(N);
  agile::copy(x1, ext1);

  // dual
  std::vector<CVector> y1;
  std::vector<CVector> y1Temp;
  for (int cnt = 0; cnt < 2; cnt++)
  {
    y1.push_back(CVector(N));
    y1[cnt].assign(N, 0.0);
    y1Temp.push_back(CVector(width * height));
  }

  CVector div1Temp(N);

  // prepare prox mappings
  // renormalize u0

  // find min/max function
  CVector u0Normed(u0.size());
  RVector u0Abs(u0.size());
  agile::absVector(u0, u0Abs);

  // TODO bad style!
  int mxInd;
  agile::maxElement(u0Abs, &mxInd);
  RVector maxEl(1);
  agile::lowlevel::get_content(u0Abs.data(), 1, 1, 0, mxInd - 1, maxEl.data(),
                               1, 1);
  std::vector<RType> maxElHost(1);
  maxEl.copyToHost(maxElHost);
  CType mx = maxElHost[0];
  agile::scale((RType)1.0 / mx, u0, u0Normed);

  CVector Lw(N);

  CVector Iu(N);
  Iu.assign(N, 1.0);

  CVector onesVec(width * height);
  onesVec.assign(width * height, 1.0);
  CVector sumU(width * height);

  // precompute: 1 + tau*abs(u0).^2
  agile::multiplyConjElementwise(u0Normed, u0Normed, sumU);
  agile::addScaledVector(onesVec, params.b1Tau, sumU, sumU);

  for (unsigned cnt = 0; cnt < coils; cnt++)
  {
    unsigned cOff = cnt * width * height;
    agile::lowlevel::multiplyConjElementwise(
        u0Normed.data(), crec.data() + cOff, Lw.data() + cOff, width * height);

    agile::lowlevel::divideElementwise(Iu.data() + cOff, sumU.data(),
                                       Iu.data() + cOff, width * height);
  }

  unsigned loopCnt = 0;

  // loop
  Log("B1Recon: Starting iteration:\n");
  while (loopCnt < maxIt)
  {
    // dual ascent step
    // p
    CVector ext1Temp(width * height);
    for (unsigned cnt = 0; cnt < coils; cnt++)
    {
      unsigned cOff = cnt * width * height;
      utils::GetSubVector(ext1, ext1Temp, cnt, width * height);
      utils::Gradient2D(ext1Temp, y1Temp, width, height);
      for (unsigned gradCnt = 0; gradCnt < 2; gradCnt++)
      {
        agile::lowlevel::addScaledVector(
            y1[gradCnt].data() + cOff, params.b1Sigma, y1Temp[gradCnt].data(),
            y1[gradCnt].data() + cOff, width * height);
        // Proximal mapping
        agile::lowlevel::scale((CType)1.0 /
                                   ((RType)1.0 + (RType)params.b1Sigma / mu),
                               y1[gradCnt].data() + cOff,
                               y1[gradCnt].data() + cOff, width * height);
      }
    }

    for (unsigned cnt = 0; cnt < coils; cnt++)
    {
      unsigned cOff = cnt * width * height;
      utils::GetSubVector(y1[0], y1Temp[0], cnt, width * height);
      utils::GetSubVector(y1[1], y1Temp[1], cnt, width * height);

      // primal descent
      // ext1
      utils::Divergence2D(y1Temp, div1Temp, width, height);
      agile::lowlevel::addScaledVector(x1.data() + cOff, params.b1Tau,
                                       div1Temp.data(), ext1.data() + cOff,
                                       width * height);
    }

    // Proximal mapping
    agile::addScaledVector(ext1, params.b1Tau, Lw, ext1);
    agile::multiplyElementwise(ext1, Iu, ext1);

    // save x_n+1
    agile::copy(ext1, x1_old);

    // extra gradient
    agile::scale(2.0f, ext1, ext1);
    agile::subVector(ext1, x1, ext1);
    // x_n = x_n+1
    agile::copy(x1_old, x1);

    loopCnt++;
    if (loopCnt % 10 == 0)
      Log(".");
  }
  Log("\n");

  // Normalize to 1
  CVector x1Norm(width * height);
  x1Norm.assign(width * height, 0);
  CVector tempNorm(width * height);
  for (unsigned cnt = 0; cnt < coils; cnt++)
  {
    unsigned cOff = cnt * width * height;
    agile::lowlevel::multiplyConjElementwise(x1.data() + cOff, x1.data() + cOff,
                                             tempNorm.data(), width * height);
    agile::addVector(x1Norm, tempNorm, x1Norm);
  }
  agile::sqrt(x1Norm, x1Norm);

  for (unsigned cnt = 0; cnt < coils; cnt++)
  {
    unsigned cOff = cnt * width * height;
    agile::lowlevel::divideElementwise(x1.data() + cOff, x1Norm.data(),
                                       x1.data() + cOff, width * height);
  }
}

void CoilConstruction::B1FromUH1(CVector &u, CVector &crec,
                                 communicator_type &com, CVector &b1)
{
  unsigned N = width * height;

  RVector crecReal(width * height * coils);
  agile::absVector(crec, crecReal);
  agile::scale(params.uH1mu, crecReal, crecReal);

  // generate a forward operator
  CVector muU(N);
  muU.assign(N, 0);
  agile::multiplyConjElementwise(u, u, muU);
  agile::scale(params.uH1mu, muU, muU);

  forward_type forward(com, muU, width, height);

  // generate a binary measure
  typedef agile::ScalarProductMeasure<communicator_type> measure_type;
  measure_type scalar_product(com);

  // generate the CG solver
  // compute the inverse using CG
  const double REL_TOLERANCE = 1e-12;
  const double ABS_TOLERANCE = 1e-12;
  const unsigned MAX_ITERATIONS = 800;

  agile::ConjugateGradient<communicator_type, forward_type, measure_type> cg(
      com, forward, scalar_product, REL_TOLERANCE, ABS_TOLERANCE,
      MAX_ITERATIONS);

  // rhs
  CVector y(N);
  RVector b1Norm(N);
  b1Norm.assign(N, 0);
  RVector temp(N);
  CVector x(N);

  // Reconstruct per coil
  for (unsigned cnt = 0; cnt < coils; cnt++)
  {
    unsigned cOff = N * cnt;
    y.assign(N, 0);
    agile::lowlevel::multiplyConjElementwise(u.data(), crecReal.data() + cOff,
                                             y.data(), N);

    // solve Ax = y using CG
    x.assign(N, 0);
    cg(y, x);
    utils::SetSubVector(x, b1, cnt, N);

    if (cg.convergence())
      Log("Coil %d CG converged in ", cnt);
    else
      Log("Coil %d Error: CG did not converge in ", cnt);

    Log("%d iterations\n "
        "Initial residual: %.4e\n "
        "Final residual: %.4e\n "
        "Ratio rho_k / rho_0: %.4e\n "
        "----------------------------------------------\n",
        cg.getIteration() + 1, cg.getRho0(), cg.getRho(),
        cg.getRho() / cg.getRho0());

    agile::multiplyConjElementwise(x, x, x);
    agile::real(x, temp);
    agile::addVector(b1Norm, temp, b1Norm);
  }

  agile::sqrt(b1Norm, b1Norm);

  // Normalize
  for (unsigned cnt = 0; cnt < coils; cnt++)
  {
    unsigned cOff = N * cnt;
    agile::lowlevel::divideElementwise(b1.data() + cOff, b1Norm.data(),
                                       b1.data() + cOff, N);
  }

  // Return abs(b1)
  agile::multiplyConjElementwise(b1, b1, b1);
  agile::sqrt(b1, b1);
}

void CoilConstruction::PerformCoilConstruction(CVector &kdata, CVector &u,
                                               CVector &b1,
                                               communicator_type &com)
{
  CVector u0(width * height);
  u0.assign(u0.size(), 0);

  CVector crec(width * height * coils);
  crec.assign(crec.size(), 0);

  TimeAveragedReconstruction(kdata, u0, crec);

  // TODO check if really abs(u0) shall be passed
  agile::multiplyConjElementwise(u0, u0, u0);
  agile::sqrt(u0, u0);

  CVector absb1(width * height * coils);
  B1FromUH1(u0, crec, com, absb1);

  // find absolute sum of each coil and maximum index
  std::vector<unsigned> inds;
  inds.push_back(FindMaximumSum(crec));
  Log("starting with coil no: %d\n", inds[0]);

  unsigned N = width * height;

  CVector b1Temp(N);
  utils::GetSubVector(absb1, b1Temp, inds[0], N);
  utils::SetSubVector(b1Temp, b1, inds[0], N);

  CVector uMax(N);
  uMax.assign(N, 0);

  UTGV2Recon(b1, crec, uMax, inds);
  utils::SetSubVector(uMax, u, 0, N);

  CVector w(N);
  w.assign(N, 0);
  CVector temp(N);
  CVector b1Max(N);
  CVector crecMax(width * height);
  RVector crecAbs1(N);
  RVector crecAbs2(N);
  CVector x1(N);
  RType maxSum = std::numeric_limits<RType>::min();
  unsigned maxInd = -1;

  for (unsigned cnt = 1; cnt < coils; cnt++)
  {
    utils::GetSubVector(absb1, b1Temp, inds[cnt - 1], N);
    agile::multiplyConjElementwise(b1Temp, b1Temp, b1Temp);
    agile::multiplyElementwise(w, w, w);
    agile::addVector(w, b1Temp, w);
    agile::sqrt(w, w);

    utils::GetSubVector(crec, crecMax, inds[cnt - 1], N);
    agile::absVector(crecMax, crecAbs1);

    // Get index of next b1 field
    maxSum = std::numeric_limits<RType>::min();
    for (unsigned b1Cnt = 0; b1Cnt < coils; b1Cnt++)
    {
      if (std::find(inds.begin(), inds.end(), b1Cnt) == inds.end())
      {
        utils::GetSubVector(crec, crecMax, b1Cnt, N);
        agile::absVector(crecMax, crecAbs2);
        agile::multiplyElementwise(crecAbs1, crecAbs2, crecAbs2);
        RType norm = agile::norm1(crecAbs2);
        if (norm > maxSum)
        {
          maxSum = norm;
          maxInd = b1Cnt;
        }
      }
    }
    Log("going for coil no: %d\n", maxInd);
    inds.push_back(maxInd);

    // Get new b1
    utils::GetSubVector(absb1, b1Max, maxInd, N);
    utils::GetSubVector(crec, crecMax, maxInd, N);
    // w * absb1 * u
    utils::GetSubVector(u, uMax, cnt - 1, N);
    agile::multiplyElementwise(b1Max, uMax, uMax);
    agile::multiplyElementwise(uMax, w, uMax);
    agile::multiplyElementwise(crecMax, w, crecMax);

    x1.assign(N, 0.0);
    B1Recon(uMax, crecMax, x1, 1, params.b1NrIt, params.b1Reg);

    // Get Phase
    RVector angleReal(N);
    CVector angle(N);
    agile::phaseVector(x1, angleReal);
    agile::scale(CType(0, 1.0), angleReal, angle);
    agile::expVector(angle, angle);
    utils::GetSubVector(absb1, b1Temp, maxInd, N);
    agile::multiplyElementwise(b1Temp, angle, x1);
    utils::SetSubVector(x1, b1, maxInd, N);

    // Get new image
    utils::GetSubVector(u, uMax, cnt - 1, N);
    UTGV2Recon(b1, crec, uMax, inds);
    utils::SetSubVector(uMax, u, cnt, N);
  }

  Log("final b1 correction:\n");
  // final b1 correction
  B1Recon(uMax, crec, b1, coils, params.b1FinalNrIt, params.b1FinalReg);
}

void CoilConstruction::ExportAdditionalResults(const char *outputDir,
                                               ResultExportCallback callback)
{
}
