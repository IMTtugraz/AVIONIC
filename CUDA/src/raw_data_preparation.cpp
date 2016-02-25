#include "../include/raw_data_preparation.h"
#include <vector>
#include <iostream>
#include "agile/calc/fft.hpp"
#include "agile/io/file.hpp"
#include "../include/cartesian_coil_construction.h"
#include "../include/dicom_reader.h"
#include "../include/ismrmrd_reader.h"
#include "../include/siemens_vd11_reader.h"

RawDataPreparation::RawDataPreparation(OptionsParser &op, bool completeData,
                                       bool removeReadOutOS, bool normalizeData,
                                       bool applyChop)
  : op(op), completeData(completeData), removeReadOutOS(removeReadOutOS),
    normalizeData(normalizeData), applyChop(applyChop), nonuniformData(false),
    fftOp(NULL), croppedFftOp(NULL), dataReader(NULL)
{
  com.allocateGPU();
}

OptionsParser zeroOptions;

RawDataPreparation::RawDataPreparation(bool completeData, bool removeReadOutOS,
                                       bool normalizeData, bool applyChop)
  : op(zeroOptions), completeData(completeData),
    removeReadOutOS(removeReadOutOS), normalizeData(normalizeData),
    applyChop(applyChop), nonuniformData(false), fftOp(NULL),
    croppedFftOp(NULL), dataReader(NULL)
{
  com.allocateGPU();
}

RawDataPreparation::~RawDataPreparation()
{
  if (fftOp != NULL)
    delete fftOp;
  if (croppedFftOp != NULL)
    delete croppedFftOp;
  if (dataReader != NULL)
    delete dataReader;
}

void RawDataPreparation::InitOSRemoval(unsigned width)
{
  fftOp = new agile::FFT<CType>(1, 2 * width);
  croppedFftOp = new agile::FFT<CType>(1, width);
  fullLine = std::vector<CType>(2 * width, 0);
  fullLineGPU = CVector(2 * width);
  croppedLineGPU = CVector(width);
  croppedLineGPU.assign(width, 0);
}

void RawDataPreparation::RemoveOS(std::vector<CType> &oversampledLine,
                                  std::vector<CType> &croppedLine,
                                  unsigned width, unsigned colOffset)
{
  // correct with asymmetric echo offset
  std::copy(oversampledLine.begin(), oversampledLine.end(),
            fullLine.begin() + colOffset);

  fullLineGPU.assignFromHost(fullLine.begin(), fullLine.end());
  fftOp->Inverse(fullLineGPU, fullLineGPU);

  // first quater
  agile::lowlevel::get_content(fullLineGPU.data(), 1, width / 2.0, 0, 0,
                               croppedLineGPU.data(), 1, width / 2.0);
  // last quater
  agile::lowlevel::get_content(
      fullLineGPU.data(), 1, width / 2.0, 0, (3.0 * 2.0 * width) / 4.0,
      croppedLineGPU.data() + (unsigned)(width / 2.0), 1, width / 2.0);

  // cropped forward FFT
  croppedFftOp->Forward(croppedLineGPU, croppedLineGPU);
  croppedLineGPU.copyToHost(croppedLine);
}

void RawDataPreparation::ComputePFZeroFillOffset(
    const Acquisition &line, unsigned centerCol, unsigned centerRow,
    unsigned &colOffset, unsigned &rowOffset, unsigned &lineStart,
    unsigned &lineEnd)
{
  if (line.centerColumn != centerCol && (line.readouts / 2.0) != centerCol)
  {
    colOffset = std::max(0, (int)centerCol - (int)line.centerColumn);

    // TODO better debug strategy
    if (line.line == 0 && line.phase == 0 && op.verbose)
    {
      std::cout << "Asymmetric echo detected (number of samples != "
                   "rawDataDims.readouts)" << std::endl;
      std::cout << "Number of Samples: " << line.readouts << std::endl;
      std::cout << "Center Sample: " << line.centerColumn << std::endl;
      std::cout << "Readouts: " << rawDataDims.readouts << std::endl;
      std::cout << "Compensate asymmetric echoes inside ISMRMRD Reader with "
                   "column offset: " << (int)colOffset << std::endl;
    }
  }
  if (line.centerRow != centerRow ||
      (line.centerRow != rawDataDims.height / 2.0))
  {
    if (rawDataDims.height > rawDataDims.encodings)
    {
      rowOffset = rawDataDims.height / 2.0 - line.centerRow;
    }
    else
      rowOffset =
          std::abs((int)(rawDataDims.encodings / 2.0) - (int)line.centerRow);
          //std::max(0, (int)(rawDataDims.encodings / 2.0) - (int)line.centerRow);

    // TODO better debug strategy
    if (line.line == 0 && line.phase == 0 && op.verbose)
    {
      std::cout << "Partial Fourier in phase direction detected:" << std::endl;
      std::cout << "Reconstruction Rows: " << rawDataDims.height << std::endl;
      std::cout << "Encoding Rows: " << rawDataDims.encodings << std::endl;
      std::cout << "Encoding Center: " << line.centerRow << std::endl;
      std::cout << "Compensate with rowOffset: " << rowOffset << std::endl;
    }
  }

  // Compute clipping
  if (rawDataDims.encodings > rawDataDims.height)
  {
    lineStart = centerRow - rawDataDims.height / 2.0;
    lineEnd = centerRow + rawDataDims.height / 2.0;
    if (line.line == 0 && line.phase == 0 && op.verbose)
    {
      std::cout << "Current nEnc: " << rawDataDims.encodings << std::endl;
      std::cout << "Number of encodings larger than reconstruction height - "
                   "clipping necessary!" << std::endl;
      std::cout << "Encoding Center Row: " << centerRow << std::endl;
      std::cout << "Overlap: " << (rawDataDims.encodings - rawDataDims.height)
                << std::endl;
      std::cout << "Range: " << lineStart << " - " << lineEnd << std::endl;
    }
  }
}

// GPU Vector overload
void RawDataPreparation::PrepareRawData(CVector &kdata, RVector &mask,
                                        RVector &w, Dimension &dims)
{
  std::vector<CType> kdataHost;
  std::vector<RType> maskHost;
  std::vector<RType> wHost;
  std::string outputDir = utils::GetParentDirectory(op.outputFilename);

  //  this->completeData = true;
  //  this->removeReadOutOS = true;
  //  this->normalizeData = true;

  // Extract dicom raw data and prepare it
  if (op.nonuniform)
  {
    // Load raw data from file/directory
    // in case of nonuniform data no data chopping is performed
    this->applyChop = false;
    this->nonuniformData = true;
  }
  else
  {
    // Load raw data from file/directory
    this->applyChop = true;
  }

  std::string extension = utils::GetFileExtension(op.kdataFilename);
  if (extension.compare(".h5") == 0)
  {
    std::cout << "H5 File passed!!!" << std::endl;
    this->PrepareIsmrmrdData(op.kdataFilename, kdataHost, maskHost, wHost,
                             dims);
  }
  else if (extension.compare(".bin") == 0)
  {
    std::cout << "AGILE Bin File passed!!!" << std::endl;
    this->PrepareDicomData(op.kdataFilename, outputDir + "/meas.dat", kdataHost,
                           maskHost, wHost, dims);
  }
  else
  {
    std::cout << "DICOM File passed!!!" << std::endl;
    this->PrepareDicomData(op.kdataFilename, outputDir + "/meas.dat", kdataHost,
                           maskHost, wHost, dims);
  }

  // Copy data to GPU device
  w.assignFromHost(wHost.begin(), wHost.end());
  kdata.assignFromHost(kdataHost.begin(), kdataHost.end());
  mask.assignFromHost(maskHost.begin(), maskHost.end());
  op.dims = dims;
}

// Overload
void RawDataPreparation::PrepareRawData(const std::string &rawDataPath,
                                        std::vector<CType> &data,
                                        std::vector<RType> &mask,
                                        Dimension &dims)
{
  std::vector<RType> w(0);
  // Set MeasDat file
  op.kdataFilename = rawDataPath;
  dataReader = new SiemensVD11Reader(op);
  dataReader->LoadRawData();
  this->PrepareRawData(data, mask, w, dims);
}

void RawDataPreparation::SetTrajectoryData(
    Acquisition &line, std::vector<RType> &mask, std::vector<RType> &w,
    const Dimension &dims, unsigned maskPhaseOffset, unsigned lineOffset,
    unsigned colOffset)
{
  if (nonuniformData)
  {
    unsigned nTraj = dims.readouts * dims.encodings;
    unsigned trajPhaseOffset = line.phase * nTraj * 2;

    // get trajectory and density from acquisition data
    std::vector<RType> &traj = line.traj;
    std::vector<RType> &dens = line.dens;

    // x
    std::copy(traj.begin(), traj.begin() + dims.readouts,
              mask.begin() + trajPhaseOffset + lineOffset);
    // y
    std::copy(traj.begin() + dims.readouts, traj.end(),
              mask.begin() + trajPhaseOffset + lineOffset + nTraj);

    // density compensation
    std::copy(dens.begin(), dens.end(),
              w.begin() + lineOffset + maskPhaseOffset);
  }
  else
  {
    if (line.hasTrajectoryInformation())
    {
      // uniform case with provided trajectory data
      // TODO loop over samples and set mask!
      for (unsigned cnt = 0; cnt < line.traj.size(); cnt++)
      {
        mask[(unsigned)line.traj[cnt] + lineOffset + maskPhaseOffset] = 1.0;
      }
    }
    else
    {
      std::vector<RType> ones(dims.width - (colOffset / 2.0));
      ones.assign(ones.size(), (RType)1.0);
      // set line in mask in case of cartesian recon
      std::copy(ones.begin(), ones.end(), mask.begin() + (colOffset / 2.0) +
                                              lineOffset + maskPhaseOffset);
    }
  }
}

void RawDataPreparation::SetData(Acquisition &line, std::vector<CType> &data,
                                 const Dimension &dims, unsigned phaseOffset,
                                 unsigned lineOffset, unsigned colOffset)
{
  // init mask with corrected width
  // but without Partial Fourier/Asymmetric Echo offset
  for (unsigned int coil = 0; coil < dims.coils; coil++)
  {
    unsigned coilOffset = coil * dims.width * dims.encodings;
    std::vector<CType> chn = line.data[coil];

    if (!nonuniformData && line.hasTrajectoryInformation())
    {
      for (unsigned cnt = 0; cnt < line.traj.size(); cnt++)
      {
        data[(unsigned)line.traj[cnt] + lineOffset + coilOffset + phaseOffset] =
            chn[cnt];
      }
    }
    else if (removeReadOutOS && !nonuniformData)
    {
      std::vector<CType> croppedLine(dims.width);
      this->RemoveOS(chn, croppedLine, dims.width, colOffset);
      std::copy(croppedLine.begin(), croppedLine.end(),
                data.begin() + phaseOffset + coilOffset + lineOffset);
    }
    else
    {
      coilOffset = coil * dims.readouts * dims.encodings;

      std::copy(chn.begin(), chn.end(), data.begin() + colOffset + phaseOffset +
                                            coilOffset + lineOffset);
    }
  }
}

void RawDataPreparation::NormalizeData(std::vector<CType> &data,
                                       std::vector<RType> &mask,
                                       std::vector<RType> &w, Dimension &dims)
{
  if (this->nonuniformData)
  {
    // - generate k-space Trajectory and
    //   density compensation data
    // - generate noncart operator
    //   and coil construction
    // - perform noncart data normalization
    unsigned nRO = dims.readouts;
    unsigned spokesPerFrame = dims.encodings;

    RVector maskGPU = RVector(mask.size());
    RVector wGPU = RVector(w.size());
    maskGPU.assignFromHost(mask.begin(), mask.end());
    wGPU.assignFromHost(w.begin(), w.end());

    NoncartesianOperator *nonCartOp =
        new NoncartesianOperator(dims.width, dims.height, dims.coils,
                                 dims.frames, spokesPerFrame * dims.frames, nRO,
                                 spokesPerFrame, maskGPU, wGPU, 3, 8, 2.0);

    NoncartesianCoilConstruction *coilConstruction =
        new NoncartesianCoilConstruction(dims.width, dims.height, dims.coils,
                                         dims.frames, op.coilParams, nonCartOp);
    coilConstruction->SetVerbose(op.verbose);

    this->NormalizeNonCartData(data, dims, coilConstruction);

    delete nonCartOp;
    delete coilConstruction;
  }
  else
  {
    RVector maskGPU = RVector(mask.size());
    maskGPU.assignFromHost(mask.begin(), mask.end());
    CartesianOperator *cartOp = new CartesianOperator(
        dims.width, dims.height, dims.coils, dims.frames, maskGPU, true);
    CoilConstruction *coilConstruction =
        new CartesianCoilConstruction(dims.width, dims.height, dims.coils,
                                      dims.frames, op.coilParams, cartOp);
    this->NormalizeData(data, dims, coilConstruction);
    delete cartOp;
    delete coilConstruction;
  }
}

// Implementation
void RawDataPreparation::PrepareRawData(std::vector<CType> &data,
                                        std::vector<RType> &mask,
                                        std::vector<RType> &w, Dimension &dims)
{
  dims = dataReader->GetRawDataDimensions();
  rawDataDims = dataReader->GetRawDataDimensions();

  if (op.verbose)
    std::cout << "Passed rawDataDims: (w,h,enc,ro,chns,frames) "
              << rawDataDims.width << "," << rawDataDims.height << ","
              << rawDataDims.encodings << "," << rawDataDims.readouts << ","
              << rawDataDims.coils << "," << rawDataDims.frames << std::endl;

  removeReadOutOS =
      removeReadOutOS && (dataReader->IsOversampledData() || op.forceOSRemoval);

  unsigned short centerCol = dataReader->GetCenterColumn();
  unsigned short centerRow = dataReader->GetCenterRow();

  unsigned colOffset = 0, rowOffset = 0;
  unsigned lineStart = 0, lineEnd = rawDataDims.encodings;

  if (removeReadOutOS)
  {
    std::cout << "Init oversampling removal (forced: " << op.forceOSRemoval
              << ")" << std::endl;
    // Reduce width by oversampling factor of 2
    // TODO check if OS 2 should be assumed
    if (dims.width == dims.readouts)
    {
      dims.width = std::ceil(dims.width / 2.0);
      rawDataDims.height = std::ceil(dims.height / 2.0);
      dims.height = rawDataDims.height;
    }
    this->InitOSRemoval(dims.width);
  }

  if (op.verbose)
    std::cout << "Completed data set size: (w,h) = " << dims.width << ","
              << dims.height << std::endl;

  // init data arrays with correct sizes
  unsigned nEnc, nRO;
  if (nonuniformData)
  {
    nRO = dims.readouts;
    nEnc = dims.encodings;
    std::cout << "Nonuniform data - number of encodings: " << nEnc << std::endl;
    // mask == trajectory, x and y values of trajectory
    mask.resize(2 * nRO * nEnc * dims.frames);
    w.resize(nRO * nEnc * dims.frames);
    w.assign(w.size(), 0.0);
    mask.assign(mask.size(), 0.0);
  }
  else
  {
    if (!removeReadOutOS)
      nRO = dims.readouts;
    else
      nRO = dims.width;

    nEnc = dims.encodings;
    mask.resize(nRO * nEnc * dims.frames);
    mask.assign(mask.size(), 0.0);
  }
  data.resize(nRO * nEnc * dims.coils * dims.frames);
  data.assign(data.size(), 0);

  // loop over all acquired lines and segments
  if (op.verbose)
  {
    std::cout << "Data element count: " << data.size() << std::endl;
    std::cout << "Number of acquisitions:"
              << dataReader->GetNumberOfAcquisitions() << std::endl;
  }

  for (unsigned acqCnt = 0; acqCnt < dataReader->GetNumberOfAcquisitions();
       acqCnt++)
  {
    Acquisition line = dataReader->GetAcquisition(acqCnt);

    if (line.isNoiseMeasurement || (line.slice != op.slice) ||
        ((line.phase + (line.line % op.tpat)) % op.tpat) != 0)
      continue;

    // Check consistency
    // Check acquisition data dimensions
    if (line.readouts > dims.readouts)
    {
      std::cout << "Inconsistent data detected. Number of readouts != number "
                   "of samples in acquisition."
                << " " << line.readouts << " != " << dims.readouts << std::endl;
      std::cout << "Trying to adapt raw data dimensions. " << std::endl;
      dims.readouts = line.readouts;
      nRO = line.readouts;

      data.resize(nRO * nEnc * dims.coils * dims.frames);
      // TODO differ between cart and non-cart
      mask.resize(2 * nRO * nEnc * dims.frames);
      w.resize(nRO * nEnc * dims.frames);

      data.assign(data.size(), 0.0);
      w.assign(w.size(), 0.0);
      mask.assign(mask.size(), 0.0);
    }

    // Compute possible partial fourier/ asymmetric echo offset
    if (!line.hasTrajectoryInformation())
      this->ComputePFZeroFillOffset(line, centerCol, centerRow, colOffset,
                                    rowOffset, lineStart, lineEnd);

    if (rowOffset > 0 && dims.encodings != dims.height)
    {
      std::cout << "Inconsistent data detected. Number of encodings differs."
                << std::endl;

      std::cout << "Current nEnc: " << dims.encodings << std::endl;
      std::cout << "centerRow: " << centerRow << " rowOffset: " << rowOffset
                << std::endl;

      dims.encodings = dims.height;  // dims.encodings + 2 * rowOffset;
      nEnc = dims.encodings;

      std::cout << "Final number of encodings: " << nEnc << std::endl;
      std::cout << "Height: " << dims.height << std::endl;

      data.resize(nRO * nEnc * dims.coils * dims.frames);
      if (nonuniformData || line.hasTrajectoryInformation())
        mask.resize(2 * nRO * nEnc * dims.frames);
      else
        mask.resize(nRO * nEnc * dims.frames);
      w.resize(nRO * nEnc * dims.frames);

      data.assign(data.size(), 0.0);
      w.assign(w.size(), 0.0);
      mask.assign(mask.size(), 0.0);
    }

    if (lineStart > 0 && dims.encodings != dims.height)
    {
      std::cout << "Inconsistent data detected. Number of encodings differs. "
                   "Resize to reconstruction space height and clip lines."
                << std::endl;
      nEnc = dims.height;
      dims.encodings = dims.height;

      data.resize(nRO * nEnc * dims.coils * dims.frames);
      if (nonuniformData || line.hasTrajectoryInformation())
        mask.resize(2 * nRO * nEnc * dims.frames);
      else
        mask.resize(nRO * nEnc * dims.frames);
      w.resize(nRO * nEnc * dims.frames);

      data.assign(data.size(), 0.0);
      w.assign(w.size(), 0.0);
      mask.assign(mask.size(), 0.0);
    }

    unsigned maskPhaseOffset, lineOffset, phaseOffset;
    if (line.line >= lineStart && line.line < lineEnd)
    {
      maskPhaseOffset = line.phase * nRO * nEnc;
      if (lineStart > 0)
        lineOffset = (line.line - lineStart) * nRO;
      else
        lineOffset = (rowOffset + line.line - lineStart) * nRO;

      phaseOffset = line.phase * nRO * nEnc * dims.coils;

      // do not pass colOffset as last parameter since in case of OS removal
      // no padding is performed
      SetTrajectoryData(line, mask, w, dims, maskPhaseOffset, lineOffset,
                        removeReadOutOS ? 0 : colOffset);
      SetData(line, data, dims, phaseOffset, lineOffset, colOffset);
    }
    else if (op.verbose)
      std::cout << "DEBUG: Skip line no. " << line.line << " (out of range)"
                << std::endl;

    if (acqCnt % 500 == 0)
      std::cout << acqCnt << " acquisitions read." << std::endl;
  }  // loop over acquisitions

  if (this->normalizeData)
  {
    this->NormalizeData(data, mask, w, dims);
  }

  if (this->applyChop)
  {
    this->ChopData(data, mask, dims);
  }
}

void RawDataPreparation::PrepareIsmrmrdData(const std::string &rawDataPath,
                                            std::vector<CType> &data,
                                            std::vector<RType> &mask,
                                            std::vector<RType> &w,
                                            Dimension &dims)
{
  // Generate MeasDat file
  op.kdataFilename = rawDataPath;
  dataReader = new IsmrmrdReader(op);
  dataReader->LoadRawData();
  this->PrepareRawData(data, mask, w, dims);
}

void RawDataPreparation::PrepareDicomData(const std::string &dicomDataPath,
                                          const std::string &rawDataPath,
                                          std::vector<CType> &data,
                                          std::vector<RType> &mask,
                                          std::vector<RType> &w,
                                          Dimension &dims)
{
  // Generate MeasDat file
  op.kdataFilename = dicomDataPath;
  dataReader = new DicomReader(op);
  dataReader->LoadRawData();
  this->PrepareRawData(data, mask, w, dims);
}

struct isLargerThan
{
  isLargerThan(const RType &maximum) : maximum(maximum)
  {
  }
  RType maximum;
  bool operator()(const RType &el)
  {
    return el >= maximum;
  }
};

RType RawDataPreparation::FindNormalizationFactor(std::vector<RType> &data)
{
  std::vector<RType>::iterator maxElement;
  maxElement = std::max_element(data.begin(), data.end());

  std::vector<RType>::iterator largest90Percent =
      std::find_if(data.begin(), data.end(), isLargerThan(*maxElement));

  std::vector<RType> largestElements;

  // Find all elements larger than 90 percent of maximum
  while (largest90Percent != data.end())
  {
    largestElements.push_back(*largest90Percent);
    largest90Percent = std::find_if(++largest90Percent, data.end(),
                                    isLargerThan(*maxElement * 0.9));
  }

  // Find median of resulting values
  if (largestElements.size() > 0)
    return utils::Median(largestElements);
  else
    throw std::runtime_error("NormalizeNonCartData: No valid normalization "
                             "factor found! Check convergence/regularization "
                             "parameters of coil construction.");
}

void RawDataPreparation::NormalizeNonCartData(
    std::vector<CType> &data, const Dimension &dims,
    NoncartesianCoilConstruction *coilConstruction)
{
  std::cout << "Normalize noncart data.." << dims.width
            << " height:  " << dims.height << std::endl;
  std::cout << "Kdata size: " << data.size() << std::endl;
  CVector kdata(data.size());
  kdata.assignFromHost(data.begin(), data.end());

  CVector u(dims.width * dims.height * dims.coils);
  CVector u0(dims.width * dims.height);
  u0.assign(u0.size(), 0);

  CVector b1(dims.width * dims.height * dims.coils);
  CVector crec(dims.width * dims.height * dims.coils);
  crec.assign(crec.size(), 0);

  coilConstruction->PerformCoilConstruction(kdata, u, b1, com);

  coilConstruction->TimeAveragedReconstruction(kdata, u0, crec, false);

  u0.assign(u0.size(), 0);
  CVector crecTemp(dims.width * dims.height);
  CVector b1Temp(dims.width * dims.height);
  for (unsigned coil = 0; coil < dims.coils; coil++)
  {
    utils::GetSubVector(crec, crecTemp, coil, dims.width * dims.height);
    utils::GetSubVector(b1, b1Temp, coil, dims.width * dims.height);
    agile::multiplyConjElementwise(b1Temp, crecTemp, crecTemp);
    agile::addVector(u0, crecTemp, u0);
  }

  RVector u0Abs(u0.size());
  agile::absVector(u0, u0Abs);

  std::vector<RType> uTemp(u0Abs.size());
  u0Abs.copyToHost(uTemp);
  CType median = FindNormalizationFactor(uTemp);
  CType datanorm = (CType)255.0 / median;

  std::cout << "Datanorm:" << datanorm << std::endl;
  agile::scale(datanorm, kdata, kdata);
  kdata.copyToHost(data);
}

void RawDataPreparation::NormalizeData(std::vector<CType> &data,
                                       const Dimension &dims,
                                       CoilConstruction *coilConstruction)
{
  CVector kdata(data.size());
  kdata.assignFromHost(data.begin(), data.end());
  CVector u0(dims.width * dims.height);
  u0.assign(u0.size(), 0);

  CVector crec(dims.width * dims.height * dims.coils);
  crec.assign(crec.size(), 0);
  coilConstruction->TimeAveragedReconstruction(kdata, u0, crec, false);
  RVector u0Abs(u0.size());
  agile::absVector(u0, u0Abs);

  std::vector<RType> uTemp(u0Abs.size());
  u0Abs.copyToHost(uTemp);
  CType median = FindNormalizationFactor(uTemp);
  CType datanorm = (CType)255.0 / median;

  std::cout << "Datanorm:" << datanorm << std::endl;
  agile::scale(datanorm, kdata, kdata);
  kdata.copyToHost(data);
}

void RawDataPreparation::GenerateChopMatrix(std::vector<RType> &chopMatrix,
                                            unsigned width, unsigned height)
{
  chopMatrix.resize(width * height);

  for (unsigned x = 0; x < width; x++)
    for (unsigned y = 0; y < height; y++)
      chopMatrix[x + y * width] = std::pow(-1.0, x + y);
}

void RawDataPreparation::ChopData(std::vector<CType> &data,
                                  std::vector<RType> &mask,
                                  const Dimension &dims)
{
  // Generate Chop matrix
  std::vector<RType> chop;
  GenerateChopMatrix(chop, dims.width, dims.height);
  RVector chopGPU(chop.size());
  chopGPU.assignFromHost(chop.begin(), chop.end());

  // Copy data and mask to GPU device
  CVector dataGPU(data.size());
  dataGPU.assignFromHost(data.begin(), data.end());
  RVector maskGPU(mask.size());
  maskGPU.assignFromHost(mask.begin(), mask.end());

  unsigned N = dims.width * dims.height;
  RVector frameMask(N);
  CVector coilData(N);

  for (unsigned frame = 0; frame < dims.frames; frame++)
  {
    // shift mask
    utils::GetSubVector(maskGPU, frameMask, frame, N);
    agile::lowlevel::ifftshift(frameMask.data(), dims.height, dims.width);
    utils::SetSubVector(frameMask, maskGPU, frame, N);

    unsigned frameOffset = frame * dims.coils;
    // apply chop matrix and shift
    for (unsigned coil = 0; coil < dims.coils; coil++)
    {
      utils::GetSubVector(dataGPU, coilData, coil + frameOffset, N);
      agile::multiplyElementwise(coilData, chopGPU, coilData);
      agile::lowlevel::ifftshift(coilData.data(), dims.height, dims.width);
      utils::SetSubVector(coilData, dataGPU, coil + frameOffset, N);
    }
  }

  // copy data back to host
  dataGPU.copyToHost(data);
  maskGPU.copyToHost(mask);
}

