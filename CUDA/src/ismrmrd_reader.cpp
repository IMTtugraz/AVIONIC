#include "../include/ismrmrd_reader.h"
#include "ismrmrd/ismrmrd.h"
#include "ismrmrd/dataset.h"
#include "ismrmrd/xml.h"
#include "ismrmrd/version.h"
#include <algorithm>

IsmrmrdReader::IsmrmrdReader(OptionsParser &op)
  : RawDataReader(op), filename(op.kdataFilename), dataset(NULL), hdr(NULL)
{
}

IsmrmrdReader::~IsmrmrdReader()
{
  if (dataset != NULL)
    delete dataset;
  if (hdr != NULL)
    delete hdr;
}

void IsmrmrdReader::InitRawDataDimensions()
{
  ISMRMRD::EncodingSpace eSpace = hdr->encoding[0].encodedSpace;
  ISMRMRD::EncodingSpace rSpace = hdr->encoding[0].reconSpace;
  ISMRMRD::MatrixSize eDims = eSpace.matrixSize;
  ISMRMRD::MatrixSize rDims = rSpace.matrixSize;

  rawDataDims.width = rDims.x;
  rawDataDims.height = rDims.y;
  rawDataDims.readouts = eDims.x;
  rawDataDims.encodings = eDims.y;

  ISMRMRD::Optional<ISMRMRD::Limit> phaseEncodingLimits =
      hdr->encoding[0].encodingLimits.kspace_encoding_step_1;
  if (phaseEncodingLimits.is_present())
  {
    rawDataDims.encodings = phaseEncodingLimits().maximum + 1;
    std::cout << "Use number of encodings from encodingLimits. Replace "
                 "eSpace.matrixSize.y " << eDims.y << " with "
              << rawDataDims.encodings << std::endl;
  }

  rawDataDims.frames = hdr->encoding[0].encodingLimits.phase().maximum + 1;

  // TODO receiverChannel information might not accord with real number of
  // used channels
  if (hdr->acquisitionSystemInformation.is_present() &&
      hdr->acquisitionSystemInformation().receiverChannels.is_present())
  {
    rawDataDims.coils = hdr->acquisitionSystemInformation().receiverChannels();
  }
  else
  {
    // TODO quick fix
    // read active channels of first acquisition
    ISMRMRD::Acquisition ismrmrdAcq;
    dataset->readAcquisition(0, ismrmrdAcq);
    rawDataDims.coils = ismrmrdAcq.active_channels();
  }
}

void IsmrmrdReader::LoadRawData()
{
  std::cout << "Trying to load file: " << filename << std::endl;
  dataset = new ISMRMRD::Dataset(filename.c_str(), "dataset", false);
  std::string xml;
  dataset->readHeader(xml);

  hdr = new ISMRMRD::IsmrmrdHeader();

  ISMRMRD::deserialize(xml.c_str(), *hdr);
  InitRawDataDimensions();
}

Dimension IsmrmrdReader::GetRawDataDimensions() const
{
  return rawDataDims;
}

unsigned IsmrmrdReader::GetCenterRow() const
{
  // return hdr->encoding[0].encodingLimits.kspace_encoding_step_1().center;
  // TODO check <-- necessary in order to detect Partial Fourier correctly
  if (hdr->encoding[0].encodingLimits.kspace_encoding_step_1.is_present())
    return hdr->encoding[0].encodingLimits.kspace_encoding_step_1().center;
  else
    return hdr->encoding[0].encodedSpace.matrixSize.y / 2.0;
}

unsigned IsmrmrdReader::GetCenterColumn() const
{
  return hdr->encoding[0].encodedSpace.matrixSize.x / 2.0;
}

unsigned IsmrmrdReader::GetNumberOfAcquisitions() const
{
  return dataset->getNumberOfAcquisitions();
}

bool IsmrmrdReader::IsNonUniformData() const
{
  // return op.nonuniform;
  // std::cout << "InNonUniform? " <<
  // (std::string("radial").compare(hdr->encoding[0].trajectory) == 0) <<
  // std::endl;
  return std::string("radial").compare(hdr->encoding[0].trajectory) == 0;
}

bool IsmrmrdReader::IsOversampledData() const
{
  // TODO check if this always means oversampled data or if the check has to be
  // done depending on acquisition cols compared to recon width!
  return !this->IsNonUniformData() &&
         (hdr->encoding[0].encodedSpace.matrixSize.x >
              hdr->encoding[0].reconSpace.matrixSize.x);
}

Acquisition IsmrmrdReader::GetAcquisition(unsigned index) const
{
  Acquisition acq;
  ISMRMRD::Acquisition ismrmrdAcq;
  dataset->readAcquisition(index, ismrmrdAcq);
  assert(rawDataDims.coils == ismrmrdAcq.active_channels());

  // TODO check index range
  // Compute line Offset due to Partial Fourier in Phase direction
  acq.line = ismrmrdAcq.idx().kspace_encode_step_1;
  acq.phase = ismrmrdAcq.idx().phase;
  acq.slice = ismrmrdAcq.idx().slice;

  //  std::cout << "Line/phase: " << acq.line << "," << acq.phase << std::endl;

  unsigned encodingRef = ismrmrdAcq.encoding_space_ref();

  // set encoding meta info
  acq.readouts = ismrmrdAcq.number_of_samples();
  acq.centerRow =
      hdr->encoding[encodingRef].encodingLimits.kspace_encoding_step_1().center;
  acq.centerColumn = ismrmrdAcq.center_sample();

  bool hasTrajectoryInformation = ismrmrdAcq.getNumberOfTrajElements() > 0;
  // if (hasTrajectoryInformation)
  //  std::cout << "Trajectory information embedded!" << std::endl;
  // Set coil data
  // unsigned N = rawDataDims.readouts;
  unsigned N = ismrmrdAcq.number_of_samples();

  if (ismrmrdAcq.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_NOISE_MEASUREMENT) /*||
      ismrmrdAcq.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_PARALLEL_CALIBRATION)*/)
  {
    std::cout << "NOISE MEASUREMENT! "
              << " lineIdx: " << acq.line << std::endl;
    acq.isNoiseMeasurement = true;
  }
  else
    acq.isNoiseMeasurement = false;

  for (unsigned coil = 0; coil < rawDataDims.coils; coil++)
  {
    unsigned coilOffset = coil * ismrmrdAcq.number_of_samples();
    std::vector<CType> coilData(ismrmrdAcq.number_of_samples());
    coilData.assign(coilData.size(), 0);

    // check if trajectory data is embedded
    if (hasTrajectoryInformation)
    {
      if (coil == 0)
      {
        acq.traj.resize(N);
        acq.dens.resize(N);

        acq.traj.assign(N, 0);
        acq.dens.assign(N, 0);
      }

      for (unsigned pos = 0; pos < ismrmrdAcq.number_of_samples(); pos++)
      {
        unsigned x = ismrmrdAcq.traj(0, pos);
        unsigned y = ismrmrdAcq.traj(1, pos);
        assert(y == acq.line);
        // TODO doesn't seem to work with test data
        // order in data array is flipped (coils, samples)
        //
        // coilData[x] = ismrmrdAcq.data(pos, coil);
        coilData[pos] = ismrmrdAcq.getDataPtr()[coil + pos * rawDataDims.coils];

        if (coil == 0)
        {
          // x
          acq.traj[pos] = x;

          // density compensation
          // TODO density information?
          acq.dens[pos] = 1.0;
        }
      }
    }
    else
    {
      // expect fully acquired line
      std::copy(ismrmrdAcq.data_begin() + coilOffset,
                ismrmrdAcq.data_begin() + coilOffset +
                    ismrmrdAcq.number_of_samples(),
                coilData.begin());
    }

    acq.data.insert(std::make_pair(coil, coilData));
  }

  // Set trajectory mask data
  if (this->IsNonUniformData())
  {
    // std::cout << "Number of samples: " << ismrmrdAcq.number_of_samples()
    //          << std::endl << "Readouts: " << rawDataDims.readouts <<
    //          std::endl;
    if (hasTrajectoryInformation)
    {
      unsigned N = rawDataDims.readouts;
      acq.traj.resize(2 * N);
      acq.dens.resize(N);

      for (unsigned enc = 0; enc < N; enc++)
      {
        // x
        acq.traj[enc] = ismrmrdAcq.traj(0, enc);
        // y
        acq.traj[enc + N] = ismrmrdAcq.traj(1, enc);
        // density compensation
        // TODO density information?
        acq.dens[enc] = 1.0;
      }
    }
    else
      this->GenerateRadialTrajectory(acq.line, acq.traj, acq.dens,
                                     rawDataDims.encodings,
                                     ismrmrdAcq.number_of_samples());
  }

  return acq;
}

