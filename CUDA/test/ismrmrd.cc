#include <gtest/gtest.h>
#include <iostream>
#include "./test_utils.h"
#include "ismrmrd/ismrmrd.h"
#include "ismrmrd/dataset.h"
#include "ismrmrd/xml.h"
#include "ismrmrd/version.h"
#include "../include/types.h"
#include "agile/calc/fft.hpp"

std::string outfile("test.h5");
std::string dataset("dataset");
unsigned readout = 10;
unsigned matrix_size = readout;
unsigned coils = 4;

TEST(Test_Ismrmrd, GenerateAcquisitionFile)
{
  // Clear data file
  std::remove(outfile.c_str());

  ISMRMRD::Dataset d(outfile.c_str(), dataset.c_str(), true);

  ISMRMRD::Acquisition acq(readout, coils, 2);
  memset((void *)acq.getDataPtr(), 0, acq.getDataSize());
  acq.available_channels() = coils;
  acq.center_sample() = (readout >> 1);
  for (size_t i = 0; i < matrix_size; i++)
  {
    acq.clearAllFlags();
    acq.idx().kspace_encode_step_1 = i;
    for (size_t c = 0; c < coils; c++)
    {
      for (size_t s = 0; s < readout; s++)
      {
        acq.data(s, c) = std::complex<float>(s, c);
      }
    }
    float ky = (1.0 * i - (matrix_size >> 1)) / (1.0 * matrix_size);
    for (size_t x = 0; x < readout; x++)
    {
      float kx = (1.0 * x - (readout >> 1)) / (1.0 * readout);
      acq.traj(0, x) = kx;
      acq.traj(1, x) = ky;
    }
    d.appendAcquisition(acq);
  }
  EXPECT_EQ(10u, d.getNumberOfAcquisitions());
}

TEST(Test_Ismrmrd, GenerateHeaderXML)
{
  ISMRMRD::Dataset d(outfile.c_str(), dataset.c_str(), true);
  ISMRMRD::IsmrmrdHeader h;
  h.version = ISMRMRD_XMLHDR_VERSION;

  ISMRMRD::AcquisitionSystemInformation sys;
  sys.institutionName = "ISMRM Synthetic Imaging Lab";
  sys.receiverChannels = coils;
  h.acquisitionSystemInformation = sys;

  // Create an encoding section¬
  ISMRMRD::Encoding e;
  e.encodedSpace.matrixSize.x = readout;
  e.encodedSpace.matrixSize.y = matrix_size;
  e.encodedSpace.matrixSize.z = 1;

  e.encodedSpace.fieldOfView_mm.x = 300;
  e.encodedSpace.fieldOfView_mm.y = 300;
  e.encodedSpace.fieldOfView_mm.z = 6;

  e.reconSpace.matrixSize.x = readout;
  e.reconSpace.matrixSize.y = matrix_size;
  e.reconSpace.matrixSize.z = 1;

  e.reconSpace.fieldOfView_mm.x = 300;
  e.reconSpace.fieldOfView_mm.y = 300;
  e.reconSpace.fieldOfView_mm.z = 6;
  e.trajectory = "cartesian";
  e.encodingLimits.kspace_encoding_step_1 =
      ISMRMRD::Limit(0, matrix_size - 1, (matrix_size >> 1));

  h.encoding.push_back(e);

  // Serialize the header¬
  std::stringstream str;
  ISMRMRD::serialize(h, str);
  std::string xml_header = str.str();

  // Write the header to the data file.¬
  d.writeHeader(xml_header);
}

TEST(Test_Ismrmrd, LoadRawDataFromFile)
{
  // Only try to open file
  ISMRMRD::Dataset d(outfile.c_str(), dataset.c_str(), false);
  std::string xml;
  d.readHeader(xml);

  ISMRMRD::IsmrmrdHeader hdr;
  ISMRMRD::deserialize(xml.c_str(), hdr);

  EXPECT_EQ(ISMRMRD_XMLHDR_VERSION, hdr.version());
  ISMRMRD::EncodingSpace e_space = hdr.encoding[0].encodedSpace;
  ISMRMRD::EncodingSpace r_space = hdr.encoding[0].reconSpace;

  EXPECT_EQ(10, e_space.matrixSize.x);
  EXPECT_EQ(10, e_space.matrixSize.y);
  EXPECT_EQ(1, e_space.matrixSize.z);

  EXPECT_EQ(10, r_space.matrixSize.x);
  EXPECT_EQ(10, r_space.matrixSize.y);
  EXPECT_EQ(1, r_space.matrixSize.z);

  EXPECT_EQ(10u, d.getNumberOfAcquisitions());

  // load raw data
  unsigned int number_of_acquisitions = d.getNumberOfAcquisitions();

  ISMRMRD::Acquisition acq;
  for (unsigned int i = 0; i < number_of_acquisitions; i++)
  {
    // Read one acquisition at a time¬
    d.readAcquisition(i, acq);
    EXPECT_EQ(coils, acq.active_channels());
    EXPECT_EQ(i, acq.idx().kspace_encode_step_1);

    for (unsigned coil = 0; coil < coils; coil++)
    {
      for (unsigned cnt = 0; cnt < e_space.matrixSize.x; cnt++)
      {
        EXPECT_EQ(acq.data(cnt, coil), std::complex<float>(cnt, coil));
        // alternative indexing
        EXPECT_EQ(acq.getDataPtr()[cnt + coil * e_space.matrixSize.x],
                  std::complex<float>(cnt, coil));
      }
    }

    EXPECT_EQ(2, acq.getHead().trajectory_dimensions);
    if (i == 0)
    {
      EXPECT_NEAR(-0.5, acq.traj(0, 0), EPS);
      EXPECT_NEAR(-0.5, acq.traj(1, 0), EPS);
    }
  }
}

TEST(Test_Ismrmrd, Reconstruct2DCartesianPhantom)
{
  agile::GPUEnvironment::allocateGPU(0);
  ISMRMRD::Dataset d("../test/data/ismrmrd/phantom.h5", "dataset", false);
  std::string xml;
  d.readHeader(xml);

  ISMRMRD::IsmrmrdHeader hdr;
  ISMRMRD::deserialize(xml.c_str(), hdr);

  EXPECT_EQ(ISMRMRD_XMLHDR_VERSION, hdr.version());
  ISMRMRD::EncodingSpace eSpace = hdr.encoding[0].encodedSpace;
  ISMRMRD::EncodingSpace rSpace = hdr.encoding[0].reconSpace;

  ISMRMRD::MatrixSize eDims = eSpace.matrixSize;

  std::cout << " EncodingSpace dims x: " << eDims.x << std::endl;
  std::cout << " EncodingSpace dims y: " << eDims.y << std::endl;
  unsigned coils = hdr.acquisitionSystemInformation().receiverChannels();

  std::cout << " Channels: " << coils << std::endl;

  unsigned int numberOfAcquisitions = d.getNumberOfAcquisitions();
  std::cout << " Number of Acquisitions: " << numberOfAcquisitions << std::endl;

  std::vector<RType> ones(eDims.x);
  ones.assign(ones.size(), (RType)1.0);

  // Data buffer
  std::vector<CType> data(eDims.x * eDims.y * coils);
  std::vector<RType> mask(eDims.x * eDims.y);

  if (hdr.encoding[0].parallelImaging.is_present())
    std::cout << "Step size: "
              << hdr.encoding[0]
                     .parallelImaging()
                     .accelerationFactor.kspace_encoding_step_1 << std::endl;
  if (hdr.encoding[0].encodingLimits.repetition.is_present())
    std::cout << "Repetitions: "
              << hdr.encoding[0].encodingLimits.repetition().maximum
              << std::endl;

  ISMRMRD::Acquisition acq;
  for (unsigned int i = 0; i < numberOfAcquisitions; i++)
  {
    // Read one acquisition at a time¬
    d.readAcquisition(i, acq);
    EXPECT_EQ(coils, acq.active_channels());

    for (unsigned coil = 0; coil < coils; coil++)
    {
      unsigned line = acq.idx().kspace_encode_step_1;
      unsigned coilOffset = coil * eDims.x;
      unsigned int offset = line * eDims.x + coil * eDims.x * eDims.y;
      std::copy(acq.data_begin() + coilOffset,
                acq.data_begin() + coilOffset + eDims.x, data.begin() + offset);

      std::copy(ones.begin(), ones.end(), mask.begin() + line * eDims.x);
    }
    if (acq.isFlagSet(ISMRMRD::ISMRMRD_ACQ_LAST_IN_SLICE))
    {
      std::cout << "Stopping after first repetition..." << std::endl;
      break;
    }
  }

  // Simple reconstruction using FFT
  CVector temp(eDims.x * eDims.y);
  CVector dataGPU;
  dataGPU.assignFromHost(data.begin(), data.end());

  agile::FFT<CType> *fftOp = new agile::FFT<CType>(eDims.y, eDims.x);
  fftOp->CenteredForward(dataGPU, temp, 0, 0);

  std::vector<CType> recon(temp.size());
  temp.copyToHost(recon);

  std::vector<size_t> dims;
  dims.push_back(eDims.x);
  dims.push_back(eDims.y);
  ISMRMRD::NDArray<CType> maskArray(dims);
  ISMRMRD::NDArray<CType> reconArray(dims);
  dims.push_back(coils);
  ISMRMRD::NDArray<CType> buffer(dims);

  std::copy(data.begin(), data.end(), buffer.begin());
  std::copy(mask.begin(), mask.end(), maskArray.begin());
  std::copy(recon.begin(), recon.end(), reconArray.begin());

  d.appendNDArray("kspace", buffer);
  d.appendNDArray("mask", maskArray);
  d.appendNDArray("recon", reconArray);
}

