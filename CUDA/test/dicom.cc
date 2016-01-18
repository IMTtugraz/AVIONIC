#include <gtest/gtest.h>
#include <iostream>
#include <boost/filesystem.hpp>
#include <map>
#include <algorithm>
#include <functional>

#include "./test_utils.h"
#include "../include/types.h"
#include "agile/io/file.hpp"
#include "agile/io/dicom.hpp"
#include "agile/io/readSiemensVD11.hpp"
#include "agile/calc/fft.hpp"
#include "../include/raw_data_preparation.h"
#include "../include/dicom_reader.h"
#include "../include/cartesian_coil_construction.h"
#include "../include/noncartesian_coil_construction.h"

class Test_Dicom : public ::testing::Test
{
 public:
  virtual void SetUp()
  {
    com.allocateGPU();
  }

  communicator_type com;
  OptionsParser op;
};

int getMaskBitValue(unsigned int mask, int bit)
{
  return std::min((Uint32(mask) & Uint32(std::pow(double(2), int(bit)))),
                  Uint32(1));
}

TEST_F(Test_Dicom, GetMaskBitValue)
{
  EXPECT_EQ(getMaskBitValue(1, 0), 1);
  EXPECT_EQ(getMaskBitValue(2, 1), 1);
  EXPECT_EQ(getMaskBitValue(4, 2), 1);
  EXPECT_EQ(getMaskBitValue(8, 3), 1);
  EXPECT_EQ(getMaskBitValue(8, 31), 0);
  EXPECT_EQ(getMaskBitValue(1048576, 20), 1);
}

TEST_F(Test_Dicom, ReadFileAndShowDicomHeader)
{
  agile::DICOM dicom;
  std::string filename("../test/data/dicom/test.dcm");

  EXPECT_EQ(dicom.readdicom_info(filename), 0);
  agile::DicomInfo info = dicom.get_dicominfo();
  std::cout << "Rows: " << info.ush_rows << std::endl;
  std::cout << "Columns: " << info.ush_columns << std::endl;
  EXPECT_EQ(info.ush_rows, 144);
  EXPECT_EQ(info.ush_columns, 328);
}

TEST_F(Test_Dicom, ReadFileAndShowExtraDicomHeader)
{
  agile::DICOM dicom;
  std::string filename("../test/data/dicom/test_large.dcm");

  EXPECT_EQ(dicom.readdicom_info(filename), 0);
  agile::DicomInfo info = dicom.get_dicominfo();
  DcmDataset *dataset = dicom.getDicomDataSet();

  // Extract dicom header field that is not
  // covered by agile::DicomInfo
  OFString institution;
  dataset->findAndGetOFString(DCM_InstitutionName, institution);
  std::cout << "Institution: " << institution.c_str() << std::endl;

  OFString imageType;
  dataset->findAndGetOFStringArray(DCM_ImageType, imageType, true);
  std::cout << "ImageType: " << imageType.c_str() << std::endl;

  EXPECT_EQ(std::string("TUG").compare(institution.c_str()), 0);
  EXPECT_EQ(std::string("ORIGINAL\\PRIMARY\\RAW").compare(imageType.c_str()),
            0);
}

TEST_F(Test_Dicom, ReadFileAndGenerateMeasDat)
{
  agile::DICOM dicom;

  std::string filename("../test/data/dicom/test_large.dcm");
  std::ofstream measdat("../test/data/dicom/test_large.dat");
  unsigned long length;
  Uint8 *pdata = NULL;

  EXPECT_EQ(dicom.readdicom(filename, measdat, length, pdata), 0);
  measdat.close();
  EXPECT_EQ(length, 4128000u);
}

TEST_F(Test_Dicom, ReadMeasDatHeader)
{
  std::string measdat("../test/data/dicom/test_large.dat");

  agile::ReadSiemensVD11 rawData(measdat);
  agile::ReadSiemensVD11::sMDH mdh;
  unsigned int bytes = rawData.getMDHinfo(0, mdh);
  std::cout << "mdh: " << std::endl << mdh << std::endl;
  EXPECT_EQ(bytes, mdh.getDMALength());
  EXPECT_EQ(66u, mdh.MeasUID);
  EXPECT_EQ(3585u, mdh.ScanCounter);
  EXPECT_EQ(18119903u, mdh.TimeStamp);
  EXPECT_EQ(290u, mdh.PMUTimeStamp);
  EXPECT_EQ(0u, mdh.SystemType);
  EXPECT_EQ(41683u, mdh.PTABPosDelay);
  EXPECT_EQ(0u, mdh.PTABPosX);
  EXPECT_EQ(0u, mdh.PTABPosY);
  //  EXPECT_EQ(4293378440, mdh.PTABPosZ);
  agile::ReadSiemensVD11::EvalInfoMask mask = mdh.getEvalInfoMask();
  EXPECT_EQ(0, mask.ACQEND);
  EXPECT_EQ(1, mask.ONLINE);
  EXPECT_EQ(1, mask.RETRO_ARRDETDISABLED);
  EXPECT_EQ(328, mdh.SamplesInScan);
  EXPECT_EQ(18, mdh.UsedChannels);
  EXPECT_EQ(136, mdh.sLoopCounter.Line);
  EXPECT_EQ(18, mdh.sLoopCounter.Phase);
  EXPECT_EQ(11, mdh.sLoopCounter.Seg);
  EXPECT_EQ(120, mdh.KSpaceCentreColumn);
  EXPECT_EQ(72, mdh.KSpaceCentreLineNo);
}

TEST_F(Test_Dicom, ReadMeasDat)
{
  std::string measdat("../test/data/dicom/test_large.dat");
  const char *output = "../test/data/dicom/parsed/test_large.bin";

  std::vector<CType> raw;
  std::vector<RType> mask;
  Dimension dim;
  RawDataPreparation rdp;
  rdp.PrepareRawData(measdat, raw, mask, dim);

  EXPECT_EQ(144u, dim.height);
  EXPECT_EQ(328u, dim.width);
  EXPECT_EQ(18u, dim.coils);
  EXPECT_EQ(26u, dim.frames);

  std::cout << "Rawdata length: " << raw.size() << std::endl;
  EXPECT_EQ(22104576u, raw.size());

  agile::VD11_Image img;
  EXPECT_EQ(120u, img.getUshConsistentKSpaceCentreColumn());
  EXPECT_EQ(72u, img.getUshConsistentKSpaceCentreLineNo());

  agile::writeVectorFile(output, raw);
}

TEST_F(Test_Dicom, DISABLED_ReadMeasDatAndCompleteData)
{
  std::string measdat("../test/data/dicom/test_large.dat");
  const char *output = "../test/data/dicom/parsed/test_large.bin";

  std::vector<CType> raw;
  std::vector<RType> mask;
  Dimension dim;
  RawDataPreparation rdp(true);
  rdp.PrepareRawData(measdat, raw, mask, dim);

  EXPECT_EQ(416u, dim.width);
  EXPECT_EQ(144u, dim.height);
  EXPECT_EQ(18u, dim.coils);
  EXPECT_EQ(26u, dim.frames);

  std::cout << "Rawdata length: " << raw.size() << std::endl;
  EXPECT_EQ(28035072u, raw.size());

  agile::writeVectorFile(output, raw);
}

TEST_F(Test_Dicom, DISABLED_ReadMeasDatCompleteDataAndRemoveOS)
{
  std::string measdat("../test/data/dicom/test_large.dat");
  const char *output = "../test/data/dicom/parsed/test_large.bin";

  std::vector<CType> raw;
  std::vector<RType> mask;
  Dimension dim;
  RawDataPreparation rdp(true, true);
  rdp.PrepareRawData(measdat, raw, mask, dim);

  EXPECT_EQ(208u, dim.width);
  EXPECT_EQ(144u, dim.height);
  EXPECT_EQ(18u, dim.coils);
  EXPECT_EQ(26u, dim.frames);

  // Test some values of the first line!
  unsigned last = dim.width * dim.height * (dim.frames - 1) * dim.coils;

  EXPECT_NEAR(-2.0241E-4, raw[last - 1].real(), 1E-5);
  EXPECT_NEAR(1.0160E-5, raw[last - 1].imag(), 1E-5);
  EXPECT_NEAR(1.1342E-5, raw[last - 2].real(), 1E-5);
  EXPECT_NEAR(1.9829E-4, raw[last - 2].imag(), 1E-5);

  std::cout << "Rawdata length: " << raw.size() << std::endl;
  EXPECT_EQ(14017536u, raw.size());

  agile::writeVectorFile(output, raw);
}

TEST_F(Test_Dicom, ListDirectoryFiles)
{
  std::string dicomDir("../../../../20150325cardiac_real_time_export/"
                       "sendmeasdat_unbuffered_iPAToff/");
  op.kdataFilename = dicomDir;
  DicomReader loader(op);

  DicomFileList dicomFiles = loader.GetDicomFileList();
  DicomFileList::iterator dicomIt = dicomFiles.begin();
  std::cout << "Dicom files ordered:" << std::endl;
  while (dicomIt != dicomFiles.end())
  {
    std::cout << "File " << dicomIt->second.c_str() << std::endl;
    dicomIt++;
  }
}

TEST_F(Test_Dicom, DISABLED_ReadAndParseDicomDirectoryFiles)
{
  std::string dicomDir("../../../../20150325cardiac_real_time_export/"
                       "sendmeasdat_unbuffered_iPAToff/");
  std::string rawDataFile("../test/data/dicom/test_really_large.dat");

  std::vector<CType> raw;
  std::vector<RType> mask, w(0);
  Dimension dim;
  RawDataPreparation rdp(true, true);

  // Extract dicom raw data and prepare it
  rdp.PrepareDicomData(dicomDir, rawDataFile, raw, mask, w, dim);

  EXPECT_EQ(14017536u, raw.size());
  std::cout << "Rawdata length: " << raw.size() << std::endl;

  const char *output = "../test/data/dicom/parsed/test_really_large.bin";
  agile::writeVectorFile(output, raw);
  const char *maskOutput = "../test/data/dicom/parsed/mask_really_large.bin";
  agile::writeVectorFile(maskOutput, mask);
}

TEST_F(Test_Dicom, FindNormalizationFactor)
{
  RawDataPreparation rdp;
  std::vector<RType> data;
  unsigned cnt = 0;
  while (cnt < 30)
    data.push_back(30 - cnt++);

  EXPECT_EQ(28.5, rdp.FindNormalizationFactor(data));
}

TEST_F(Test_Dicom, DISABLED_ReadMeasDatCompleteDataRemoveOSAndNormalize)
{
  std::string measdat("../test/data/dicom/test_large.dat");
  const char *output = "../test/data/dicom/parsed/test_large.bin";

  std::vector<CType> raw;
  std::vector<RType> mask;
  Dimension dim;
  RawDataPreparation rdp(true, true, true);
  rdp.PrepareRawData(measdat, raw, mask, dim);

  EXPECT_EQ(208u, dim.width);
  EXPECT_EQ(144u, dim.height);
  EXPECT_EQ(18u, dim.coils);
  EXPECT_EQ(26u, dim.frames);

  std::cout << "Rawdata length: " << raw.size() << std::endl;
  EXPECT_EQ(14017536u, raw.size());

  agile::writeVectorFile(output, raw);
}

void GenerateChopMatrix(std::vector<RType> &chopMatrix, unsigned width,
                        unsigned height)
{
  chopMatrix.resize(width * height);

  for (unsigned x = 0; x < width; x++)
    for (unsigned y = 0; y < height; y++)
      chopMatrix[x + y * width] = std::pow(-1.0, x + y);
}

TEST_F(Test_Dicom, GenerateChopMatrix)
{
  unsigned width = 128;
  unsigned height = 128;
  std::vector<RType> data(width * height);

  RawDataPreparation rdp;

  rdp.GenerateChopMatrix(data, width, height);

  EXPECT_EQ(width * height, data.size());

  EXPECT_NEAR(1.0, data[0], EPS);
  EXPECT_NEAR(-1.0, data[1], EPS);
  EXPECT_NEAR(1.0, data[2], EPS);
  EXPECT_NEAR(-1.0, data[3], EPS);
  agile::writeVectorFile("../test/data/dicom/test_chop.bin", data);
}

TEST_F(Test_Dicom, DISABLED_ApplyChopToMaskAndData)
{
  std::vector<CType> data;
  std::vector<RType> mask;

  const char *dataPath = "../test/data/dicom/parsed/test_really_large.bin";
  agile::readVectorFile(dataPath, data);
  const char *maskPath = "../test/data/dicom/parsed/mask_really_large.bin";
  agile::readVectorFile(maskPath, mask);

  std::vector<RType> chop;
  Dimension dim(208, 144, 208, 144, 18, 26);

  RawDataPreparation rdp;
  rdp.ChopData(data, mask, dim);
  agile::writeVectorFile("../test/data/dicom/parsed/data_chopped.bin", data);
  agile::writeVectorFile("../test/data/dicom/parsed/mask_chopped.bin", mask);
}

TEST_F(Test_Dicom, DISABLED_ReadMeasDatCompleteDataRemoveOSNormalizeAndChop)
{
  std::string measdat("../test/data/dicom/test_large.dat");
  const char *output = "../test/data/dicom/parsed/test_large.bin";

  std::vector<CType> raw;
  std::vector<RType> mask;
  Dimension dim;
  RawDataPreparation rdp(true, true, true, true);
  rdp.PrepareRawData(measdat, raw, mask, dim);

  EXPECT_EQ(208u, dim.width);
  EXPECT_EQ(144u, dim.height);
  EXPECT_EQ(18u, dim.coils);
  EXPECT_EQ(26u, dim.frames);

  std::cout << "Rawdata length: " << raw.size() << std::endl;
  EXPECT_EQ(14017536u, raw.size());

  agile::writeVectorFile(output, raw);
}

