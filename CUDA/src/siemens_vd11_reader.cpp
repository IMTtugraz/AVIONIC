#include "../include/siemens_vd11_reader.h"

SiemensVD11Reader::SiemensVD11Reader(OptionsParser &op)
  : RawDataReader(op), rawDataPath(op.kdataFilename)
{
}

SiemensVD11Reader::SiemensVD11Reader(OptionsParser &op,
                                     const std::string &rawDataPath)
  : RawDataReader(op), rawDataPath(rawDataPath)
{
}

SiemensVD11Reader::~SiemensVD11Reader()
{
}

//TODO: check for 3d dimensions (right now just rows2)
void SiemensVD11Reader::InitRawDataDimensions()
{
  unsigned int rows, cols, rows2, coils;
  unsigned short int acq, sli, par, echo, pha, rep, set, seg;

  rawData.getRawdata_info(rows, cols, coils, acq, sli, par, echo, pha, rep, set,
                          seg);

  std::cout << "Rows: " << rows << ", Columns: " << cols << ", Coils: " << coils
            << std::endl;

  std::cout << "Acquisition: " << acq << ", Slice:" << sli
            << ", Partition:" << par << std::endl;
  std::cout << "Echo: " << echo << ", Phase:" << pha << ", Repetition:" << rep
            << std::endl;
  std::cout << "Set: " << set << ", Segment:" << seg << ", Repetition:" << rep
            << std::endl;

  // TODO check Phase value and fix it
  rawDataDims = Dimension(cols, rows, rows2, cols, rows, rows2, coils, pha + 1);
}

void SiemensVD11Reader::LoadRawData()
{
  std::cout << "Load RawData: " << rawDataPath << std::endl;
  rawData = agile::ReadSiemensVD11(rawDataPath);
  int error = rawData.readfile(true);
  std::cout << "ReadFile Ret.Code:" << error << std::endl;

  InitRawDataDimensions();

  rawDataList = rawData.getRawdataList();
  std::cout << "RawDataDims: width " << rawDataDims.width << " height"
            << rawDataDims.height << std::endl;
}

Dimension SiemensVD11Reader::GetRawDataDimensions() const
{
  return rawDataDims;
}

unsigned SiemensVD11Reader::GetCenterRow() const
{
  agile::VD11_Image img;
  return img.getUshConsistentKSpaceCentreLineNo();
}

unsigned SiemensVD11Reader::GetCenterColumn() const
{
  agile::VD11_Image img;
  return img.getUshConsistentKSpaceCentreColumn();
}

unsigned SiemensVD11Reader::GetNumberOfAcquisitions() const
{
  return rawDataList.size();
}

bool SiemensVD11Reader::IsNonUniformData() const
{
  return op.nonuniform;
}

bool SiemensVD11Reader::IsOversampledData() const
{
  // Siemens raw data acquisitions always contain oversampling factor 2 in
  // readout direction
  return true;
}


Acquisition SiemensVD11Reader::GetAcquisition(unsigned index) const
{
  // TODO check index range
  std::list<agile::VD11_Image>::const_iterator it = rawDataList.begin();
  std::advance(it, index);
  agile::VD11_Image img = *it;
  Acquisition acq;
  acq.line = img.get_ushLine();
  acq.phase = img.getUshPhase();
  acq.centerColumn = img.ushKSpaceCentreColumn;
  acq.centerRow = img.ushKSpaceCentreLineNo;
  acq.readouts = rawDataDims.readouts;

  // Set coil data
  for (unsigned cnt = 0; cnt < rawDataDims.coils; cnt++)
  {
    acq.data.insert(std::make_pair(cnt, img.get_channeldata(cnt)));
  }

  // Set trajectory mask data
  if (this->IsNonUniformData())
    this->GenerateRadialTrajectory(acq.line, acq.traj, acq.dens, rawDataDims.encodings, rawDataDims.readouts);

  return acq;
}

