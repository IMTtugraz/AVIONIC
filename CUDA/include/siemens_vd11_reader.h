#ifndef INCLUDE_SIEMENS_VD11_READER_H_

#define INCLUDE_SIEMENS_VD11_READER_H_

#include <string>
#include "agile/io/readSiemensVD11.hpp"
#include "../include/raw_data_reader.h"

class SiemensVD11Reader : public RawDataReader
{
 public:
  SiemensVD11Reader(OptionsParser &op);
  virtual ~SiemensVD11Reader();

  Dimension GetRawDataDimensions() const;

  unsigned GetCenterRow() const;
  unsigned GetCenterColumn() const;

  unsigned GetNumberOfAcquisitions() const;
  Acquisition GetAcquisition(unsigned index) const;

  bool IsNonUniformData() const;
  bool IsOversampledData() const;

  void LoadRawData();

 protected:
  SiemensVD11Reader(OptionsParser &op, const std::string &rawDataPath);
  std::string rawDataPath;
  agile::ReadSiemensVD11 rawData;
  Dimension rawDataDims;

  std::list<agile::VD11_Image> rawDataList;

  void InitRawDataDimensions();
};

#endif  // INCLUDE_SIEMENS_VD11_READER_H_
