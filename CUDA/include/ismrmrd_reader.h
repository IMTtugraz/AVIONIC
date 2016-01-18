#ifndef INCLUDE_ISMRMRD_READER_H_

#define INCLUDE_ISMRMRD_READER_H_

#include "./raw_data_reader.h"
#include "ismrmrd/ismrmrd.h"
#include "ismrmrd/dataset.h"
#include "ismrmrd/xml.h"
#include "ismrmrd/version.h"

/**
 * \brief
 *
 *
 */
class IsmrmrdReader : public RawDataReader
{
 public:
  IsmrmrdReader(OptionsParser &op);
  virtual ~IsmrmrdReader();

  void LoadRawData();
  Dimension GetRawDataDimensions() const;

  unsigned GetCenterRow() const;
  unsigned GetCenterColumn() const;

  unsigned GetNumberOfAcquisitions() const;
  Acquisition GetAcquisition(unsigned index) const;

  bool IsNonUniformData() const;
  bool IsOversampledData() const;

 private:
  const std::string &filename;
  Dimension rawDataDims;
  ISMRMRD::Dataset *dataset;
  ISMRMRD::IsmrmrdHeader *hdr;

  void InitRawDataDimensions();
};

#endif  // INCLUDE_ISMRMRD_READER_H_
