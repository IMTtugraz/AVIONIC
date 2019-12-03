#ifndef INCLUDE_DICOM_LOADER_H_

#define INCLUDE_DICOM_LOADER_H_

#include <boost/filesystem.hpp>
#include <iostream>
#include "agile/io/readSiemensVD11.hpp"
#include "agile/io/dicom.hpp"
#include "../include/siemens_vd11_reader.h"

/**
 * \brief
 *
 *
 */
namespace fs = boost::filesystem;

// typedef std::multimap<std::time_t, fs::path> DicomFileList;
typedef std::multimap<std::string, fs::path> DicomFileList;

class DicomReader : public SiemensVD11Reader
{
 public:
  DicomReader(OptionsParser &op);
  virtual ~DicomReader();

  void LoadRawData();

  DicomFileList GetDicomFileList();

  void GenerateMeasDat(const std::string &rawDataFile);

 private:
  const std::string &filepath;
};

#endif  // INCLUDE_DICOM_LOADER_H_
