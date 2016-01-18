#include "../include/dicom_reader.h"

DicomReader::DicomReader(OptionsParser &op)
  : SiemensVD11Reader(op, utils::GetParentDirectory(op.outputFilename) +
                              "/meas.dat"),
    filepath(op.kdataFilename)
{
}

DicomReader::~DicomReader()
{
}

DicomFileList DicomReader::GetDicomFileList()
{
  DicomFileList dicomFiles;
  fs::path dir(filepath);

  if (fs::exists(dir))
  {
    if (fs::is_directory(dir))
    {
      for (fs::directory_iterator dirIt(dir); dirIt != fs::directory_iterator();
           dirIt++)
      {
        if (fs::is_regular_file(dirIt->status()) &&
            dirIt->path().extension() == ".dcm")
        {
          dicomFiles.insert(
              // std::make_pair(fs::last_write_time(dirIt->path()),
              // dirIt->path()));
              std::make_pair(dirIt->path().filename().c_str(), dirIt->path()));
        }
      }
    }
  }
  else if (fs::is_regular_file(dir) && dir.extension() == ".dcm")
  {
    // in case path points to exactly one dicom file
    dicomFiles.insert(std::make_pair(dir.filename().c_str(), dir));
  }
  else
    std::cerr << "Error loading file/directory: " << dir << " -> not found!"
              << std::endl;

  return dicomFiles;
}

void DicomReader::GenerateMeasDat(const std::string &rawDataFile)
{
  DicomFileList dicomFiles = GetDicomFileList();
  DicomFileList::iterator dicomIt = dicomFiles.begin();
  std::ofstream measdat(rawDataFile.c_str());
  while (dicomIt != dicomFiles.end())
  {
    std::cout << "Loading file " << dicomIt->second.c_str() << std::endl;

    unsigned long length;
    Uint8 *pdata = NULL;

    agile::DICOM dicom;

    dicom.readdicom(dicomIt->second.c_str(), measdat, length, pdata);
    std::cout << length << " bytes read..." << std::endl;

    dicomIt++;
  }
  measdat.close();
}

void DicomReader::LoadRawData()
{
  std::cout << "Generate meas dat in:  " << rawDataPath << std::endl;
  this->GenerateMeasDat(rawDataPath);
  SiemensVD11Reader::LoadRawData();
}

