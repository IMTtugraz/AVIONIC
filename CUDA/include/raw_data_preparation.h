#ifndef INCLUDE_RAW_DATA_PREPARATION_H_

#define INCLUDE_RAW_DATA_PREPARATION_H_

#include "../include/options_parser.h"
#include "agile/io/dicom.hpp"
#include "agile/io/readSiemensVD11.hpp"
#include "agile/calc/fft.hpp"
#include "../include/types.h"
#include "../include/coil_construction.h"
#include "../include/noncartesian_coil_construction.h"
#include "../include/cartesian_coil_construction.h"
#include "../include/raw_data_reader.h"

/**
 * \brief
 *
 *
 */
class RawDataPreparation
{
 public:
  RawDataPreparation(OptionsParser &op, bool completeData = false,
                     bool removeReadOutOS = false, bool normalizeData = false,
                     bool applyChop = false);

  RawDataPreparation(bool completeData = false, bool removeReadOutOS = false,
                     bool normalizeData = false, bool applyChop = false);

  virtual ~RawDataPreparation();

  void PrepareDicomData(const std::string &dicomDataPath,
                        const std::string &rawDataPath,
                        std::vector<CType> &data, std::vector<RType> &mask,
                        std::vector<RType> &w, Dimension &dims);

  void PrepareIsmrmrdData(const std::string &rawDataPath,
                          std::vector<CType> &data, std::vector<RType> &mask,
                          std::vector<RType> &w, Dimension &dims);

  void PrepareRawData(CVector &kdata, RVector &mask, RVector &w,
                      Dimension &dims);

  void PrepareRawData(const std::string &rawDataPath, std::vector<CType> &data,
                      std::vector<RType> &mask, Dimension &dims);

  void PrepareRawData(std::vector<CType> &data, std::vector<RType> &mask,
                      std::vector<RType> &w, Dimension &dims);

  void NormalizeData(std::vector<CType> &data, const Dimension &dims,
                     CoilConstruction *coilConstruction);

  void NormalizeNonCartData(std::vector<CType> &data, const Dimension &dims,
                            NoncartesianCoilConstruction *coilConstruction);

  void GenerateChopMatrix(std::vector<RType> &chopMatrix, unsigned width,
                          unsigned height);

  void ChopData(std::vector<CType> &data, std::vector<RType> &mask,
                const Dimension &dims);

  RType FindNormalizationFactor(std::vector<RType> &data);

 private:
  OptionsParser &op;
  communicator_type com;
  bool completeData;
  bool removeReadOutOS;
  bool normalizeData;
  bool applyChop;
  bool nonuniformData;

  void InitOSRemoval(unsigned width);

  void RemoveOS(std::vector<CType> &oversampledLine,
                std::vector<CType> &croppedLine, unsigned width,
                unsigned colOffset);

  void ComputePFZeroFillOffset(const Acquisition &line, unsigned centerCol,
                               unsigned centerRow, unsigned &colOffset,
                               unsigned &rowOffset, unsigned &lineStart,
                               unsigned &lineEnd);

  void SetTrajectoryData(Acquisition &line, std::vector<RType> &mask,
                         std::vector<RType> &w, const Dimension &dims,
                         unsigned maskPhaseOffset, unsigned lineOffset,
                         unsigned colOffset);

  void SetData(Acquisition &line, std::vector<CType> &data,
               const Dimension &dims, unsigned phaseOffset, unsigned lineOffset,
               unsigned colOffset);

  void NormalizeData(std::vector<CType> &data, std::vector<RType> &mask,
                     std::vector<RType> &w, Dimension &dims);

  // Operators/Vectors used in remove oversampling path
  agile::FFT<CType> *fftOp;
  agile::FFT<CType> *croppedFftOp;
  RawDataReader *dataReader;
  std::vector<CType> fullLine;
  CVector fullLineGPU;
  CVector croppedLineGPU;
  Dimension rawDataDims;
};

#endif  // INCLUDE_RAW_DATA_PREPARATION_H_
