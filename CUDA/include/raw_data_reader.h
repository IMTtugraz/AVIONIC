#ifndef INCLUDE_RAW_DATA_READER_H_

#define INCLUDE_RAW_DATA_READER_H_

#include "./options_parser.h"
#include <map>

/**
 * \brief
 *
 *
 */
typedef struct Acquisition
{
  std::map<unsigned, std::vector<CType> /*formatter*/> data;
  std::vector<RType> traj;
  std::vector<RType> dens;

  unsigned line;
  unsigned phase;
  unsigned slice;

  unsigned readouts;
  unsigned centerRow;
  unsigned centerColumn;

  bool hasTrajectoryInformation()
  {
    return traj.size() > 0;
  };

  bool isNoiseMeasurement;
} Acquisition;

/**
 * \brief
 *
 */
class RawDataReader
{
 public:
  RawDataReader(OptionsParser &op);
  virtual ~RawDataReader();

  virtual void LoadRawData() = 0;
  virtual Dimension GetRawDataDimensions() const = 0;

  virtual unsigned GetCenterRow() const = 0;
  virtual unsigned GetCenterColumn() const = 0;

  virtual unsigned GetNumberOfAcquisitions() const = 0;
  virtual Acquisition GetAcquisition(unsigned index) const = 0;

  virtual bool IsNonUniformData() const = 0;
  virtual bool IsOversampledData() const = 0;

 protected:
  OptionsParser &op;

  void GenerateRadialTrajectory(unsigned lineIdx, std::vector<RType> &traj,
                                std::vector<RType> &dens, unsigned nEnc,
                                unsigned nRO) const;

 private:
};

#endif  // INCLUDE_RAW_DATA_READER_H_
