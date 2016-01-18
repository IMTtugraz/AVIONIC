#include "../include/raw_data_reader.h"

RawDataReader::RawDataReader(OptionsParser &op) : op(op)
{
}

RawDataReader::~RawDataReader()
{
}

void RawDataReader::GenerateRadialTrajectory(unsigned lineIdx,
                                             std::vector<RType> &traj,
                                             std::vector<RType> &dens,
                                             unsigned nEnc, unsigned nRO) const
{
  traj.resize(2 * nRO);
  dens.resize(nRO);

  for (unsigned enc = 0; enc < nRO; enc++)
  {
    RType rho = enc / (RType)nRO - (RType)0.5;
    RType phi = lineIdx / (RType)nEnc * 1.0 * M_PI;
    // x
    traj[enc] = rho * cos(phi);
    // y
    traj[enc + nRO] = rho * sin(phi);
    // density compensation
    dens[enc] = std::abs(rho);
  }
}

