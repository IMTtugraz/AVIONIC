#ifndef INCLUDE_OPTIONS_PARSER_H_

#define INCLUDE_OPTIONS_PARSER_H_

#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/regex.hpp>
#include <fstream>
#include <iostream>
#include "../include/ictgv2.h"
#include "../include/ictv.h"
#include "../include/tgv2.h"
#include "../include/tgv2_3d.h"
#include "../include/h1_recon.h"
#include "../include/tv.h"
#include "../include/tv_temp.h"
#include "../include/coil_construction.h"
#include "agile/agile.hpp"
#include "agile/io/file.hpp"

namespace po = boost::program_options;

/**
 * \brief Reconstruction method option enum
 *
 */
typedef enum Method
{
  TV,
  TVtemp,
  TGV2,
  TGV2_3D,
  ICTV,
  ICTGV2,
  BS_RECON
} Method;

/**
 * \brief GpuNUFFT parameter collection
 */
typedef struct GpuNUFFTParams
{
  GpuNUFFTParams()
  {
  }
  GpuNUFFTParams(DType kernelWidth, DType sectorWidth, DType osf)
    : kernelWidth(kernelWidth), sectorWidth(sectorWidth), osf(osf)
  {
  }
  DType kernelWidth;
  DType sectorWidth;
  DType osf;
} GpuNUFFTParams;

/**
 * \brief Options parser for command line input arguments
 *
 * Extracts reconstruction method, problem dimensions, etc.
 */
class OptionsParser
{
 public:
  OptionsParser();
  virtual ~OptionsParser();
  void Usage();

  bool ParseOptions(int argc, char *argv[]);

  TVParams tvParams;
  TVtempParams tvtempParams;
  TGV2Params tgv2Params;
  ICTVParams ictvParams;
  ICTGV2Params ictgv2Params;
  TGV2_3DParams tgv2_3DParams;
  CoilConstructionParams coilParams;
  H1Params h1Params;


  std::string kdataFilename;
  std::string maskFilename;
  std::string kdataFilenameH1;
  std::string maskFilenameH1;
  std::string outputFilename;
  std::string outputFilenameFinal;

  Method method;
  Dimension dims;
  bool verbose;
  int debugstep;
  unsigned int slice;
  int tpat;
  int gpu_device_nr;
  
  std::string sensitivitiesFilename;
  std::string u0Filename;
  std::string densityFilename;
  bool nonuniform;
  bool normalize;
  bool extradata;
  GpuNUFFTParams gpuNUFFTParams;
  AdaptLambdaParams adaptLambdaParams;
  bool rawdata;
  bool forceOSRemoval;

 private:
  po::options_description desc;
  po::options_description conf;
  po::options_description hidden;

  std::string parameterFile;

  po::options_description cmdline_options;
  po::options_description config_options;
  po::options_description visible;
  po::positional_options_description p;

  void AddCoilConstrConfigurationParameters();

  void AddTVConfigurationParameters();

  void AddTVtempConfigurationParameters();

  void AddTGV2ConfigurationParameters();

  void AddTGV2_3DConfigurationParameters();

  void AddICTVConfigurationParameters();

  void AddICTGV2ConfigurationParameters();

  void AddGPUNUFFTConfigurationParameters();

  void AddH1ConfigurationParameters();

  void AddAdaptLambdaConfigurationParameters();

  void SetMaxIt(int maxIt);
  
  void SetStopPDGap(float stopPDGap);

  void SetAdaptLambdaParams();
};

std::istream &operator>>(std::istream &in, Method &method);

void validate(boost::any &v, const std::vector<std::string> &values,
              Dimension *target, int c);

void validate3D(boost::any &v, const std::vector<std::string> &values,
              Dimension *target, int c);

#endif  // INCLUDE_OPTIONS_PARSER_H_
