#ifndef INCLUDE_OPTIONS_PARSER_H_

#define INCLUDE_OPTIONS_PARSER_H_

#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/regex.hpp>
#include <fstream>
#include <iostream>
#include "../include/ictgv2.h"
#include "../include/tv.h"
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
  TGV2,
  ICTGV2
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
  TGV2Params tgv2Params;
  ICTGV2Params ictgv2Params;
  CoilConstructionParams coilParams;

  std::string kdataFilename;
  std::string maskFilename;
  std::string outputFilename;

  Method method;
  Dimension dims;
  bool verbose;
  int debugstep;

  std::string sensitivitiesFilename;
  std::string u0Filename;
  std::string densityFilename;
  bool nonuniform;
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

  void AddTGV2ConfigurationParameters();

  void AddICTGV2ConfigurationParameters();

  void AddGPUNUFFTConfigurationParameters();

  void AddAdaptLambdaConfigurationParameters();

  void SetMaxIt(int maxIt);
  void SetAdaptLambdaParams();
};

std::istream &operator>>(std::istream &in, Method &method);

void validate(boost::any &v, const std::vector<std::string> &values,
              Dimension *target, int c);

#endif  // INCLUDE_OPTIONS_PARSER_H_
