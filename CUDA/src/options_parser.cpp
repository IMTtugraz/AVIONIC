#include "../include/options_parser.h"
#include <boost/lexical_cast.hpp>
#include "../include/config_dir.h"

std::istream &operator>>(std::istream &in, Method &method)
{
  std::string token;
  in >> token;
  token = boost::to_upper_copy(token);

  if (token == "TV")
  {
    method = TV;
  }
  else if (token == "TVTEMP")
  {
    method = TVtemp;
  }
  else if (token == "TGV2")
  {
    method = TGV2;
  }
  else if (token == "TGV2_3D")
  {
    method = TGV2_3D;
  }
  else if (token == "ICTV")
  {
    method = ICTV;
  } 
  else if (token == "ICTGV2")
  {
    method = ICTGV2;
  }
  else if (token == "BS_RECON")
  {
	method = BS_RECON;
  }
  else if (token == "ASLTGV2RECON4D")
  {
    method = ASLTGV2RECON4D;
  }									 
  else
  {
    throw std::runtime_error("invalid method selected");
  }
  return in;
}

void validate(boost::any &v, const std::vector<std::string> &values,
              Dimension *target, int c)
{
  static boost::regex r("(\\d.*):(\\d.*):(\\d.*):(\\d.*):(\\d.*):(\\d.*):(\\d.*):(\\d.*)");
  // po::validators::check_first_occurence(v);
  const std::string &s = po::validators::get_single_string(values);
  boost::smatch match;

  if (boost::regex_match(s, match, r))
  {
    unsigned width = boost::lexical_cast<unsigned>(match[1]);
    unsigned height = boost::lexical_cast<unsigned>(match[2]);
    unsigned depth = boost::lexical_cast<unsigned>(match[3]);
    unsigned readouts = boost::lexical_cast<unsigned>(match[4]);
    unsigned encodings = boost::lexical_cast<unsigned>(match[5]);
    unsigned encodings2 = boost::lexical_cast<unsigned>(match[6]);
    unsigned coils = boost::lexical_cast<unsigned>(match[7]);
    unsigned frames = boost::lexical_cast<unsigned>(match[8]);
    v = Dimension(width, height, depth, readouts, encodings, encodings2, coils, frames);
  }
  else
  {
    throw std::runtime_error("invalid dimensions");
  }
}

OptionsParser::OptionsParser()
  : desc("Allowed options"), conf("Configuration"), hidden("Hidden options")
{
    desc.add_options()("help,h", "show help message")("debugstep,g", po::value<int>(&debugstep)->default_value(10),"flag to export PDGap")(
      "verbose,v", po::bool_switch(&verbose)->default_value(false),
      "verbose console output")("dims,d", po::value<Dimension>(),
                                "Data dimensions conforming to "
                                "width:height:depth:readouts:encodings_y:encodings_z:"
                                "coils:frames")(
      "nonuniform,n", po::bool_switch(&nonuniform)->default_value(false),
      "flag to indicate nonuniform data")(
                "normalize,o", po::bool_switch(&normalize)->default_value(false),
                "flag to indicate data normalization")(
      "extradata,e", po::bool_switch(&extradata)->default_value(false),
      "flag to enable export of additional result data")(
      "adaptlambda,a",
      po::bool_switch(&adaptLambdaParams.adaptLambda)->default_value(false),
      "flag to enable dynamic adaptation of lambda depending on [adaptlambda] "
      "configuration "
      "parameters (k,d)")("dens,w", po::value<std::string>(&densityFilename),
                         "Density compensation data.")(
      "params,p", po::value<std::string>(&parameterFile)->default_value(
                      std::string(boost::lexical_cast<std::string>(CONFIG_DIR) +
                                  "default.cfg").c_str()),
      "parameter configuration file")(
      "sens,s", po::value<std::string>(&sensitivitiesFilename),
      "Coil sensitivity data.")("uzero,u", po::value<std::string>(&u0Filename),
                                "Initial image u0.")(
	  "labeldata,l", po::value<std::string>(&kdataLFilename),
      "Labeled ASL data.")(															   
      "rawdata,r", po::bool_switch(&rawdata)->default_value(false),
      "flag to indicate raw data import")(
      "forceOSRemoval,f", po::bool_switch(&forceOSRemoval)->default_value(false),
      "flag to force OS removal in raw data preparation")(
      "tpat,t", po::value<int>(&tpat)->default_value(1),
      "artifical TPAT interleave")(
      "slice,z", po::value<unsigned int>(&slice)->default_value(0),
      "slice to reconstruct")("gpudevice,b", po::value<int>(&gpu_device_nr)->default_value(-1),"GPU Device Nr")
      ("dataBS,q", po::value<std::string>(&kdataFilenameH1), "file name data for BS reconstruction")
      ("maskBS,x", po::value<std::string>(&maskFilenameH1),  "file name mask for BS reconstruction")
      ("finalOutBS,y", po::value<std::string>(&outputFilenameFinal), "output file name for BS map");

  conf.add_options()("method,m", po::value<Method>()->default_value(ICTGV2),
                     "reconstruction method (TV, TVTEMP, TGV2, TGV2_3D, ICTV, ICTGV2, BS_RECON, ASLTGV2RECON4D)")(
      "maxIt,i", po::value<int>()->default_value(500),
      "Maximum number of iterations")(
          "stopPDGap,j", po::value<float>()->default_value(0),
          "use PDGap as stopping criterion");

  AddCoilConstrConfigurationParameters();
  AddTVConfigurationParameters();
  AddTVtempConfigurationParameters();
  AddTGV2ConfigurationParameters();
  AddTGV2_3DConfigurationParameters();
  AddICTVConfigurationParameters(); 
  AddICTGV2ConfigurationParameters();
  AddASLTGV2RECON4DConfigurationParameters();										
  AddGPUNUFFTConfigurationParameters();
  AddH1ConfigurationParameters();
  AddAdaptLambdaConfigurationParameters();

  hidden.add_options()("kdata", po::value<std::string>(), "k-space data file")(
      "mask", po::value<std::string>(), "k-space mask file")(
      "output", po::value<std::string>(), "reconstructed image");

  p.add("kdata", 1);
  p.add("mask", 1);
  p.add("output", 1);

  cmdline_options.add(desc).add(conf).add(hidden);

  config_options.add(conf).add(hidden);

  visible.add(desc);
}

OptionsParser::~OptionsParser()
{
}

void OptionsParser::Usage()
{
  std::cout << "Usage: avionic [options] [-d w:h:d:nRO:nEnc1:nEnc2:c:f "
               "<kdata> <mask/traj> | -r <rawdata>] <output>\n";
  std::cout << "\nAVIONIC: Accelerated Variational dynamic MRI Reconstruction \n\n";
  std::cout << "Required:\n";
  std::cout << "  Either:\n";
  std::cout << "\t  -d \t problem dimensions (see below)\n";
  std::cout << "\t  kdata \t acquired k-space data file \n";
  std::cout << "\t  mask/traj \t k-space sampling pattern (trajectory)\n";
  std::cout << "  or:  \n";
  std::cout << "\t  -r \t enable raw data import (see below)\n";
  std::cout << "\t  rawdata \t acquired rawdata file/directory\n";
  std::cout << "  output \t reconstructed output image\n";
  std::cout << visible << std::endl;
}

void OptionsParser::AddCoilConstrConfigurationParameters()
{
  conf.add_options()("coil.uH1mu", po::value<RType>(&coilParams.uH1mu))(
      "coil.uReg", po::value<RType>(&coilParams.uReg))(
      "coil.uNrIt", po::value<unsigned>(&coilParams.uNrIt))(
      "coil.uTau", po::value<RType>(&coilParams.uTau))(
      "coil.uSigma", po::value<RType>(&coilParams.uSigma))(
      "coil.uSigmaTauRatio", po::value<RType>(&coilParams.uSigmaTauRatio))(
      "coil.uAlpha0", po::value<RType>(&coilParams.uAlpha0))(
      "coil.uAlpha1", po::value<RType>(&coilParams.uAlpha1))(
      "coil.b1Reg", po::value<RType>(&coilParams.b1Reg))(
      "coil.b1NrIt", po::value<unsigned>(&coilParams.b1NrIt))(
      "coil.b1Tau", po::value<RType>(&coilParams.b1Tau))(
      "coil.b1Sigma", po::value<RType>(&coilParams.b1Sigma))(
      "coil.b1SigmaTauRatio", po::value<RType>(&coilParams.b1SigmaTauRatio))(
      "coil.b1FinalReg", po::value<RType>(&coilParams.b1FinalReg))(
      "coil.b1FinalNrIt", po::value<unsigned>(&coilParams.b1FinalNrIt));
}

void OptionsParser::AddTVConfigurationParameters()
{
  conf.add_options()("tv.dx", po::value<RType>(&tvParams.dx))(
      "tv.dy", po::value<RType>(&tvParams.dy))(
      "tv.dt", po::value<RType>(&tvParams.dt))(
      "tv.sigma", po::value<RType>(&tvParams.sigma))(
      "tv.tau", po::value<RType>(&tvParams.tau))(
      "tv.sigmaTauRatio", po::value<RType>(&tvParams.sigmaTauRatio))(
      "tv.timeSpaceWeight", po::value<RType>(&tvParams.timeSpaceWeight))(
      "tv.lambda", po::value<RType>(&tvParams.lambda));
}

void OptionsParser::AddTVtempConfigurationParameters()
{
  conf.add_options()("tvtemp.dt", po::value<RType>(&tvtempParams.dt))(
      "tvtemp.sigma", po::value<RType>(&tvtempParams.sigma))(
      "tvtemp.tau", po::value<RType>(&tvtempParams.tau))(
      "tvtemp.sigmaTauRatio", po::value<RType>(&tvtempParams.sigmaTauRatio))(
      "tvtemp.lambda", po::value<RType>(&tvtempParams.lambda));
}

void OptionsParser::AddTGV2ConfigurationParameters()
{
  conf.add_options()("tgv2.dx", po::value<RType>(&tgv2Params.dx))(
      "tgv2.dy", po::value<RType>(&tgv2Params.dy))(
      "tgv2.dt", po::value<RType>(&tgv2Params.dt))(
      "tgv2.sigma", po::value<RType>(&tgv2Params.sigma))(
      "tgv2.tau", po::value<RType>(&tgv2Params.tau))(
      "tgv2.sigmaTauRatio", po::value<RType>(&tgv2Params.sigmaTauRatio))(
      "tgv2.timeSpaceWeight", po::value<RType>(&tgv2Params.timeSpaceWeight))(
      "tgv2.lambda", po::value<RType>(&tgv2Params.lambda))(
      "tgv2.alpha0", po::value<RType>(&tgv2Params.alpha0))(
      "tgv2.alpha1", po::value<RType>(&tgv2Params.alpha1));
}

void OptionsParser::AddTGV2_3DConfigurationParameters()
{
  conf.add_options()("tgv2_3D.dx", po::value<RType>(&tgv2_3DParams.dx))(
      "tgv2_3D.dy", po::value<RType>(&tgv2_3DParams.dy))(
      "tgv2_3D.dz", po::value<RType>(&tgv2_3DParams.dz))(
      "tgv2_3D.sigma", po::value<RType>(&tgv2_3DParams.sigma))(
      "tgv2_3D.tau", po::value<RType>(&tgv2_3DParams.tau))(
      "tgv2_3D.sigmaTauRatio", po::value<RType>(&tgv2_3DParams.sigmaTauRatio))(
      "tgv2_3D.lambda", po::value<RType>(&tgv2_3DParams.lambda))(
      "tgv2_3D.alpha0", po::value<RType>(&tgv2_3DParams.alpha0))(
      "tgv2_3D.alpha1", po::value<RType>(&tgv2_3DParams.alpha1));
}

void OptionsParser::AddICTVConfigurationParameters()
{
  conf.add_options()("ictv.dx", po::value<RType>(&ictvParams.dx))(
      "ictv.dy", po::value<RType>(&ictvParams.dy))(
      "ictv.dt", po::value<RType>(&ictvParams.dt))(
      "ictv.sigma", po::value<RType>(&ictvParams.sigma))(
      "ictv.tau", po::value<RType>(&ictvParams.tau))(
      "ictv.sigmaTauRatio", po::value<RType>(&ictvParams.sigmaTauRatio))(
      "ictv.timeSpaceWeight",
      po::value<RType>(&ictvParams.timeSpaceWeight))(
      "ictv.lambda", po::value<RType>(&ictvParams.lambda))(
      "ictv.alpha1", po::value<RType>(&ictvParams.alpha1))(
      "ictv.alpha", po::value<RType>(&ictvParams.alpha))(
      "ictv.timeSpaceWeight2",
      po::value<RType>(&ictvParams.timeSpaceWeight2))(
      "ictv.dx2", po::value<RType>(&ictvParams.dx2))(
      "ictv.dy2", po::value<RType>(&ictvParams.dy2))(
      "ictv.dt2", po::value<RType>(&ictvParams.dt2));
}
void OptionsParser::AddICTGV2ConfigurationParameters()
{
  conf.add_options()("ictgv2.dx", po::value<RType>(&ictgv2Params.dx))(
      "ictgv2.dy", po::value<RType>(&ictgv2Params.dy))(
      "ictgv2.dt", po::value<RType>(&ictgv2Params.dt))(
      "ictgv2.sigma", po::value<RType>(&ictgv2Params.sigma))(
      "ictgv2.tau", po::value<RType>(&ictgv2Params.tau))(
      "ictgv2.sigmaTauRatio", po::value<RType>(&ictgv2Params.sigmaTauRatio))(
      "ictgv2.timeSpaceWeight",
      po::value<RType>(&ictgv2Params.timeSpaceWeight))(
      "ictgv2.lambda", po::value<RType>(&ictgv2Params.lambda))(
      "ictgv2.alpha0", po::value<RType>(&ictgv2Params.alpha0))(
      "ictgv2.alpha1", po::value<RType>(&ictgv2Params.alpha1))(
      "ictgv2.alpha", po::value<RType>(&ictgv2Params.alpha))(
      "ictgv2.timeSpaceWeight2",
      po::value<RType>(&ictgv2Params.timeSpaceWeight2))(
      "ictgv2.dx2", po::value<RType>(&ictgv2Params.dx2))(
      "ictgv2.dy2", po::value<RType>(&ictgv2Params.dy2))(
      "ictgv2.dt2", po::value<RType>(&ictgv2Params.dt2));
}

void OptionsParser::AddASLTGV2RECON4DConfigurationParameters()
{
    conf.add_options()("asltgv2recon4D.dx", po::value<RType>(&asltgv2recon4DParams.dx))(
      "asltgv2recon4D.dy", po::value<RType>(&asltgv2recon4DParams.dy))(
      "asltgv2recon4D.dz", po::value<RType>(&asltgv2recon4DParams.dz))(
      "asltgv2recon4D.dt", po::value<RType>(&asltgv2recon4DParams.dt))(
      "asltgv2recon4D.sigma", po::value<RType>(&asltgv2recon4DParams.sigma))(
      "asltgv2recon4D.tau", po::value<RType>(&asltgv2recon4DParams.tau))(
      "asltgv2recon4D.sigmaTauRatio", po::value<RType>(&asltgv2recon4DParams.sigmaTauRatio))(
      "asltgv2recon4D.timeSpaceWeight",
      po::value<RType>(&asltgv2recon4DParams.timeSpaceWeight))(
      "asltgv2recon4D.lambda_l", po::value<RType>(&asltgv2recon4DParams.lambda_l))(
      "asltgv2recon4D.lambda_c", po::value<RType>(&asltgv2recon4DParams.lambda_c))(
      "asltgv2recon4D.alpha0", po::value<RType>(&asltgv2recon4DParams.alpha0))(
      "asltgv2recon4D.alpha1", po::value<RType>(&asltgv2recon4DParams.alpha1))(
      "asltgv2recon4D.alpha", po::value<RType>(&asltgv2recon4DParams.alpha));
}
void OptionsParser::AddGPUNUFFTConfigurationParameters()
{
  conf.add_options()(
      "gpunufft.kernelWidth",
      po::value<DType>(&gpuNUFFTParams.kernelWidth)->default_value(3.0))(
      "gpunufft.sectorWidth",
      po::value<DType>(&gpuNUFFTParams.sectorWidth)->default_value(8.0))(
      "gpunufft.osf",
      po::value<DType>(&gpuNUFFTParams.osf)->default_value(2.0));
}

void OptionsParser::AddH1ConfigurationParameters()
{
  conf.add_options()("h1.mu", po::value<RType>(&h1Params.mu))(
      "h1.relTol", po::value<RType>(&h1Params.relTol))(
      "h1.absTol", po::value<RType>(&h1Params.absTol))(
      "h1.maxIt", po::value<unsigned>(&h1Params.maxIt));
}



void OptionsParser::AddAdaptLambdaConfigurationParameters()
{
  conf.add_options()("adaptlambda.k", po::value<DType>(&adaptLambdaParams.k)
                                          ->default_value(0.0))(
      "adaptlambda.d",
      po::value<DType>(&adaptLambdaParams.d)->default_value(1.0));
}

void OptionsParser::SetMaxIt(int maxIt)
{
  tvParams.maxIt = maxIt;
  tvtempParams.maxIt = maxIt;
  tgv2Params.maxIt = maxIt;
  tgv2_3DParams.maxIt = maxIt;
  ictvParams.maxIt = maxIt;
  ictgv2Params.maxIt = maxIt;
  asltgv2recon4DParams.maxIt = maxIt;								
}

void OptionsParser::SetStopPDGap(float stopPDGap)
{
  tvParams.stopPDGap = stopPDGap;
  tvtempParams.stopPDGap = stopPDGap;
  tgv2Params.stopPDGap = stopPDGap;
  tgv2_3DParams.stopPDGap = stopPDGap;
  ictvParams.stopPDGap = stopPDGap;
  ictgv2Params.stopPDGap = stopPDGap;
}

void OptionsParser::SetAdaptLambdaParams()
{
  tvParams.adaptLambdaParams = adaptLambdaParams;
  tvtempParams.adaptLambdaParams = adaptLambdaParams;
  tgv2Params.adaptLambdaParams = adaptLambdaParams;
  tgv2_3DParams.adaptLambdaParams = adaptLambdaParams;
  ictvParams.adaptLambdaParams = adaptLambdaParams;
  ictgv2Params.adaptLambdaParams = adaptLambdaParams;
}

bool OptionsParser::ParseOptions(int argc, char *argv[])
{
  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv)
                .options(cmdline_options)
                .positional(p)
                .run(),
            vm);
  po::notify(vm);

  bool isRawDataPassed = vm.count("kdata") && rawdata;
  if (vm.count("help") ||
      !((vm.count("dims") && vm.count("kdata") && vm.count("mask") &&
         vm.count("output")) ||
        isRawDataPassed))
  {
    Usage();
    return false;
  }

  // in case of -r (rawdata) option
  // the positional parameters mask and output have to
  // swapped
  kdataFilename = vm["kdata"].as<std::string>();
  maskFilename = rawdata ? "" : vm["mask"].as<std::string>();
  outputFilename =
      rawdata ? vm["mask"].as<std::string>() : vm["output"].as<std::string>();
  dims = rawdata ? Dimension() : vm["dims"].as<Dimension>();  //< problem dims
  // are extracted
  // from meta-data

  std::ifstream ifs(parameterFile.c_str());
  if (ifs)
  {
    std::cout << "Parsing configuration file: " << parameterFile.c_str()
              << std::endl;
    po::store(po::parse_config_file(ifs, config_options), vm);
    po::notify(vm);
    ifs.close();
  }
  else
  {
    std::cout << "Warning: no configuration file found in "
              << parameterFile.c_str() << std::endl;
    std::cout << "         using default parameters... " << std::endl;
  }
  SetAdaptLambdaParams();

  method = vm["method"].as<Method>();
  SetMaxIt(vm["maxIt"].as<int>());
  SetStopPDGap(vm["stopPDGap"].as<float>());


  return true;
}
