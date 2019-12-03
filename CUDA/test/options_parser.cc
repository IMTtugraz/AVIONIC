#include <gtest/gtest.h>
#include "../include/options_parser.h"
#include <string>

class Test_Options : public ::testing::Test
{
 public:
  virtual void SetUp()
  {
  }
};

TEST_F(Test_Options, NoArgumentPassed)
{
  OptionsParser op;
  int argc = 0;
  const char *argv[] = {};

  EXPECT_FALSE(op.ParseOptions(argc, const_cast<char **>(argv)));
}

TEST_F(Test_Options, OnlyOneArgumentPassed)
{
  OptionsParser op;
  int argc = 2;
  const char *argv[] = { "./fredy_mri", "test.out" };
  EXPECT_FALSE(op.ParseOptions(argc, const_cast<char **>(argv)));
}

TEST_F(Test_Options, KDataMaskOutputAndDimsCorrectlyPassed)
{
  OptionsParser op;
  int argc = 6;
  const char *argv[] = { "./fredy_mri", "kdata.bin", "mask.bin",
                         "output.bin",  "-d",        "128:128:256:64:18:20" };
  EXPECT_TRUE(op.ParseOptions(argc, const_cast<char **>(argv)));

  EXPECT_EQ("output.bin", op.outputFilename);
  EXPECT_EQ("kdata.bin", op.kdataFilename);
  EXPECT_EQ("mask.bin", op.maskFilename);

  EXPECT_EQ(128u, op.dims.width);
  EXPECT_EQ(128u, op.dims.height);
  EXPECT_EQ(256u, op.dims.readouts);
  EXPECT_EQ(64u, op.dims.encodings);
  EXPECT_EQ(18u, op.dims.coils);
  EXPECT_EQ(20u, op.dims.frames);
}

TEST_F(Test_Options, RawDataParamsCorrectlyPassed)
{
  OptionsParser op;
  int argc = 4;
  const char *argv[] = { "./fredy_mri", "-r", "dataDir", "output.bin" };
  EXPECT_TRUE(op.ParseOptions(argc, const_cast<char **>(argv)));
  EXPECT_TRUE(op.rawdata);
  EXPECT_EQ("dataDir", op.kdataFilename);
  EXPECT_EQ("output.bin", op.outputFilename);
  EXPECT_EQ("", op.maskFilename);
}

TEST_F(Test_Options, NonUniformParamsPassed)
{
  OptionsParser op;
  int argc = 7;
  const char *argv[] = { "./fredy_mri", "kdata.bin", "traj.bin",
                         "output.bin",  "-d",        "128:128:256:64:18:20",
                         "-n" };
  EXPECT_TRUE(op.ParseOptions(argc, const_cast<char **>(argv)));
  EXPECT_TRUE(op.nonuniform);
}

TEST_F(Test_Options, VerboseFlagPassed)
{
  OptionsParser op;
  int argc = 7;
  const char *argv[] = { "./fredy_mri", "kdata.bin", "traj.bin",
                         "output.bin",  "-d",        "128:128:256:64:18:20",
                         "-v" };
  EXPECT_TRUE(op.ParseOptions(argc, const_cast<char **>(argv)));
  EXPECT_TRUE(op.verbose);
}

TEST_F(Test_Options, ExtraDataFlagPassed)
{
  OptionsParser op;
  int argc = 7;
  const char *argv[] = { "./fredy_mri", "kdata.bin", "traj.bin",
                         "output.bin",  "-d",        "128:128:256:64:18:20",
                         "-e" };
  EXPECT_TRUE(op.ParseOptions(argc, const_cast<char **>(argv)));
  EXPECT_TRUE(op.extradata);
}

TEST_F(Test_Options, AdaptLambdaFlagPassed)
{
  OptionsParser op;
  int argc = 7;
  const char *argv[] = { "./fredy_mri", "kdata.bin", "traj.bin",
                         "output.bin",  "-d",        "128:128:256:64:18:20",
                         "-a" };
  EXPECT_TRUE(op.ParseOptions(argc, const_cast<char **>(argv)));
  EXPECT_TRUE(op.adaptLambdaParams.adaptLambda);
}

TEST_F(Test_Options, DensityDataPassed)
{
  OptionsParser op;
  int argc = 8;
  const char *argv[] = { "./fredy_mri", "kdata.bin",  "traj.bin",
                         "output.bin",  "-d",         "128:128:256:64:18:20",
                         "-w",          "density.bin" };
  EXPECT_TRUE(op.ParseOptions(argc, const_cast<char **>(argv)));
  EXPECT_EQ("density.bin", op.densityFilename);
}

TEST_F(Test_Options, B1DataPassed)
{
  OptionsParser op;
  int argc = 8;
  const char *argv[] = { "./fredy_mri", "kdata.bin", "traj.bin",
                         "output.bin",  "-d",        "128:128:256:64:18:20",
                         "-s",          "b1.bin" };
  EXPECT_TRUE(op.ParseOptions(argc, const_cast<char **>(argv)));
  EXPECT_EQ("b1.bin", op.sensitivitiesFilename);
}

TEST_F(Test_Options, U0DataPassed)
{
  OptionsParser op;
  int argc = 8;
  const char *argv[] = { "./fredy_mri", "kdata.bin", "traj.bin",
                         "output.bin",  "-d",        "128:128:256:64:18:20",
                         "-u",          "u0.bin" };
  EXPECT_TRUE(op.ParseOptions(argc, const_cast<char **>(argv)));
  EXPECT_EQ("u0.bin", op.u0Filename);
}

