#include <gtest/gtest.h>

#include "./test_utils.h"
#include "../include/utils.h"

TEST(Test_Utils, Ellipke)
{
  EXPECT_NEAR(1.5708, utils::Ellipke(0), EPS);
  EXPECT_NEAR(1.3506, utils::Ellipke(0.5), EPS);
  EXPECT_NEAR(1.2111, utils::Ellipke(0.75), EPS);
  EXPECT_THROW(utils::Ellipke(1.5), std::invalid_argument);
}

TEST(Test_Utils, Median)
{
  std::vector<float> data;
  data.push_back(1.6);
  data.push_back(1.2);
  data.push_back(1.4);
  data.push_back(1.3);
  data.push_back(1.7);

  EXPECT_NEAR(1.4, utils::Median(data), EPS);
}

TEST(Test_Utils, MedianEvenDataSize)
{
  std::vector<float> data;
  data.push_back(1.6);
  data.push_back(1.2);
  data.push_back(1.4);
  data.push_back(1.3);
  data.push_back(1.7);
  data.push_back(1.8);

  EXPECT_NEAR(1.5, utils::Median(data), EPS);
}

TEST(Test_Utils, GetFileExtension)
{
  EXPECT_EQ(".h5", utils::GetFileExtension("test.h5"));
  EXPECT_EQ("", utils::GetFileExtension("test"));
  EXPECT_EQ(".dcm", utils::GetFileExtension("test.dcm"));
}

