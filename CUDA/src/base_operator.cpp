#include "../include/base_operator.h"

BaseOperator::BaseOperator(unsigned width, unsigned height, unsigned coils,
                           unsigned frames)
  : width(width), height(height), coils(coils), frames(frames)
{
}

BaseOperator::~BaseOperator()
{
}

