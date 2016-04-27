#include "../include/base_operator.h"

BaseOperator::BaseOperator(unsigned width, unsigned height, unsigned depth, unsigned coils,
                           unsigned frames)
  : width(width), height(height), depth(depth), coils(coils), frames(frames)
{
}

BaseOperator::~BaseOperator()
{
}

