#ifndef INCLUDE_TYPES_H_

#define INCLUDE_TYPES_H_

#include <complex>
#include "agile/gpu_vector.hpp"

/** \file Type defs used in ICTGV reconstruction. */

/** \brief Floating point type definition */
typedef float DType;

/** \brief Complex type definition */
typedef std::complex<DType> CType;
/** \brief Complex gpu vector type definition */
typedef agile::GPUVector<CType> CVector;

/** \brief Real type definition */
typedef DType RType;
/** \brief Real gpu vector type definition */
typedef agile::GPUVector<RType> RVector;

/**
 * \brief Dimension option struct
 *
 */
typedef struct Dimension
{
  Dimension()
  {
  }
  Dimension(unsigned width, unsigned height, unsigned readouts,
            unsigned encodings, unsigned coils, unsigned frames)
    : width(width), height(height), readouts(readouts), encodings(encodings),
      coils(coils), frames(frames)
  {
  }
  unsigned width;
  unsigned height;
  unsigned readouts;
  unsigned encodings;
  unsigned coils;
  unsigned frames;

} Dimension;

#endif  // INCLUDE_TYPES_H_
