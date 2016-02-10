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
  Dimension(unsigned width, unsigned height, unsigned depth,
	    unsigned readouts, unsigned encodings, unsigned encodings2,
	    unsigned coils, unsigned frames)
    : width(width), height(height), depth(depth), readouts(readouts), encodings(encodings), encodings2(encodings2),
      coils(coils), frames(frames)
  {
  }
  unsigned width;
  unsigned height;
  unsigned depth;
  unsigned readouts;
  unsigned encodings;
  unsigned encodings2;
  unsigned coils;
  unsigned frames;

} Dimension;

#endif  // INCLUDE_TYPES_H_
