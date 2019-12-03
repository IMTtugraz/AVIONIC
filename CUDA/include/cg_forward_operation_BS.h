#ifndef INCLUDE_CG_FORWARD_OPERATION_H_

#define INCLUDE_CG_FORWARD_OPERATION_H_

#include "agile/agile.hpp"
#include "agile/operator/cg.hpp"
#include "agile/gpu_vector_base.hpp"
#include "../include/cartesian_operator3d.h"


/** \brief Basic parameter struct used in primal-dual (PD) reconstructions. */
typedef struct H1Params
{
  /** \brief Maximum number of iterations for main iteration. */
  unsigned maxIt;

  /** \brief Step size for difference computation in x-spatial direction */
  RType dx;

  /** \brief Step size for difference computation in y-spatial direction */
  RType dy;

  /** \brief Step size for difference computation in z-spatial direction */
  RType dz;

  /** \residum relative Tolerance for CG. */
  float relTol;

  /** \esidum absolute Tolerance for CG. */
  float absTol;

  /** \brief Regularization parameter */
  RType mu;

} H1Params;

/**
 * \brief Forward operation used in CG for coil construction
 *
 * \see CoilConstruction::B1FromUH1
 */
template <typename TCommunicator, typename TVector, bool TIsAdjoint = false,
          bool TIsDistributed = false>
class ForwardOperation
    : public agile::ForwardOperatorExpression<
          ForwardOperation<TCommunicator, TVector, TIsAdjoint, TIsDistributed>
          /*formatter*/
          >
{
 public:
  /** \brief The adjoint type. */
  typedef ForwardOperation<TCommunicator, TVector, !TIsAdjoint, TIsDistributed>
      adjoint_type;

  /** \brief Constructor. */
  ForwardOperation(TCommunicator &communicator, CartesianOperator3D* mrop, H1Params &par, TVector &b1, unsigned width,
                   unsigned height, unsigned depth, unsigned coils)
    : m_communicator(communicator),  temp(width*height*depth),
      divergence(width*height*depth),b1(b1), kspace_temp(width*height*depth*coils), temp2(width*height*depth), width(width),
      height(height), depth(depth), coils(coils), mrop(mrop), par(par)
  {
    gradient.push_back(TVector(width * height * depth));
    gradient.push_back(TVector(width * height * depth));
    gradient.push_back(TVector(width * height * depth));
  }

  /** \brief Copy constructor. */
  ForwardOperation(const ForwardOperation &other)
    : m_communicator(other.m_communicator, other.mrop, other.par, other.width, other.height, other.depth)
  {
  }

  /** \brief Apply the operation.
  *
  * \param[in] x Accumulated vector.
  * \param[out] y K(x) (distributed vector).
  */
  template <typename TVectorType>
  void operator()(const TVectorType &x, TVectorType &y)
  {
    unsigned int N = x.size();
    temp.assign(temp.size(), 0);

    agile::lowlevel::diff3sym(1, width, height, x.data(), gradient[0].data(), N);
    agile::lowlevel::diff3sym(2, width, height, x.data(), gradient[1].data(), N);
    agile::lowlevel::diff3sym(3, width, height, x.data(), gradient[2].data(), N);
    agile::scale(par.dx, gradient[0], gradient[0]);
    agile::scale(par.dy, gradient[1], gradient[1]);
    agile::scale(par.dz, gradient[2], gradient[2]);


    agile::lowlevel::bdiff3sym_mbh(1, width, height, gradient[0].data(), divergence.data(), N);
    agile::scale(par.dx, divergence, divergence);

    agile::lowlevel::bdiff3sym_mbh(2, width, height, gradient[1].data(), temp.data(), N);
    agile::scale(par.dy, temp, temp);
    agile::addVector(temp, divergence, divergence);

    agile::lowlevel::bdiff3sym_mbh(3, width, height, gradient[2].data(), temp.data(),N);
    agile::scale(par.dz, temp, temp);
    agile::addVector(temp, divergence, divergence);
    agile::scale((DType)-1.0, divergence, divergence);

    kspace_temp.assign(N*coils, 0);
    temp.assign(temp.size(), 0);
    y.assign(x.size(), 0);
    agile::addVector(x,y,y);
    mrop->BackwardOperation(y,kspace_temp,b1, temp2);
    mrop->ForwardOperation(kspace_temp,temp,b1, temp2);
    agile::scale(par.mu, temp, temp);
    agile::addVector(temp, divergence, y);
  }

  /** \brief Get the adjoint operator.
   *
   * This method returns a new operator that applies the adjoint matrix
   * to a vector.
   */
  adjoint_type getAdjoint() const
  {
      std::cout<<__LINE__<<std::endl;
    return adjoint_type(m_communicator);
  }

 private:
  /** \brief Reference to the communicator for parallel communication. */
  TCommunicator &m_communicator;

  TVector temp;
  TVector divergence;
  TVector b1;
  TVector kspace_temp;
  TVector temp2;
  unsigned width;
  unsigned height;
  unsigned depth;
  unsigned coils;
  std::vector<TVector> gradient;
  CartesianOperator3D *mrop;
  H1Params par;
};

#endif  // INCLUDE_CG_FORWARD_OPERATION_H_
