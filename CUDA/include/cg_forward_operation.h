#ifndef INCLUDE_CG_FORWARD_OPERATION_H_

#define INCLUDE_CG_FORWARD_OPERATION_H_

#include "agile/agile.hpp"
#include "agile/operator/cg.hpp"

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
  ForwardOperation(TCommunicator &communicator, TVector &muU, unsigned width,
                   unsigned height)
    : m_communicator(communicator), muU(muU), temp(muU.size()),
      divergence(muU.size()), width(width), height(height)
  {
    gradient.push_back(TVector(width * height));
    gradient.push_back(TVector(width * height));
  }

  /** \brief Copy constructor. */
  ForwardOperation(const ForwardOperation &other)
    : m_communicator(other.m_communicator, other.muU, other.width, other.height)
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

    agile::lowlevel::diff3(1, width, height, x.data(), gradient[0].data(), N,
                           true);
    agile::lowlevel::diff3(2, width, height, x.data(), gradient[1].data(), N,
                           true);

    agile::lowlevel::bdiff3(1, width, height, gradient[0].data(),
                            divergence.data(), N, true);

    agile::lowlevel::bdiff3(2, width, height, gradient[1].data(), temp.data(),
                            N, true);

    agile::addVector(temp, divergence, divergence);
    agile::scale((DType)-1.0, divergence, divergence);

    y.assign(y.size(), 0);
    agile::multiplyElementwise(muU, x, y);
    agile::addVector(y, divergence, y);
  }

  /** \brief Get the adjoint operator.
   *
   * This method returns a new operator that applies the adjoint matrix
   * to a vector.
   */
  adjoint_type getAdjoint() const
  {
    return adjoint_type(m_communicator);
  }

 private:
  /** \brief Reference to the communicator for parallel communication. */
  TCommunicator &m_communicator;

  TVector &muU;
  TVector temp;
  TVector divergence;
  unsigned width;
  unsigned height;
  std::vector<TVector> gradient;
};

#endif  // INCLUDE_CG_FORWARD_OPERATION_H_
