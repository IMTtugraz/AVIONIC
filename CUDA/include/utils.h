#ifndef INCLUDE_UTILS_H_

#define INCLUDE_UTILS_H_

#include "./types.h"
#include <vector>

/**
 * \file
 * \brief Collection of util functions for Gradient computations, etc.
 *
 */
namespace utils
{
/** \brief Computation of ellipke function. Transcribed from MATLAB
 *implementation.
 *
 * References:
 * [1] M. Abramowitz and I.A. Stegun, "Handbook of Mathematical
 *     Functions" Dover Publications", 1965, 17.6.
 */
RType Ellipke(RType value);

/** \brief Compute 3-d gradient for given image data vector.
 *
 * \param[in] data_gpu 3-d data vector, dim width*height*frames
 * \param[in] width
 * \param[in] height
 * \param[in] dx Step size in x dim
 * \param[in] dy Step size in y dim
 * \param[in] dz Step size in z dim
 * \param[out] gradient components stored component-wise in vector as (dx,dy,dz)
 * */
void Gradient(CVector &data_gpu, std::vector<CVector> &gradient, unsigned width,
              unsigned height, DType dx = 1.0, DType dy = 1.0, DType dz = 1.0);

/** \brief Compute 3-d gradient for given image data vector.
 *
 * \param[in] data_gpu 3-d data vector, dim width*height*frames
 * \param[in] width
 * \param[in] height
 * \param[in] dx Step size in x dim
 * \param[in] dy Step size in y dim
 * \param[in] dz Step size in z dim
 * \return gradient components stored component-wise in vector as (dx,dy,dz)
 * */
std::vector<CVector> Gradient(CVector &data_gpu, unsigned width,
                              unsigned height, DType dx = 1.0, DType dy = 1.0,
                              DType dz = 1.0);

/** \brief Compute 2-d gradient for given image data vector.
 *
 * \param[in] data_gpu 2-d data vector, dim width*height*frames
 * \param[in] width
 * \param[in] height
 * \param[in] dx Step size in x dim
 * \param[in] dy Step size in y dim
 * \param[out] gradient components stored component-wise in vector as (dx,dy)
 * */
void Gradient2D(CVector &data_gpu, std::vector<CVector> &gradient,
                unsigned width, unsigned height, DType dx = 1.0,
                DType dy = 1.0);

/** \brief Compute 2-d gradient for given image data vector.
 *
 * \param[in] data_gpu 2-d data vector, dim width*height*frames
 * \param[in] width
 * \param[in] height
 * \param[in] dx Step size in x dim
 * \param[in] dy Step size in y dim
 * \return gradient components stored component-wise in vector as (dx,dy)
 * */
std::vector<CVector> Gradient2D(CVector &data_gpu, unsigned width,
                                unsigned height, DType dx = 1.0,
                                DType dy = 1.0);

/** \brief Computation of norm, i.e. sqrt(abs(dx).^2 + abs(dy).^2 + abs(dz).^2)
 * */
void GradientNorm(const std::vector<CVector> &gradient, CVector &norm);

/** \brief Computation of norm, i.e. sqrt(abs(dx).^2 + abs(dy).^2 + abs(dz).^2)
 * */
CVector GradientNorm(const std::vector<CVector> &gradient);

/** \brief Computation of norm, i.e. sqrt(abs(dx).^2 + abs(dy).^2)
 * */
void GradientNorm2D(const std::vector<CVector> &gradient, CVector &norm);

/** \brief Computation of norm, i.e. sqrt(abs(dx).^2 + abs(dy).^2)
 * */
CVector GradientNorm2D(const std::vector<CVector> &gradient);

/** \brief Compute 3-d second symmetric gradient for given image gradient
 *vector.
 *
 * \param[in] data_gpu vector of 3-d data gradient (dx,dy,dz)
 * \param[in] width
 * \param[in] height
 * \param[in] dx Step size in x dim
 * \param[in] dy Step size in y dim
 * \param[in] dz Step size in z dim
 * \param[out] gradient symmetric gradient components stored component-wise in
 *vector as
 *(dxx,dyy,dzz,dxy,dxz,dyz)
 * */
void SymmetricGradient(const std::vector<CVector> &data_gpu,
                       std::vector<CVector> &gradient, unsigned width,
                       unsigned height, DType dx = 1.0, DType dy = 1.0,
                       DType dz = 1.0);

/** \brief Compute 3-d second symmetric gradient for given image gradient
 *vector.
 *
 * \param[in] data_gpu vector of 3-d data gradient (dx,dy,dz)
 * \param[in] width
 * \param[in] height
 * \param[in] dx Step size in x dim
 * \param[in] dy Step size in y dim
 * \param[in] dz Step size in z dim
 * \param[out] gradient symmetric gradient components stored component-wise in
 *vector as
 *(dxx,dyy,dzz,dxy,dxz,dyz)
 * */
std::vector<CVector> SymmetricGradient(const std::vector<CVector> &data_gpu,
                                       unsigned width, unsigned height,
                                       DType dx = 1.0, DType dy = 1.0,
                                       DType dz = 1.0);

/** \brief Compute 2-d second symmetric gradient for given image gradient
 *vector.
 *
 * \param[in] data_gpu vector of 2-d data gradient (dx,dy)
 * \param[in] width
 * \param[in] height
 * \param[in] dx Step size in x dim
 * \param[in] dy Step size in y dim
 * \param[out] gradient symmetric gradient components stored component-wise in
 *vector as
 *(dxx,dyy,dxy)
 * */
void SymmetricGradient2D(const std::vector<CVector> &data_gpu,
                         std::vector<CVector> &gradient, unsigned width,
                         unsigned height, DType dx = 1.0, DType dy = 1.0);

/** \brief Compute 2-d second symmetric gradient for given image gradient
 *vector.
 *
 * \param[in] data_gpu vector of 2-d data gradient (dx,dy)
 * \param[in] width
 * \param[in] height
 * \param[in] dx Step size in x dim
 * \param[in] dy Step size in y dim
 * \return gradient symmetric gradient components stored component-wise in
 *vector as
 *(dxx,dyy,dxy)
 * */
std::vector<CVector> SymmetricGradient2D(const std::vector<CVector> &data_gpu,
                                         unsigned width, unsigned height,
                                         DType dx = 1.0, DType dy = 1.0);

/** \brief Computation of symmetric gradient norm, i.e. sqrt(abs(dx).^2 +
 abs(dy).^2 + abs(dz).^2 + 2.0*abs(dxy).^2 +
   2.0*abs(dxz).^2 + 2.0*abs(dyz).^2)
 * */
void SymmetricGradientNorm(const std::vector<CVector> &gradient, CVector &norm);

/** \brief Computation of symmetric gradient norm, i.e. sqrt(abs(dx).^2 +
 abs(dy).^2 + abs(dz).^2 + 2.0*abs(dxy).^2 +
   2.0*abs(dxz).^2 + 2.0*abs(dyz).^2)
 * */
CVector SymmetricGradientNorm(const std::vector<CVector> &gradient);

/** \brief Computation of symmetric gradient norm, i.e. sqrt(abs(dx).^2 +
 * abs(dy).^2 + 2.0*abs(dxy).^2)
 * */
void SymmetricGradientNorm2D(const std::vector<CVector> &gradient,
                             CVector &norm);

/** \brief Computation of symmetric gradient norm, i.e. sqrt(abs(dx).^2 +
 * abs(dy).^2 + 2.0*abs(dxy).^2)
 * */
CVector SymmetricGradientNorm2D(const std::vector<CVector> &gradient);

/** \brief Compute 3-d divergence with backward differences for given 3-d
 *component vector (i.e. gradient).
 *
 * Since, \f$grad^* = -div\f$
 * the divergence
 * \f$div = -\nabla \cdot p \f$
 * is computed as dual operator of the gradient (minus div).
 *
 * \param[in] gradient vector of 3-d gradient (dx,dy,dz)
 * \param[in] width
 * \param[in] height
 * \param[in] frames
 * \param[in] dx Step size in x dim
 * \param[in] dy Step size in y dim
 * \param[in] dz Step size in z dim
 * \param[out] divergence computed divergence
 * */
void Divergence(std::vector<CVector> &gradient, CVector &divergence,
                unsigned width, unsigned height, unsigned frames,
                DType dx = 1.0, DType dy = 1.0, DType dz = 1.0);

/** \brief Compute 3-d divergence with backward differences for given 3-d
 *component vector (i.e. gradient).
 *
 * Since, \f$grad^* = -div\f$
 * the divergence
 * \f$div = -\nabla \cdot p \f$
 * is computed as dual operator of the gradient (minus div).
 *
 * \param[in] gradient vector of 3-d gradient (dx,dy,dz)
 * \param[in] width
 * \param[in] height
 * \param[in] frames
 * \param[in] dx Step size in x dim
 * \param[in] dy Step size in y dim
 * \param[in] dz Step size in z dim
 * \return divergence computed divergence
 * */
CVector Divergence(std::vector<CVector> &gradient, unsigned width,
                   unsigned height, unsigned frames, DType dx = 1.0,
                   DType dy = 1.0, DType dz = 1.0);

/** \brief Compute 2-d divergence with backward differences for given 2-d
 *component vector (i.e. gradient).
 *
 * Since, \f$grad^* = -div\f$
 * the divergence
 * \f$div = -\nabla \cdot p \f$
 * is computed as dual operator of the gradient (minus div).
 *
 * \param[in] gradient vector of 2-d gradient (dx,dy)
 * \param[in] width
 * \param[in] height
 * \param[in] dx Step size in x dim
 * \param[in] dy Step size in y dim
 * \param[out] divergence computed divergence
 * */
void Divergence2D(std::vector<CVector> &gradient, CVector &divergence,
                  unsigned width, unsigned height, DType dx = 1.0,
                  DType dy = 1.0);

/** \brief Compute 2-d divergence with backward differences for given 2-d
 *component vector (i.e. gradient).
 *
 * Since, \f$grad^* = -div\f$
 * the divergence
 * \f$div = -\nabla \cdot p \f$
 * is computed as dual operator of the gradient (minus div).
 *
 * \param[in] gradient vector of 2-d gradient (dx,dy)
 * \param[in] width
 * \param[in] height
 * \param[in] dx Step size in x dim
 * \param[in] dy Step size in y dim
 * \return divergence computed divergence
 * */
CVector Divergence2D(std::vector<CVector> &gradient, unsigned width,
                     unsigned height, DType dx = 1.0, DType dy = 1.0);

/** \brief Compute 3-d symmetric divergence with backward differences for given
 *3-d symmetric component vector (i.e. symmetric gradient).
 *
 * Since, \f$grad^* = -div\f$
 * the divergence
 * \f$div = -\nabla \cdot p \f$
 * is computed as dual operator of the gradient (minus div).
 *
 * \param[in] gradient symmetric vector of 3-d gradient (dx,dy,dz, dxy, dxz,
 *dyz)
 * \param[in] width
 * \param[in] height
 * \param[in] frames
 * \param[in] dx Step size in x dim
 * \param[in] dy Step size in y dim
 * \param[in] dz Step size in z dim
 * \param[out] divergence computed symmetric divergence
 * */
void SymmetricDivergence(std::vector<CVector> &gradient,
                         std::vector<CVector> &divergence, unsigned width,
                         unsigned height, unsigned frames, DType dx = 1.0,
                         DType dy = 1.0, DType dz = 1.0);

/** \brief Compute 3-d symmetric divergence with backward differences for given
 *3-d symmetric component vector (i.e. symmetric gradient).
 *
 * Since, \f$grad^* = -div\f$
 * the divergence
 * \f$div = -\nabla \cdot p \f$
 * is computed as dual operator of the gradient (minus div).
 *
 * \param[in] gradient symmetric vector of 3-d gradient (dx,dy,dz, dxy, dxz,
 *dyz)
 * \param[in] width
 * \param[in] height
 * \param[in] frames
 * \param[in] dx Step size in x dim
 * \param[in] dy Step size in y dim
 * \param[in] dz Step size in z dim
 * \return divergence computed symmetric divergence
 * */
std::vector<CVector> SymmetricDivergence(std::vector<CVector> &gradient,
                                         unsigned width, unsigned height,
                                         unsigned frames, DType dx = 1.0,
                                         DType dy = 1.0, DType dz = 1.0);

/** \brief Compute 2-d symmetric divergence with backward differences for given
 *2-d symmetric component vector (i.e. symmetric gradient).
 *
 * Since, \f$grad^* = -div\f$
 * the divergence
 * \f$div = -\nabla \cdot p \f$
 * is computed as dual operator of the gradient (minus div).
 *
 * \param[in] gradient symmetric vector of 3-d gradient (dx,dy, dxy)
 * \param[in] width
 * \param[in] height
 * \param[in] dx Step size in x dim
 * \param[in] dy Step size in y dim
 * \param[out] divergence computed symmetric divergence
 * */
void SymmetricDivergence2D(std::vector<CVector> &gradient,
                           std::vector<CVector> &divergence, unsigned width,
                           unsigned height, DType dx = 1.0, DType dy = 1.0);

/** \brief Compute 2-d symmetric divergence with backward differences for given
 *2-d symmetric component vector (i.e. symmetric gradient).
 *
 * Since, \f$grad^* = -div\f$
 * the divergence
 * \f$div = -\nabla \cdot p \f$
 * is computed as dual operator of the gradient (minus div).
 *
 * \param[in] gradient symmetric vector of 3-d gradient (dx,dy, dxy)
 * \param[in] width
 * \param[in] height
 * \param[in] dx Step size in x dim
 * \param[in] dy Step size in y dim
 * \return divergence computed symmetric divergence
 * */
std::vector<CVector> SymmetricDivergence2D(std::vector<CVector> &gradient,
                                           unsigned width, unsigned height,
                                           DType dx = 1.0, DType dy = 1.0);

/**
 * \brief Computation of TV norm
 *
 * \f$ TV(x) = \sum |\nabla x| \f$
 *
 * \param[in] data_gpu data vector
 * \param[in] width
 * \param[in] height
 * \param[in] frames
 * \param[in] dx Step size in x dim
 * \param[in] dy Step size in y dim
 * \param[in] dt Step size in z dim (time)
 * \return TV norm
 */
RType TVNorm(CVector &data_gpu, unsigned width, unsigned height, DType dx = 1.0,
             DType dy = 1.0, DType dt = 1.0);

/**
 * \brief Computation of TGV2 norm
 *
 * \f$ TGV2_{\alpha_0,\alpha_1}(x,v) = \alpha_1 \sum |\nabla x - v| + \alpha_0
 *\sum
 *|\frac{1}{2} (\nabla v + \nabla v^T) \f$
 *
 * \param[in] data1_gpu data vector, gradient (x)
 * \param[in] data2_gpu data vector, symmetric gradient (v)
 * \param[in] alpha0 norm parameter
 * \param[in] alpha1 norm parameter
 * \param[in] width
 * \param[in] height
 * \param[in] dx Step size in x dim
 * \param[in] dy Step size in y dim
 * \param[in] dz Step size in z dim (time)
 * \return TGV2 norm
 */
RType TGV2Norm(CVector &data1_gpu, std::vector<CVector> &data2_gpu,
               RType alpha0, RType alpha1, unsigned width, unsigned height,
               DType dx = 1.0, DType dy = 1.0, DType dz = 1.0);

RType TGV2Norm(CVector &data1_gpu, std::vector<CVector> &data2_gpu,
               std::vector<CVector> &temp3, std::vector<CVector> &temp6,
               RType alpha0, RType alpha1, unsigned width, unsigned height,
               DType dx = 1.0, DType dy = 1.0, DType dz = 1.0);

/**
 * \brief Computation of ICTGV2 norm
 *
 * \f$ ICTGV2_{\alpha_0,\alpha_1,\alpha}(u,v,w,x) =
 *\frac{\alpha}{min(alpha,1-alpha}
 *TGV2_{\alpha_0,\alpha_1}(u,v) + \frac{1-\alpha}{min(alpha,1-alpha}
 *TGV2_{\alpha_0,\alpha_1}(w,x)\f$
 *
 * \param[in] data1_gpu data vector, gradient (u)
 * \param[in] data2_gpu data vector, symmetric gradient (v)
 * \param[in] data3_gpu data vector, gradient (w)
 * \param[in] data4_gpu data vector, symmetric gradient (x)
 * \param[in] alpha0 norm parameter
 * \param[in] alpha1 norm parameter
 * \param[in] alpha  norm parameter
 * \param[in] width
 * \param[in] height
 * \param[in] ds Step size in spatial dim
 * \param[in] dt Step size in temporal dim (time)
 * \param[in] ds2 Step size 2 in spatial dim
 * \param[in] dt2 Step size 2 in temporal dim (time)
 * \return ICTGV2 norm
 */
RType ICTGV2Norm(CVector &data1_gpu, std::vector<CVector> &data2_gpu,
                 CVector &data3_gpu, std::vector<CVector> &data4_gpu,
                 RType alpha0, RType alpha1, RType alpha, unsigned width,
                 unsigned height, DType ds = 1.0, DType dt = 1.0,
                 DType ds2 = 1.0, DType dt2 = 1.0);

RType ICTGV2Norm(CVector &data1_gpu, std::vector<CVector> &data2_gpu,
                 CVector &data3_gpu, std::vector<CVector> &data4_gpu,
                 std::vector<CVector> &temp3, std::vector<CVector> &temp6,
                 RType alpha0, RType alpha1, RType alpha, unsigned width,
                 unsigned height, DType ds = 1.0, DType dt = 1.0,
                 DType ds2 = 1.0, DType dt2 = 1.0);

/**
 * \brief Function to perform element-wise division for each vector in the
 * collection of vectors y by the vector v.
 *
 * In order avoid division by zero, v is checked by
 * \f$v = max(v, 1.0)\f$
 * i.e.
 * \f$ y[i] \leftarrow \frac{y[i]}{max(v,1.0)} element-wise, for
 *i=1..vecElements \f$
 *
 * \param[in] y collection of vectors
 * \param[in] v vector of divisor values
 * \param[out] vecElements amount of vector elements in collection y
 */
void DivideVectorElementwise(std::vector<CVector> &y, CVector v,
                             unsigned vecElements);

/**
 * \brief Function to perform element-wise division for each vector in the
 * collection of vectors y by the vector v, which is scaled by param scale.
 *
 * In order avoid division by zero, v is checked by
 * \f$v = max(scale \cdot v, 1.0)\f$
 * i.e.
 * \f$ y[i] \leftarrow \frac{y[i]}{max(v,1.0)} element-wise, for
 *i=1..vecElements \f$
 *
 * \param[in] y collection of vectors
 * \param[in] v vector of divisor values
 * \param[in] scale scale factor
 * \param[out] vecElements amount of vector elements in collection y
 */
void DivideVectorScaledElementwise(std::vector<CVector> &y, CVector &v,
                                   RType scale, unsigned vecElements);

/**
 * \brief Perform proximal mapping for each 2-d gradient vector component y by
 *scaled gradient norm
 *
 * \f$ y[i] \leftarrow \frac{y[i]}{max(scale \cdot norm(y),1.0)} element-wise
 *\f$
 *
 * \param[in,out] y gradient vecotr
 * \param[in] scale factor
 */
void ProximalMap2D(std::vector<CVector> &y, RType scale);

/**
 * \brief Perform proximal mapping for each 2-d symmetric gradient vector
 *component
 *y by scaled gradient norm
 *
 * \f$ y[i] \leftarrow \frac{y[i]}{max(scale \cdot norm(y),1.0)} element-wise
 *\f$
 *
 * \param[in,out] y gradient vecotr
 * \param[in] scale factor
 */
void ProximalMap2DSym(std::vector<CVector> &y, RType scale);

/**
 * \brief Perform proximal mapping for each 3-d gradient vector component y by
 *scaled gradient norm
 *
 * \f$ y[i] \leftarrow \frac{y[i]}{max(scale \cdot norm(y),1.0)} element-wise
 *\f$
 *
 * \param[in,out] y gradient vecotr
 * \param[in] scale factor
 */
void ProximalMap3(std::vector<CVector> &y, RType scale);

/**
 * \brief Perform proximal mapping for each 3-d gradient vector component y by
 *scaled gradient norm
 *
 * \f$ y[i] \leftarrow \frac{y[i]}{max(scale \cdot norm(y),1.0)} element-wise
 *\f$
 *
 * \param[in,out] y gradient vecotr
 * \param[in] scale factor
 */
void ProximalMap6(std::vector<CVector> &y, RType scale);

/**
 * \brief Compute sum of squares for three-component vector x
 */
void SumOfSquares3(std::vector<CVector> &x, CVector &sum);
/**
 * \brief Compute sum of squares for six-component vector x
 */
void SumOfSquares6(std::vector<CVector> &x, CVector &sum);

/**
 * \brief Extract sub vector out of full vector
 *
 * \param[in] full Full vector
 * \param[out] stride Extracted stride
 * \param[in] index index of stride to extract
 * \param[in] strideLength length of one stride
 */
template <typename TType>
void GetSubVector(TType &full, TType &stride, unsigned index,
                  unsigned strideLength)
{
  agile::lowlevel::get_content(full.data(), 1, strideLength, 0,
                               index * strideLength, stride.data(), 1,
                               strideLength);
}

/**
 * \brief Set sub vector elements of full vector
 *
 * \param[in] stride Stride to insert/update into full vector
 * \param[in,out] full Full vector
 * \param[in] index index of stride to extract
 * \param[in] strideLength length of one stride
 */
template <typename TType>
void SetSubVector(TType &stride, TType &full, unsigned index,
                  unsigned strideLength)
{
  agile::lowlevel::get_content(stride.data(), 1, strideLength, 0, 0,
                               full.data() + index * strideLength, 1,
                               strideLength);
}

/**
 * \brief Computation of median element inside data vector.
 *
 * \param[in] data Data vector
 * \return Median element value
 */
RType Median(std::vector<RType> data);

/**
 * \brief Resolve path of parent directory
 *
 * \param[in] filename File name to analyze
 * \return Absolute path of passed file name
 */
std::string GetParentDirectory(const std::string &filename);


/**
 * \brief Resolve filename of file with extension and path
 *
 * \param[in] filename File name to analyze
 * \return Absolute filename without extension
 */
std::string GetFilename(const std::string &filename);

/**
 * \brief Resolve extension of file
 *
 * \param[in] filename File to analyze
 * \return Extension passed file name
 */
std::string GetFileExtension(const std::string &filename);

/**
 * \brief Write vector to h5 file into dataset `dataset/<fieldname>`
 *
 * \param[in] filename h5 output file name
 * \param[in] fieldname dataset variable name
 * \param[in] dims vector containing data dimension description
 * \param[in] data data vector
 */
void WriteH5File(const std::string &filename, const std::string &fieldname,
                 std::vector<size_t> dims, std::vector<CType> data);

/**
 * \brief Extract sub vector out of full vector
 *
 * \param[in] full Full vector
 * \param[out] stride Extracted stride
 * \param[in] index index of stride to extract
 * \param[in] strideLength length of one stride
 */
void GetSubVector(CVector &full, CVector &stride, unsigned index,
                  unsigned strideLength);

/**
 * \brief Set sub vector elements of full vector
 *
 * \param[in] stride Stride to insert/update into full vector
 * \param[in,out] full Full vector
 * \param[in] index index of stride to extract
 * \param[in] strideLength length of one stride
 */
void SetSubVector(CVector &stride, CVector &full, unsigned index,
                  unsigned strideLength);


/**
 * \brief Get CFL file header information
*/
bool ReadCflHeader(const std::string &filename, long * dimensions); //Dimension &dim);

//bool readCflFile(const char* filename, unsigned size, std::vector<CVector> & data);
 
}

#endif  // INCLUDE_UTILS_H_
