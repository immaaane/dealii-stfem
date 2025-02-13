#include "operators.h"

namespace dealii
{
  template void
  tensorproduct_add(BlockVectorT<double> &,
                    FullMatrix<double> const &,
                    VectorT<double> const &,
                    int);
  template void
  tensorproduct_add(BlockVectorT<float> &,
                    FullMatrix<float> const &,
                    VectorT<float> const &,
                    int);
  template BlockVectorT<double>
  operator*(const FullMatrix<double> &, VectorT<double> const &);
  template BlockVectorT<float>
  operator*(const FullMatrix<float> &, VectorT<float> const &);

  template void
  tensorproduct_add(BlockVectorT<double> &,
                    FullMatrix<double> const &,
                    BlockVectorT<double> const &,
                    int);
  template void
  tensorproduct_add(BlockVectorT<float> &,
                    FullMatrix<float> const &,
                    BlockVectorT<float> const &,
                    int);

  template void
  tensorproduct_add(MutableBlockVectorSliceT<double> &,
                    FullMatrix<double> const &,
                    BlockVectorSliceT<double> const &,
                    int);
  template void
  tensorproduct_add(MutableBlockVectorSliceT<float> &,
                    FullMatrix<float> const &,
                    BlockVectorSliceT<float> const &,
                    int);

  template BlockVectorT<double>
  operator*(const FullMatrix<double> &, BlockVectorT<double> const &);
  template BlockVectorT<float>
  operator*(const FullMatrix<float> &, BlockVectorT<float> const &);

  namespace internal
  {
    template void
    scatter(BlockVectorT<double> &,
            const BlockVectorT<double> &,
            const FullMatrix<double> &,
            const BlockSlice &,
            unsigned int,
            unsigned int,
            unsigned int,
            unsigned int,
            unsigned int,
            unsigned int);
    template void
    scatter(BlockVectorT<float> &,
            const BlockVectorT<float> &,
            const FullMatrix<float> &,
            const BlockSlice &,
            unsigned int,
            unsigned int,
            unsigned int,
            unsigned int,
            unsigned int,
            unsigned int);

    template void
    scatter(BlockVectorT<double> &,
            const BlockVectorT<double> &,
            const FullMatrix<double> &,
            const BlockSlice &,
            unsigned int,
            unsigned int,
            unsigned int,
            unsigned int,
            unsigned int);
    template void
    scatter(BlockVectorT<float> &,
            const BlockVectorT<float> &,
            const FullMatrix<float> &,
            const BlockSlice &,
            unsigned int,
            unsigned int,
            unsigned int,
            unsigned int,
            unsigned int);
  } // namespace internal
} // namespace dealii
