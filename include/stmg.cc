#include "stmg.h"

namespace dealii
{
  template class MGTwoLevelBlockTransfer<2, double>;
  template class MGTwoLevelBlockTransfer<3, double>;
  template class MGTwoLevelBlockTransfer<2, float>;
  template class MGTwoLevelBlockTransfer<3, float>;

  template class STMGTransferBlockMatrixFree<2, double>;
  template class STMGTransferBlockMatrixFree<3, double>;
  template class STMGTransferBlockMatrixFree<2, float>;
  template class STMGTransferBlockMatrixFree<3, float>;

  template auto
  build_stmg_transfers(
    TimeStepType const,
    const MGLevelObject<std::shared_ptr<const DoFHandler<2>>> &,
    const MGLevelObject<std::shared_ptr<const AffineConstraints<double>>> &,
    const std::function<
      void(const unsigned int, VectorT<double> &, const unsigned int)> &,
    const bool,
    const std::vector<MGType> &,
    const std::vector<BlockSlice> &);
  template auto
  build_stmg_transfers(
    TimeStepType const,
    const MGLevelObject<std::shared_ptr<const DoFHandler<3>>> &,
    const MGLevelObject<std::shared_ptr<const AffineConstraints<double>>> &,
    const std::function<
      void(const unsigned int, VectorT<double> &, const unsigned int)> &,
    const bool,
    const std::vector<MGType> &,
    const std::vector<BlockSlice> &);
  template auto
  build_stmg_transfers(
    TimeStepType const,
    const MGLevelObject<std::shared_ptr<const DoFHandler<2>>> &,
    const MGLevelObject<std::shared_ptr<const AffineConstraints<float>>> &,
    const std::function<
      void(const unsigned int, VectorT<float> &, const unsigned int)> &,
    const bool,
    const std::vector<MGType> &,
    const std::vector<BlockSlice> &);
  template auto
  build_stmg_transfers(
    TimeStepType const,
    const MGLevelObject<std::shared_ptr<const DoFHandler<3>>> &,
    const MGLevelObject<std::shared_ptr<const AffineConstraints<float>>> &,
    const std::function<
      void(const unsigned int, VectorT<float> &, const unsigned int)> &,
    const bool,
    const std::vector<MGType> &,
    const std::vector<BlockSlice> &);


  template auto
  build_stmg_transfers(
    TimeStepType const,
    const MGLevelObject<std::vector<const DoFHandler<2> *>> &,
    const MGLevelObject<std::vector<const dealii::AffineConstraints<float> *>>
      &,
    const std::function<
      void(const unsigned int, VectorT<float> &, const unsigned int)> &,
    const bool,
    const std::vector<MGType> &,
    const std::vector<BlockSlice> &);
  template auto
  build_stmg_transfers(
    TimeStepType const,
    const MGLevelObject<std::vector<const DoFHandler<2> *>> &,
    const MGLevelObject<std::vector<const dealii::AffineConstraints<double> *>>
      &,
    const std::function<
      void(const unsigned int, VectorT<double> &, const unsigned int)> &,
    const bool,
    const std::vector<MGType> &,
    const std::vector<BlockSlice> &);
  template auto
  build_stmg_transfers(
    TimeStepType const,
    const MGLevelObject<std::vector<const DoFHandler<3> *>> &,
    const MGLevelObject<std::vector<const dealii::AffineConstraints<float> *>>
      &,
    const std::function<
      void(const unsigned int, VectorT<float> &, const unsigned int)> &,
    const bool,
    const std::vector<MGType> &,
    const std::vector<BlockSlice> &);
  template auto
  build_stmg_transfers(
    TimeStepType const,
    const MGLevelObject<std::vector<const DoFHandler<3> *>> &,
    const MGLevelObject<std::vector<const dealii::AffineConstraints<double> *>>
      &,
    const std::function<
      void(const unsigned int, VectorT<double> &, const unsigned int)> &,
    const bool,
    const std::vector<MGType> &,
    const std::vector<BlockSlice> &);

  template class PreconditionVanka<double>;
  template class PreconditionVanka<float>;
} // namespace dealii
