// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by Nils Margenberg and Peter Munch

#pragma once
#include <deal.II/base/parameter_handler.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix_tools.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.templates.h>
#include <deal.II/multigrid/multigrid.h>

#include <variant>

#include "compute_block_matrix.h"
#include "fe_time.h"
#include "stokes.h"
#include "types.h"

enum class TransferType : bool
{
  Time  = true,
  Space = false,
};

namespace dealii
{
  template <int dim, typename Number>
  class MGTwoLevelBlockTransfer
  {
    using VectorType      = VectorT<Number>;
    using BlockVectorType = BlockVectorT<Number>;
    BlockSlice blk_index; // Only one: In space block structure doesn't change
    std::vector<SmartPointer<const MGTwoLevelTransfer<dim, VectorType>>>
      transfer;

  public:
    MGTwoLevelBlockTransfer() = default;
    MGTwoLevelBlockTransfer(
      BlockSlice const                          &blk_index_,
      MGTwoLevelTransfer<dim, VectorType> const &transfer_)
      : blk_index(blk_index_)
      , transfer(1,
                 &const_cast<MGTwoLevelTransfer<dim, VectorType> &>(
                   static_cast<const MGTwoLevelTransfer<dim, VectorType> &>(
                     Utilities::get_underlying_value(transfer_))))

    {}
    MGTwoLevelBlockTransfer(
      BlockSlice const &blk_index_,
      std::vector<std::shared_ptr<MGTwoLevelTransfer<dim, VectorType>>> const
        &transfer_)
      : blk_index(blk_index_)
    {
      AssertDimension(transfer_.size(), blk_index_.n_variables());
      transfer.reserve(transfer_.size());
      for (unsigned int v = 0; v < transfer_.size(); ++v)
        transfer.emplace_back(
          &const_cast<MGTwoLevelTransfer<dim, VectorType> &>(
            static_cast<const MGTwoLevelTransfer<dim, VectorType> &>(
              Utilities::get_underlying_value(transfer_[v]))));
    }

    MGTwoLevelBlockTransfer(MGTwoLevelBlockTransfer const &other)
      : blk_index(other.blk_index)
      , transfer(other.transfer)
    {}
    MGTwoLevelBlockTransfer &
    operator=(MGTwoLevelBlockTransfer const &) = default;

    void
    prolongate_and_add(BlockVectorType &dst, const BlockVectorType &src) const
    {
      for (unsigned int i = 0; i < src.n_blocks(); ++i)
        {
          auto const &[tsp, v, td] = blk_index.decompose(i);
          transfer[v]->prolongate_and_add(dst.block(i), src.block(i));
        }
    }

    void
    restrict_and_add(BlockVectorType &dst, const BlockVectorType &src) const
    {
      for (unsigned int i = 0; i < src.n_blocks(); ++i)
        {
          auto const &[tsp, v, td] = blk_index.decompose(i);
          transfer[v]->restrict_and_add(dst.block(i), src.block(i));
        }
    }
  };
  template <int dim, typename Number>
  using MGTwoLevelTransferSpace = MGTwoLevelBlockTransfer<dim, Number>;

  template <typename Number>
  class MGTwoLevelTransferTime
  {
    using BlockVectorType = BlockVectorT<Number>;
    BlockSlice         blk_index_hi;
    BlockSlice         blk_index_lo;
    FullMatrix<Number> prolongation_matrix;
    FullMatrix<Number> restriction_matrix;

    void
    transfer_and_add(BlockVectorType          &dst,
                     BlockSlice const         &blk_dst,
                     FullMatrix<Number> const &matrix,
                     BlockVectorType const    &src,
                     BlockSlice const         &blk_src) const
    {
      AssertDimension(matrix.n() * blk_src.n_variables(), src.n_blocks());
      AssertDimension(matrix.m() * blk_src.n_variables(), dst.n_blocks());
      AssertDimension(blk_src.n_blocks(), src.n_blocks());
      AssertDimension(blk_dst.n_blocks(), dst.n_blocks());
      for (unsigned int v = 0; v < blk_src.n_variables(); ++v)
        {
          BlockVectorSliceT<Number>        src_v = blk_src.get_time(src, v);
          MutableBlockVectorSliceT<Number> dst_v = blk_dst.get_time(dst, v);
          tensorproduct_add(dst_v, matrix, src_v);
        }
    }

  public:
    MGTwoLevelTransferTime() = default;
    MGTwoLevelTransferTime(BlockSlice const  &blk_index_hi_,
                           BlockSlice const  &blk_index_lo_,
                           TimeStepType const type,
                           const bool         restrict_is_transpose_prolongate,
                           MGType             mg_type)
      : blk_index_hi(blk_index_hi_)
      , blk_index_lo(blk_index_lo_)
    {
      Assert((blk_index_hi.n_timedofs() == blk_index_lo.n_timedofs()) !=
               (blk_index_hi.n_timesteps_at_once() ==
                blk_index_lo.n_timesteps_at_once()),
             ExcInternalError());
      bool k_mg   = mg_type == MGType::k;
      bool tau_mg = mg_type == MGType::tau;

      unsigned int r                   = (type == TimeStepType::DG) ?
                                           blk_index_hi.n_timedofs() - 1 :
                                           blk_index_hi.n_timedofs();
      unsigned int r_lo                = (type == TimeStepType::DG) ?
                                           blk_index_lo.n_timedofs() - 1 :
                                           blk_index_lo.n_timedofs();
      unsigned int n_timesteps_at_once = blk_index_hi.n_timesteps_at_once();

      prolongation_matrix =
        k_mg ?
          get_time_projection_matrix<Number>(type,
                                             r_lo,
                                             r,
                                             n_timesteps_at_once) :
        tau_mg ?
          get_time_prolongation_matrix<Number>(type, r, n_timesteps_at_once) :
          dealii::IdentityMatrix(blk_index_hi.n_timedofs() *
                                 blk_index_hi.n_timesteps_at_once());
      if (restrict_is_transpose_prolongate)
        {
          restriction_matrix.reinit(prolongation_matrix.n(),
                                    prolongation_matrix.m());
          restriction_matrix.copy_transposed(prolongation_matrix);
        }
      else
        restriction_matrix =
          k_mg ?
            get_time_projection_matrix<Number>(type,
                                               r,
                                               r_lo,
                                               n_timesteps_at_once) :
          tau_mg ?
            get_time_restriction_matrix<Number>(type, r, n_timesteps_at_once) :
            dealii::IdentityMatrix(blk_index_hi.n_timedofs() *
                                   blk_index_hi.n_timesteps_at_once());
    }

    MGTwoLevelTransferTime(MGTwoLevelTransferTime<Number> const &other)
      : blk_index_hi(other.blk_index_hi)
      , blk_index_lo(other.blk_index_lo)
      , prolongation_matrix(other.prolongation_matrix)
      , restriction_matrix(other.restriction_matrix)

    {}
    MGTwoLevelTransferTime &
    operator=(MGTwoLevelTransferTime const &) = default;

    void
    prolongate_and_add(BlockVectorType &dst, const BlockVectorType &src) const
    {
      transfer_and_add(
        dst, blk_index_hi, prolongation_matrix, src, blk_index_lo);
    }

    void
    restrict_and_add(BlockVectorType &dst, const BlockVectorType &src) const
    {
      transfer_and_add(
        dst, blk_index_lo, restriction_matrix, src, blk_index_hi);
    }
  };

  template <int dim, typename Number>
  class TwoLevelTransferOperator
  {
    using VectorType      = VectorT<Number>;
    using BlockVectorType = BlockVectorT<Number>;
    std::variant<MGTwoLevelTransferSpace<dim, Number>,
                 MGTwoLevelTransferTime<Number>>
      transfer_variant;

  public:
    TwoLevelTransferOperator(
      BlockSlice const &blk_index,
      std::vector<std::shared_ptr<MGTwoLevelTransfer<dim, VectorType>>> const
        &transfer)
      : transfer_variant(
          MGTwoLevelTransferSpace<dim, Number>(blk_index, transfer))
    {}
    TwoLevelTransferOperator(
      MGTwoLevelTransfer<dim, VectorType> const &transfer)
      : transfer_variant(MGTwoLevelTransferSpace<dim, Number>(transfer))
    {}
    TwoLevelTransferOperator(MGTwoLevelTransferTime<Number> const &transfer)
      : transfer_variant(transfer)
    {}
    TwoLevelTransferOperator(TwoLevelTransferOperator<dim, Number> const &other)
      : transfer_variant(other.transfer_variant)
    {}
    TwoLevelTransferOperator &
    operator=(const TwoLevelTransferOperator &) = default;
    TwoLevelTransferOperator()                  = default;

    void
    prolongate_and_add(BlockVectorType &dst, const BlockVectorType &src) const
    {
      std::visit([&dst, &src](
                   auto &transfer) { transfer.prolongate_and_add(dst, src); },
                 transfer_variant);
    }

    void
    restrict_and_add(BlockVectorType &dst, const BlockVectorType &src) const
    {
      std::visit([&dst, &src](
                   auto &transfer) { transfer.restrict_and_add(dst, src); },
                 transfer_variant);
    }
  };

  template <int dim, typename Number>
  class STMGTransferBlockMatrixFree final
    : public MGTransferBase<BlockVectorT<Number>>
  {
    using VectorType      = VectorT<Number>;
    using BlockVectorType = BlockVectorT<Number>;

  public:
    static const bool supports_dof_handler_vector = true;

    STMGTransferBlockMatrixFree(
      std::vector<
        std::vector<std::shared_ptr<MGTwoLevelTransfer<dim, VectorType>>>> const
        &space_transfers_,
      MGLevelObject<TwoLevelTransferOperator<dim, Number>> const &transfers_,
      std::vector<BlockSlice>                                     blk_indices_,
      std::vector<
        std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>>>
        partitioners_)
      : space_transfers(space_transfers_)
      , transfer(transfers_)
      , blk_indices(blk_indices_)
      , same_for_all(blk_indices.back().n_variables() == 1)
      , partitioners(partitioners_)
    {}

    virtual ~STMGTransferBlockMatrixFree() override = default;

    void
    prolongate(const unsigned int     to_level,
               BlockVectorType       &dst,
               const BlockVectorType &src) const override final
    {
      dst = Number(0.0);
      prolongate_and_add(to_level, dst, src);
    }

    void
    prolongate_and_add(const unsigned int     to_level,
                       BlockVectorType       &dst,
                       const BlockVectorType &src) const override final
    {
      transfer[to_level].prolongate_and_add(dst, src);
    }

    void
    restrict_and_add(const unsigned int     from_level,
                     BlockVectorType       &dst,
                     const BlockVectorType &src) const override final
    {
      transfer[from_level].restrict_and_add(dst, src);
    }

    template <typename BlockVectorType2>
    void
    copy_from_mg(const std::vector<const DoFHandler<dim> *> &,
                 BlockVectorType2                     &dst,
                 const MGLevelObject<BlockVectorType> &src) const
    {
      if (dst.n_blocks() != blk_indices.back().n_blocks())
        dst.reinit(blk_indices.back().n_blocks());
      dst.zero_out_ghost_values();
      dst.copy_locally_owned_data_from(src[src.max_level()]);
    }

    template <typename BlockVectorType2>
    void
    copy_from_mg(const DoFHandler<dim> &,
                 BlockVectorType2                     &dst,
                 const MGLevelObject<BlockVectorType> &src) const
    {
      const std::vector<const DoFHandler<dim> *> mg_dofs;
      copy_from_mg(mg_dofs, dst, src);
    }

    template <typename BlockVectorType2>
    void
    copy_to_mg(const DoFHandler<dim>          &dof_handler,
               MGLevelObject<BlockVectorType> &dst,
               const BlockVectorType2         &src) const
    {
      Assert(same_for_all,
             ExcMessage(
               "This object was initialized with support for usage with one "
               "DoFHandler for each block, but this method assumes that "
               "the same DoFHandler is used for all the blocks!"));
      const std::vector<const DoFHandler<dim> *> mg_dofs(src.n_blocks(),
                                                         &dof_handler);
      copy_to_mg(mg_dofs, dst, src);
    }

    template <typename BlockVectorType2>
    void
    copy_to_mg(const std::vector<const DoFHandler<dim> *> &,
               MGLevelObject<BlockVectorType> &dst,
               const BlockVectorType2         &src) const
    {
      AssertDimension(blk_indices.back().n_blocks(), src.n_blocks());
      for (unsigned int level = dst.min_level(); level <= dst.max_level();
           ++level)
        {
          if (dst[level].n_blocks() != blk_indices[level].n_blocks())
            dst[level].reinit(blk_indices[level].n_blocks());
          initialize_dof_vector(level, dst[level], level == dst.max_level());
        }
      dst[dst.max_level()].copy_locally_owned_data_from(src);
    }

  private:
    void
    initialize_dof_vector(const unsigned int level,
                          BlockVectorType   &vec,
                          const bool         omit_zeroing_entries) const
    {
      for (unsigned int i = 0; i < vec.n_blocks(); ++i)
        {
          auto const &[tsp, v, td] =
            blk_indices[level - transfer.min_level()].decompose(i);
          auto const &partitioner =
            partitioners[level - transfer.min_level()][v];
          Assert(partitioners.empty() || (transfer.min_level() <= level &&
                                          level <= transfer.max_level()),
                 ExcInternalError());

          if (vec.block(i).get_partitioner().get() == partitioner.get() ||
              (vec.block(i).size() == partitioner->size() &&
               vec.block(i).locally_owned_size() ==
                 partitioner->locally_owned_size()))
            {
              if (!omit_zeroing_entries)
                vec.block(i) = 0;
            }
          else
            vec.block(i).reinit(partitioner, omit_zeroing_entries);
        }
    }

    std::vector<
      std::vector<std::shared_ptr<MGTwoLevelTransfer<dim, VectorType>>>>
                                                         space_transfers;
    MGLevelObject<TwoLevelTransferOperator<dim, Number>> transfer;

    std::vector<BlockSlice> blk_indices;
    bool                    same_for_all;
    std::vector<std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>>>
      partitioners;
  };

  inline auto
  get_blk_indices(TimeStepType const               type,
                  unsigned int                     n_timesteps_at_once,
                  unsigned int                     n_variables,
                  unsigned int                     n_levels,
                  const std::vector<MGType>       &mg_type_level,
                  std::vector<unsigned int> const &poly_time_sequence)
  {
    AssertDimension(n_levels - 1, mg_type_level.size());
    std::vector<BlockSlice> blk_indices(n_levels);
    auto                    p_mg = poly_time_sequence.rbegin();
#ifdef DEBUG
    unsigned int n_k_levels = 0, n_tau_levels = 0;
    for (auto const &el : mg_type_level)
      if (el == MGType::k)
        ++n_k_levels;
      else if (el == MGType::tau)
        ++n_tau_levels;

    Assert((type == TimeStepType::DG ? *p_mg + 1 : *p_mg >= n_k_levels),
           ExcLowerRange(*p_mg, n_k_levels));
#endif

    unsigned int i = n_levels - 1;
    for (auto mgt = mg_type_level.rbegin(); mgt != mg_type_level.rend();
         ++mgt, --i)
      {
        unsigned int n_dofs_intvl =
          (type == TimeStepType::DG) ? *p_mg + 1 : *p_mg;
        blk_indices[i] =
          BlockSlice(n_timesteps_at_once, n_variables, n_dofs_intvl);
        if (*mgt == MGType::k)
          ++p_mg;
        else if (*mgt == MGType::tau)
          n_timesteps_at_once /= 2;
      }
    Assert(p_mg == poly_time_sequence.rend() - 1, ExcInternalError());
    blk_indices[0] = BlockSlice(n_timesteps_at_once,
                                n_variables,
                                (type == TimeStepType::DG) ? *p_mg + 1 : *p_mg);
    return blk_indices;
  }

  template <int dim, typename Number>
  auto
  build_stmg_transfers(
    TimeStepType const type,
    const MGLevelObject<std::shared_ptr<const DoFHandler<dim>>>
      &mg_dof_handlers,
    const MGLevelObject<std::shared_ptr<const AffineConstraints<Number>>>
                                                  &mg_constraints,
    const std::function<void(const unsigned int,
                             VectorT<Number> &,
                             const unsigned int)> &initialize_dof_vector,
    const bool                     restrict_is_transpose_prolongate,
    const std::vector<MGType>     &mg_type_level,
    const std::vector<BlockSlice> &blk_indices)
  {
    unsigned int const min_level = mg_dof_handlers.min_level();
    unsigned int const max_level = mg_dof_handlers.max_level();
    MGLevelObject<std::vector<const DoFHandler<dim> *>> mg_dof_handlers_(
      min_level, max_level);
    MGLevelObject<std::vector<const dealii::AffineConstraints<Number> *>>
      mg_constraints_(min_level, max_level);
    for (unsigned int l = min_level; l <= max_level; ++l)
      {
        mg_dof_handlers_[l].push_back(mg_dof_handlers[l].get());
        mg_constraints_[l].push_back(mg_constraints[l].get());
      }
    return build_stmg_transfers(type,
                                mg_dof_handlers_,
                                mg_constraints_,
                                initialize_dof_vector,
                                restrict_is_transpose_prolongate,
                                mg_type_level,
                                blk_indices);
  }

  template <int dim, typename Number>
  auto
  build_stmg_transfers(
    TimeStepType const                                         type,
    const MGLevelObject<std::vector<const DoFHandler<dim> *>> &mg_dof_handlers,
    const MGLevelObject<std::vector<const dealii::AffineConstraints<Number> *>>
                                                  &mg_constraints,
    const std::function<void(const unsigned int,
                             VectorT<Number> &,
                             const unsigned int)> &initialize_dof_vector,
    const bool                     restrict_is_transpose_prolongate,
    const std::vector<MGType>     &mg_type_level,
    const std::vector<BlockSlice> &blk_indices)
  {
    using BlockVectorType          = BlockVectorT<Number>;
    using VectorType               = VectorT<Number>;
    unsigned int const min_level   = mg_dof_handlers.min_level();
    unsigned int const max_level   = mg_dof_handlers.max_level();
    unsigned int const n_levels    = mg_dof_handlers.n_levels();
    unsigned int const n_variables = mg_dof_handlers.back().size();

    AssertDimension(n_levels - 1, mg_type_level.size());
    MGLevelObject<TwoLevelTransferOperator<dim, Number>> transfer(min_level,
                                                                  max_level);
    std::vector<unsigned int>                            n_blocks(n_levels);
    std::vector<std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>>>
      partitioners(
        n_levels,
        std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>>(
          n_variables));

    std::vector<
      std::vector<std::shared_ptr<MGTwoLevelTransfer<dim, VectorType>>>>
      space_transfers(
        n_levels,
        std::vector<std::shared_ptr<MGTwoLevelTransfer<dim, VectorType>>>(
          n_variables));

    unsigned int i = n_levels - 1;

    for (unsigned int l = min_level; l <= max_level; ++l)
      {
        for (unsigned int v = 0; v < n_variables; ++v)
          {
            VectorType vector;
            initialize_dof_vector(l, vector, v);
            partitioners[l - min_level][v] = vector.get_partitioner();
          }
      }
    for (auto mgt = mg_type_level.rbegin(); mgt != mg_type_level.rend();
         ++mgt, --i)
      {
        if (is_space_lvl(*mgt))
          {
            for (unsigned int v = 0; v < n_variables; ++v)
              {
                space_transfers[i][v] =
                  std::make_shared<MGTwoLevelTransfer<dim, VectorType>>();
                space_transfers[i][v]->reinit(*mg_dof_handlers[i][v],
                                              *mg_dof_handlers[i - 1][v],
                                              *mg_constraints[i][v],
                                              *mg_constraints[i - 1][v]);
                space_transfers[i][v]->enable_inplace_operations_if_possible(
                  partitioners[i - 1][v], partitioners[i][v]);
              }
            transfer[i] =
              TwoLevelTransferOperator<dim, Number>(blk_indices[i],
                                                    space_transfers[i]);
          }
        else
          transfer[i] = TwoLevelTransferOperator<dim, Number>(
            MGTwoLevelTransferTime<Number>(blk_indices[i],
                                           blk_indices[i - 1],
                                           type,
                                           restrict_is_transpose_prolongate,
                                           *mgt));
      }
    return std::make_unique<STMGTransferBlockMatrixFree<dim, Number>>(
      space_transfers, transfer, blk_indices, partitioners);
  }

  template <typename Number>
  class PreconditionVanka
  {
    using BlockVectorType = BlockVectorT<Number>;
    using VectorType      = VectorT<Number>;

  public:
    template <int dim>
    PreconditionVanka(
      TimerOutput                                           &timer,
      std::shared_ptr<const BlockSparseMatrixType> const    &K_,
      std::shared_ptr<const BlockSparseMatrixType> const    &M_,
      std::shared_ptr<const BlockSparsityPatternType> const &SP_,
      const FullMatrix<Number>                              &Alpha,
      const FullMatrix<Number>                              &Beta,
      std::vector<const DoFHandler<dim> *> const            &dof_handler,
      const BlockSlice                                      &blk_slice_,
      const Table<2, bool> &K_mask = Table<2, bool>(),
      const Table<2, bool> &M_mask = Table<2, bool>())
      : timer(timer)
      , blk_slice(blk_slice_)
    {
      AssertDimension(SP_->n_block_rows(), dof_handler.size());
      std::vector<IndexSet> locally_relevant_dofs(SP_->n_block_rows());
      BlockVectorType       valence(SP_->n_block_rows());

      for (unsigned int i = 0; i < locally_relevant_dofs.size(); ++i)
        {
          DoFTools::extract_locally_relevant_dofs(*dof_handler[i],
                                                  locally_relevant_dofs[i]);
          valence.block(i).reinit(dof_handler[i]->locally_owned_dofs(),
                                  locally_relevant_dofs[i],
                                  dof_handler[i]->get_communicator());
        }

      indices.resize(SP_->n_block_rows());
      for (unsigned int i = 0; i < dof_handler.size(); ++i)
        for (const auto &cell : dof_handler[i]->active_cell_iterators())
          {
            if (cell->is_locally_owned())
              {
                std::vector<types::global_dof_index> my_indices(
                  cell->get_fe().n_dofs_per_cell());
                cell->get_dof_indices(my_indices);
                for (auto const &dof_index : my_indices)
                  valence.block(i)[dof_index] += static_cast<Number>(1);

                indices[i].emplace_back(my_indices);
              }
          }

      valence.compress(VectorOperation::add);
      valence.update_ghost_values();

      auto K_blocks = SparseMatrixTools::restrict_to_full_block_matrices_(
        *K_, *SP_, indices, indices, valence, K_mask);
      auto M_blocks = SparseMatrixTools::restrict_to_full_block_matrices_(
        *M_, *SP_, indices, indices, valence, M_mask);
      unsigned int td =
        blk_slice.n_timedofs() * blk_slice.n_timesteps_at_once();

      blocks.resize(K_blocks(0, 0).size());
      for (unsigned int ii = 0; ii < blocks.size(); ++ii)
        {
          unsigned int n_sd = 0;
          for (unsigned int i = 0; i < blk_slice.n_variables(); ++i)
            n_sd += indices[i][ii].size();
          auto &B = blocks[ii];
          B       = FullMatrix<Number>(n_sd * td, n_sd * td);
          for (unsigned int i = 0, r_o = 0; i < blk_slice.n_blocks(); ++i)
            {
              auto const &[it, iv, id] = blk_slice.decompose(i);
              for (unsigned int j = 0, c_o = 0; j < blk_slice.n_blocks(); ++j)
                {
                  auto const &[jt, jv, jd] = blk_slice.decompose(j);
                  const auto &K            = K_blocks(iv, jv)[ii];
                  if (Beta(i, j) != 0.0 && (M_mask.empty() || M_mask(iv, jv)))
                    {
                      const auto &M = M_blocks(iv, jv)[ii];
                      for (unsigned int k = 0; k < M.m(); ++k)
                        for (unsigned int l = 0; l < M.n(); ++l)
                          B(r_o + k, c_o + l) += Beta(i, j) * M(k, l);
                    }
                  if (Alpha(i, j) != 0.0 && (K_mask.empty() || K_mask(iv, jv)))
                    for (unsigned int k = 0; k < K.m(); ++k)
                      for (unsigned int l = 0; l < K.n(); ++l)
                        B(r_o + k, c_o + l) += Alpha(i, j) * K(k, l);
                  c_o += indices[jv][ii].size();
                }
              r_o += indices[iv][ii].size();
            }
          B.gauss_jordan();
        }
    }

    template <int dim>
    PreconditionVanka(TimerOutput                                      &timer,
                      std::shared_ptr<const SparseMatrixType> const    &K_,
                      std::shared_ptr<const SparseMatrixType> const    &M_,
                      std::shared_ptr<const SparsityPatternType> const &SP_,
                      const FullMatrix<Number>                         &Alpha,
                      const FullMatrix<Number>                         &Beta,
                      std::shared_ptr<const DoFHandler<dim>> const &dof_handler)
      : timer(timer)
    {
      std::vector<FullMatrix<Number>> K_blocks, M_blocks;
      IndexSet                        locally_relevant_dofs;
      DoFTools::extract_locally_relevant_dofs(*dof_handler,
                                              locally_relevant_dofs);
      VectorType valence;
      valence.reinit(dof_handler->locally_owned_dofs(),
                     locally_relevant_dofs,
                     dof_handler->get_communicator());
      this->indices.resize(1);
      auto &indices = this->indices[0];

      for (const auto &cell : dof_handler->active_cell_iterators())
        {
          if (cell->is_locally_owned())
            {
              std::vector<types::global_dof_index> my_indices(
                cell->get_fe().n_dofs_per_cell());
              cell->get_dof_indices(my_indices);
              for (auto const &dof_index : my_indices)
                valence(dof_index) += static_cast<Number>(1);

              indices.emplace_back(my_indices);
            }
        }
      valence.compress(VectorOperation::add);
      valence.update_ghost_values();


      SparseMatrixTools::restrict_to_full_matrices_(
        *K_, *SP_, indices, indices, K_blocks, valence);
      SparseMatrixTools::restrict_to_full_matrices_(
        *M_, *SP_, indices, indices, M_blocks, valence);

      blocks.resize(K_blocks.size());
      for (unsigned int ii = 0; ii < blocks.size(); ++ii)
        {
          const auto &K = K_blocks[ii];
          const auto &M = M_blocks[ii];
          auto       &B = blocks[ii];

          B = FullMatrix<Number>(K.m() * Alpha.m(), K.n() * Alpha.n());

          for (unsigned int i = 0; i < Alpha.m(); ++i)
            for (unsigned int j = 0; j < Alpha.n(); ++j)
              if (Beta(i, j) != 0.0 || Alpha(i, j) != 0.0)
                for (unsigned int k = 0; k < K.m(); ++k)
                  for (unsigned int l = 0; l < K.n(); ++l)
                    B(k + i * K.m(), l + j * K.n()) =
                      Beta(i, j) * M(k, l) + Alpha(i, j) * K(k, l);

          B.gauss_jordan();
        }
    }

    void
    vmult(BlockVectorType &dst, const BlockVectorType &src) const
    {
      TimerOutput::Scope scope(timer, "vanka");

      dst = 0.0;

      Vector<Number>     dst_local;
      Vector<Number>     src_local;
      const unsigned int n_blocks = src.n_blocks();
      for (unsigned int i = 0; i < n_blocks; ++i)
        src.block(i).update_ghost_values();

      for (unsigned int i = 0; i < blocks.size(); ++i)
        {
          // gather
          src_local.reinit(blocks[i].m());
          dst_local.reinit(blocks[i].m());

          for (unsigned int b = 0, c = 0; b < n_blocks; ++b)
            {
              auto const &[tsp, v, td] = blk_slice.decompose(b);
              for (unsigned int j = 0; j < indices[v][i].size(); ++j, ++c)
                src_local[c] = src.block(b)[indices[v][i][j]];
            }
          // patch solver
          blocks[i].vmult(dst_local, src_local);

          // scatter
          for (unsigned int b = 0, c = 0; b < n_blocks; ++b)
            {
              auto const &[tsp, v, td] = blk_slice.decompose(b);
              for (unsigned int j = 0; j < indices[v][i].size(); ++j, ++c)
                dst.block(b)[indices[v][i][j]] += dst_local[c];
            }
        }

      for (unsigned int i = 0; i < n_blocks; ++i)
        src.block(i).zero_out_ghost_values();
      dst.compress(VectorOperation::add);
    }

    void
    clear()
    {
      indices.clear();
      blocks.clear();
    }

    void
    smooth(BlockVectorType &u, BlockVectorType const &rhs) const
    {
      vmult(u, rhs);
    }

    void
    set_blk_slice(BlockSlice const &blk_slice_)
    {
      AssertDimension(blk_slice_.n_variables(), 1);
      blk_slice = blk_slice_;
    }

  private:
    TimerOutput &timer;

    Number damp = 1.0;

    BlockSlice                                                     blk_slice;
    std::vector<std::vector<std::vector<types::global_dof_index>>> indices;
    std::vector<FullMatrix<Number>>                                blocks;
  };


  template <typename Number, typename PreconType>
  class PreconditionSTMG
  {
    using BlockVectorType = BlockVectorT<Number>;
    using VectorType      = VectorT<Number>;

  public:
    using AdditionalData = std::variant<PreconditionIdentity::AdditionalData,
                                        typename PreconType::AdditionalData>;
    static constexpr size_t id = 0, precon = 1;

    AdditionalData additional_data;

    PreconditionSTMG()
    {}

    PreconditionSTMG(PreconType const &smoother_)
      : smoother_variant(smoother_)
    {}


    template <typename MatrixType>
    void
    initialize(
      const MatrixType                                 &matrix,
      std::variant<PreconditionIdentity::AdditionalData,
                   typename PreconType::AdditionalData> additional_data)
    {
      if (additional_data.index() == id)
        {
          auto &s = smoother_variant.template emplace<id>();
          s.initialize(matrix, std::get<id>(additional_data));
        }
      else
        {
          auto &s = smoother_variant.template emplace<precon>();
          s.initialize(matrix, std::get<precon>(additional_data));
        }
    }

    void
    vmult(BlockVectorType &dst, const BlockVectorType &src) const
    {
      std::visit([&dst, &src](auto &s) { s.vmult(dst, src); },
                 smoother_variant);
    }
    void
    Tvmult(BlockVectorType &dst, const BlockVectorType &src) const
    {
      std::visit([&dst, &src](auto &s) { s.Tvmult(dst, src); },
                 smoother_variant);
    }
    void
    clear()
    {
      std::visit([](auto &s) { s.clear(); }, smoother_variant);
    }

    void
    smooth(BlockVectorType &u, BlockVectorType const &rhs) const
    {
      vmult(u, rhs);
    }

  private:
    std::variant<PreconditionIdentity, PreconType> smoother_variant;
  };

  struct PreconditionerGMGAdditionalData
  {
    double       smoothing_range               = 1;
    unsigned int smoothing_degree              = 5;
    unsigned int smoothing_eig_cg_n_iterations = 20;
    unsigned int smoothing_steps               = 1;

    double relaxation = 0.0;

    std::string coarse_grid_smoother_type = "Smoother";

    unsigned int coarse_grid_maxiter = 10;
    double       coarse_grid_abstol  = 1e-20;
    double       coarse_grid_reltol  = 1e-4;

    bool restrict_is_transpose_prolongate = true;
    bool variable                         = true;
  };
  template <int dim>
  struct Parameters
  {
    bool         do_output         = false;
    bool         print_timing      = false;
    bool         space_time_mg     = true;
    bool         time_before_space = false;
    TimeStepType type              = TimeStepType::CGP;
    ProblemType  problem           = ProblemType::wave;

    CoarseningType coarsening_type        = CoarseningType::space_or_time;
    bool           space_time_level_first = true;
    bool           use_pmg                = false;
    MGTransferGlobalCoarseningTools::PolynomialCoarseningSequenceType
      poly_coarsening = MGTransferGlobalCoarseningTools::
        PolynomialCoarseningSequenceType::decrease_by_one;
    unsigned int n_timesteps_at_once     = 1;
    int          n_timesteps_at_once_min = -1;
    unsigned int fe_degree               = 1;
    int          fe_degree_min           = -1;
    int          fe_degree_min_space     = -1;
    unsigned int n_deg_cycles            = 1;
    unsigned int n_ref_cycles            = 1;
    double       frequency               = 1.0;
    double       rel_tol                 = 1.0e-12;
    int          refinement              = 2;
    bool         space_time_conv_test    = true;
    bool         extrapolate             = true;
    bool         colorize_boundary       = false;
    bool         nitsche_boundary        = false;
    std::string  functional_file         = "functionals.txt";
    std::string  grid_descriptor         = "hyperRectangle";
    std::string  additional_file         = "";
    Point<dim>   hyperrect_lower_left =
      dim == 2 ? Point<dim>(0., 0.) : Point<dim>(0., 0., 0.);
    Point<dim> hyperrect_upper_right =
      dim == 2 ? Point<dim>(1., 1.) : Point<dim>(1., 1., 1.);
    std::vector<unsigned int> subdivisions  = std::vector<unsigned int>(dim, 1);
    double                    distort_grid  = 0.0;
    double                    distort_coeff = 0.0;
    Point<dim> source = .5 * hyperrect_lower_left + .5 * hyperrect_upper_right;
    double     end_time = 1.0;


    PreconditionerGMGAdditionalData mg_data;
    void
    parse(const std::string file_name)
    {
      std::string type_, problem_, p_mg_ = "decrease_by_one",
                                   c_type_ = "space_or_time";
      dealii::ParameterHandler prm;
      prm.add_parameter("doOutput", do_output);
      prm.add_parameter("printTiming", print_timing);
      prm.add_parameter("spaceTimeMg", space_time_mg);
      prm.add_parameter("mgTimeBeforeSpace", time_before_space);
      prm.add_parameter("timeType", type_);
      prm.add_parameter("problemType", problem_);
      prm.add_parameter("pMgType", p_mg_);
      prm.add_parameter("coarseningType", c_type_);
      prm.add_parameter("spaceTimeLevelFirst", space_time_level_first);
      prm.add_parameter("usePMg", use_pmg);
      prm.add_parameter("nTimestepsAtOnce", n_timesteps_at_once);
      prm.add_parameter("nTimestepsAtOnceMin", n_timesteps_at_once_min);
      prm.add_parameter("feDegree", fe_degree);
      prm.add_parameter("feDegreeMin", fe_degree_min);
      prm.add_parameter("feDegreeMinSpace", fe_degree_min_space);
      prm.add_parameter("nDegCycles", n_deg_cycles);
      prm.add_parameter("nRefCycles", n_ref_cycles);
      prm.add_parameter("frequency", frequency);
      prm.add_parameter("relativeTolerance", rel_tol);
      prm.add_parameter("refinement", refinement);
      prm.add_parameter("spaceTimeConvergenceTest", space_time_conv_test);
      prm.add_parameter("extrapolate", extrapolate);
      prm.add_parameter("colorizeBoundary", colorize_boundary);
      prm.add_parameter("nitscheBoundary", nitsche_boundary);
      prm.add_parameter("functionalFile", functional_file);
      prm.add_parameter("gridDescriptor", grid_descriptor);
      prm.add_parameter("additionalFile", additional_file);
      prm.add_parameter("hyperRectLowerLeft", hyperrect_lower_left);
      prm.add_parameter("hyperRectUpperRight", hyperrect_upper_right);
      prm.add_parameter("subdivisions", subdivisions);
      prm.add_parameter("distortGrid", distort_grid);
      prm.add_parameter("distortCoeff", distort_coeff);
      prm.add_parameter("sourcePoint", source);
      prm.add_parameter("endTime", end_time);

      prm.add_parameter("smoothingDegree", mg_data.smoothing_degree);
      prm.add_parameter("smoothingSteps", mg_data.smoothing_steps);
      prm.add_parameter("smoothingRange", mg_data.smoothing_range);
      prm.add_parameter("relaxation", mg_data.relaxation);
      prm.add_parameter("coarseGridSmootherType",
                        mg_data.coarse_grid_smoother_type);
      prm.add_parameter("coarseGridMaxiter", mg_data.coarse_grid_maxiter);
      prm.add_parameter("coarseGridAbstol", mg_data.coarse_grid_abstol);
      prm.add_parameter("coarseGridReltol", mg_data.coarse_grid_reltol);
      prm.add_parameter("restrictIsTransposeProlongate",
                        mg_data.restrict_is_transpose_prolongate);
      prm.add_parameter("variable", mg_data.variable);
      AssertIsFinite(frequency);
      AssertIsFinite(distort_grid);
      AssertIsFinite(distort_coeff);
      AssertIsFinite(end_time);

      std::ifstream file;
      file.open(file_name);
      prm.parse_input_from_json(file, true);
      type            = str_to_time_type.at(type_);
      problem         = str_to_problem_type.at(problem_);
      poly_coarsening = str_to_polynomial_coarsening_type.at(p_mg_);
      coarsening_type = str_to_coarsening_type.at(c_type_);
      if (n_timesteps_at_once_min == -1)
        n_timesteps_at_once_min = n_timesteps_at_once / 2;

      n_timesteps_at_once_min =
        std::clamp(n_timesteps_at_once_min,
                   1,
                   static_cast<int>(n_timesteps_at_once));
      const int lowest_degree = type == TimeStepType::DG ? 0 : 1;
      if (fe_degree_min == -1)
        fe_degree_min = fe_degree - 1;
      fe_degree_min =
        std::clamp(fe_degree_min, lowest_degree, static_cast<int>(fe_degree));
      if (fe_degree_min_space == -1)
        fe_degree_min_space = fe_degree_min;
    }
  };

  template <int dim, typename Number, typename LevelMatrixType>
  class GMG
  {
    using BlockVectorType = BlockVectorT<Number>;
    using VectorType      = VectorT<Number>;

    using MGTransferType = STMGTransferBlockMatrixFree<dim, Number>;

    using SmootherPreconditionerType = PreconditionVanka<Number>;
    using CoarsePreconditionerType =
      PreconditionRelaxation<LevelMatrixType, SmootherPreconditionerType>;
    using SmootherType = PreconditionSTMG<
      Number,
      PreconditionRelaxation<LevelMatrixType, SmootherPreconditionerType>>;
    using MGSmootherType =
      MGSmootherPrecondition<LevelMatrixType, SmootherType, BlockVectorType>;

  public:
    GMG(
      TimerOutput                                &timer,
      Parameters<dim> const                      &parameters,
      unsigned int const                          n_timesteps_at_once,
      const std::vector<MGType>                  &mg_type_level,
      std::vector<unsigned int> const            &poly_time_sequence,
      const std::vector<const DoFHandler<dim> *> &dof_handler,
      const MGLevelObject<std::vector<const DoFHandler<dim> *>>
        &mg_dof_handlers,
      const MGLevelObject<
        std::vector<const dealii::AffineConstraints<Number> *>> &mg_constraints,
      const MGLevelObject<std::shared_ptr<const LevelMatrixType>> &mg_operators,
      const MGLevelObject<std::shared_ptr<SmootherPreconditionerType>>
                                        &mg_smoother_,
      std::unique_ptr<BlockVectorType> &&tmp1,
      std::unique_ptr<BlockVectorType> &&tmp2)
      : timer(timer)
      , additional_data(parameters.mg_data)
      , min_level(mg_dof_handlers.min_level())
      , max_level(mg_dof_handlers.max_level())
      , mg_sequence(mg_type_level)
      , precondition_sequence(
          get_precondition_stmg_types(mg_sequence,
                                      parameters.coarsening_type,
                                      parameters.time_before_space,
                                      parameters.space_time_level_first))
      , src_(std::move(tmp1))
      , dst_(std::move(tmp2))
      , dof_handler(dof_handler)
      , mg_dof_handlers(mg_dof_handlers)
      , mg_constraints(mg_constraints)
      , mg_operators(mg_operators)
      , precondition_vanka(mg_smoother_)

    {
      unsigned int const n_levels    = mg_dof_handlers.n_levels();
      unsigned int const n_variables = mg_dof_handlers.back().size();

      std::vector<BlockSlice> blk_indices = get_blk_indices(parameters.type,
                                                            n_timesteps_at_once,
                                                            n_variables,
                                                            n_levels,
                                                            mg_type_level,
                                                            poly_time_sequence);

      transfer_block = build_stmg_transfers<dim, Number>(
        parameters.type,
        mg_dof_handlers,
        mg_constraints,
        [&](const unsigned int l, VectorType &vec, const unsigned int v) {
          this->mg_operators[l]->initialize_dof_vector(vec, v);
        },
        additional_data.restrict_is_transpose_prolongate,
        mg_type_level,
        blk_indices);
    }
    GMG(
      TimerOutput                     &timer,
      Parameters<dim> const           &parameters,
      unsigned int const               n_timesteps_at_once,
      const std::vector<MGType>       &mg_type_level,
      const std::vector<unsigned int> &poly_time_sequence,
      const DoFHandler<dim>           &dof_handler,
      const MGLevelObject<std::shared_ptr<const DoFHandler<dim>>>
        &mg_dof_handlers,
      const MGLevelObject<std::shared_ptr<const AffineConstraints<Number>>>
        &mg_constraints,
      const MGLevelObject<std::shared_ptr<const LevelMatrixType>> &mg_operators,
      const MGLevelObject<std::shared_ptr<SmootherPreconditionerType>>
                                        &mg_smoother_,
      std::unique_ptr<BlockVectorType> &&tmp1,
      std::unique_ptr<BlockVectorType> &&tmp2)
      : timer(timer)
      , additional_data(parameters.mg_data)
      , min_level(mg_dof_handlers.min_level())
      , max_level(mg_dof_handlers.max_level())
      , mg_sequence(mg_type_level)
      , precondition_sequence(
          get_precondition_stmg_types(mg_sequence,
                                      parameters.coarsening_type,
                                      parameters.time_before_space,
                                      parameters.space_time_level_first))
      , src_(std::move(tmp1))
      , dst_(std::move(tmp2))
      , dof_handler(1, &dof_handler)
      , mg_dof_handlers(min_level, max_level)
      , mg_constraints(min_level, max_level)
      , mg_operators(mg_operators)
      , precondition_vanka(mg_smoother_)
    {
      for (unsigned int l = min_level; l <= max_level; ++l)
        {
          this->mg_dof_handlers[l].push_back(mg_dof_handlers[l].get());
          this->mg_constraints[l].push_back(mg_constraints[l].get());
        }
      unsigned int const      n_levels    = mg_dof_handlers.n_levels();
      std::vector<BlockSlice> blk_indices = get_blk_indices(parameters.type,
                                                            n_timesteps_at_once,
                                                            1u,
                                                            n_levels,
                                                            mg_type_level,
                                                            poly_time_sequence);

      transfer_block = build_stmg_transfers<dim, Number>(
        parameters.type,
        this->mg_dof_handlers,
        this->mg_constraints,
        [&](const unsigned int l, VectorType &vec, const unsigned int) {
          this->mg_operators[l]->initialize_dof_vector(vec);
        },
        additional_data.restrict_is_transpose_prolongate,
        mg_type_level,
        blk_indices);
      for (unsigned int i = 0, l = min_level; l <= max_level; ++i, ++l)
        precondition_vanka[l]->set_blk_slice(blk_indices[i]);
    }

    void
    reinit() const
    {
      // wrap level operators
      mg_matrix = mg::Matrix<BlockVectorType>(mg_operators);

      MGLevelObject<typename SmootherType::AdditionalData> smoother_data(
        min_level, max_level);

      // setup smoothers on each level
      for (unsigned int level = min_level, i = 0; level <= max_level;
           ++level, ++i)
        {
          if (precondition_sequence[i] == SmootherType::precon)
            {
              auto &sd =
                smoother_data[level].template emplace<SmootherType::precon>();
              sd.preconditioner = precondition_vanka[level];
              sd.n_iterations   = additional_data.smoothing_steps;
              sd.relaxation     = additional_data.relaxation;
              sd.eigenvalue_algorithm =
                internal::EigenvalueAlgorithm::power_iteration;
              sd.eig_cg_n_iterations =
                additional_data.smoothing_eig_cg_n_iterations;
              sd.smoothing_range = additional_data.smoothing_range;
            }
          else
            smoother_data[level].template emplace<SmootherType::id>();
        }
      mg_smoother = std::make_unique<MGSmootherType>(1,
                                                     additional_data.variable,
                                                     false,
                                                     false);
      mg_smoother->initialize(mg_operators, smoother_data);

      if (additional_data.coarse_grid_smoother_type != "Smoother")
        {
          solver_control_coarse = std::make_unique<IterationNumberControl>(
            additional_data.coarse_grid_maxiter,
            additional_data.coarse_grid_abstol,
            false,
            false);
          typename SolverGMRES<BlockVectorType>::AdditionalData const
            gmres_additional_data(additional_data.coarse_grid_maxiter);
          gmres_coarse = std::make_unique<SolverGMRES<BlockVectorType>>(
            *solver_control_coarse, gmres_additional_data);
          typename CoarsePreconditionerType::AdditionalData coarse_precon_data;
          coarse_precon_data.relaxation     = additional_data.relaxation;
          coarse_precon_data.n_iterations   = additional_data.smoothing_steps;
          coarse_precon_data.preconditioner = precondition_vanka[min_level];
          coarse_precon_data.eigenvalue_algorithm =
            internal::EigenvalueAlgorithm::power_iteration;
          coarse_precon_data.eig_cg_n_iterations =
            additional_data.smoothing_eig_cg_n_iterations;
          coarse_precon_data.smoothing_range = additional_data.smoothing_range;

          coarse_preconditioner = std::make_unique<CoarsePreconditionerType>();
          coarse_preconditioner->initialize(*(mg_operators[min_level]),
                                            coarse_precon_data);

          mg_coarse = std::make_unique<
            dealii::MGCoarseGridIterativeSolver<BlockVectorType,
                                                SolverGMRES<BlockVectorType>,
                                                LevelMatrixType,
                                                CoarsePreconditionerType>>(
            *gmres_coarse, *mg_operators[min_level], *coarse_preconditioner);
        }
      else
        {
          mg_coarse =
            std::make_unique<MGCoarseGridApplySmoother<BlockVectorType>>(
              *mg_smoother);
        }

      // create multigrid algorithm (put level operators, smoothers, transfer
      // operators and smoothers together)
      mg = std::make_unique<Multigrid<BlockVectorType>>(mg_matrix,
                                                        *mg_coarse,
                                                        *transfer_block,
                                                        *mg_smoother,
                                                        *mg_smoother,
                                                        min_level,
                                                        max_level);

      // convert multigrid algorithm to preconditioner
      if (dof_handler.size() == 1)
        preconditioner = std::make_unique<
          PreconditionMG<dim, BlockVectorType, MGTransferType>>(
          *dof_handler[0], *mg, *transfer_block);
      else
        preconditioner = std::make_unique<
          PreconditionMG<dim, BlockVectorType, MGTransferType>>(
          dof_handler, *mg, *transfer_block);
    }

    template <typename SolutionVectorType = BlockVectorType>
    void
    vmult(SolutionVectorType &dst, const SolutionVectorType &src) const
    {
      TimerOutput::Scope scope(timer, "gmg");
      if (std::is_same_v<SolutionVectorType, BlockVectorType>)
        preconditioner->vmult(dst, src);
      else
        {
          src_->copy_locally_owned_data_from(src);
          preconditioner->vmult(*dst_, *src_);
          dst.copy_locally_owned_data_from(*dst_);
        }
    }

    std::unique_ptr<const GMG<dim, Number, LevelMatrixType>>
    clone() const
    {
      return std::make_unique<GMG<dim, Number, LevelMatrixType>>(
        dof_handler, mg_dof_handlers, mg_constraints, mg_operators);
    }

  private:
    TimerOutput &timer;

    PreconditionerGMGAdditionalData additional_data;
    const unsigned int              min_level;
    const unsigned int              max_level;
    std::vector<MGType>             mg_sequence;
    std::vector<size_t>             precondition_sequence;

    std::unique_ptr<BlockVectorType> src_;
    std::unique_ptr<BlockVectorType> dst_;

    std::vector<const DoFHandler<dim> *>                dof_handler;
    MGLevelObject<std::vector<const DoFHandler<dim> *>> mg_dof_handlers;
    MGLevelObject<std::vector<const dealii::AffineConstraints<Number> *>>
                                                                mg_constraints;
    const MGLevelObject<std::shared_ptr<const LevelMatrixType>> mg_operators;
    const MGLevelObject<std::shared_ptr<SmootherPreconditionerType>>
      precondition_vanka;



    std::unique_ptr<MGTransferType> transfer_block;

    mutable mg::Matrix<BlockVectorType> mg_matrix;


    mutable std::unique_ptr<MGSmootherType> mg_smoother;

    mutable std::unique_ptr<MGCoarseGridBase<BlockVectorType>> mg_coarse;
    mutable std::unique_ptr<SolverControl>                solver_control_coarse;
    mutable std::unique_ptr<SolverGMRES<BlockVectorType>> gmres_coarse;
    mutable std::unique_ptr<CoarsePreconditionerType>     coarse_preconditioner;

    mutable std::unique_ptr<Multigrid<BlockVectorType>> mg;

    mutable std::unique_ptr<
      PreconditionMG<dim, BlockVectorType, MGTransferType>>
      preconditioner;
  };
} // namespace dealii
