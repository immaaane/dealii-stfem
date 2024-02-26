#pragma once

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/sparse_matrix_tools.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/multigrid.h>

#include "types.h"

namespace dealii
{
  template <typename Number>
  class VankaSmoother : public MGSmoother<BlockVectorType>
  {
  public:
    struct AdditionalData
    {};
    template <int dim>
    VankaSmoother(
      TimerOutput                                                     &timer,
      MGLevelObject<std::shared_ptr<const SparseMatrixType>> const    &K_,
      MGLevelObject<std::shared_ptr<const SparseMatrixType>> const    &M_,
      MGLevelObject<std::shared_ptr<const SparsityPatternType>> const &SP_,
      const FullMatrix<Number>                                        &Alpha,
      const FullMatrix<Number>                                        &Beta,
      MGLevelObject<std::shared_ptr<const DoFHandler<dim>>> const
        &mg_dof_handlers)
      : dealii::MGSmoother<BlockVectorType>(
          1,     // number of smoothing steps
          true,  // double number of smoothing steps
          false, // symmetric = symmetric smoothing (false)
          false)
      , timer(timer)
      , indices(mg_dof_handlers.min_level(), mg_dof_handlers.max_level())
      , valence(mg_dof_handlers.min_level(), mg_dof_handlers.max_level())
      , blocks(mg_dof_handlers.min_level(), mg_dof_handlers.max_level())
    {
      std::vector<FullMatrix<Number>> K_blocks, M_blocks;
      for (unsigned int l = mg_dof_handlers.min_level();
           l <= mg_dof_handlers.max_level();
           ++l)
        {
          IndexSet locally_relevant_dofs;
          DoFTools::extract_locally_relevant_dofs(*mg_dof_handlers[l],
                                                  locally_relevant_dofs);

          valence[l].reinit(mg_dof_handlers[l]->locally_owned_dofs(),
                            locally_relevant_dofs,
                            mg_dof_handlers[l]->get_communicator());
          for (const auto &cell : mg_dof_handlers[l]->active_cell_iterators())
            {
              if (cell->is_locally_owned())
                {
                  std::vector<types::global_dof_index> my_indices(
                    cell->get_fe().n_dofs_per_cell());
                  cell->get_dof_indices(my_indices);
                  for (auto const &dof_index : my_indices)
                    valence[l](dof_index) += static_cast<Number>(1);

                  indices[l].emplace_back(my_indices);
                }
            }
          valence[l].compress(VectorOperation::add);


          SparseMatrixTools::restrict_to_full_matrices(*K_[l],
                                                       *SP_[l],
                                                       indices[l],
                                                       K_blocks);
          SparseMatrixTools::restrict_to_full_matrices(*M_[l],
                                                       *SP_[l],
                                                       indices[l],
                                                       M_blocks);

          blocks[l].resize(K_blocks.size());
          for (unsigned int ii = 0; ii < blocks[l].size(); ++ii)
            {
              const auto &K = K_blocks[ii];
              const auto &M = M_blocks[ii];
              auto       &B = blocks[l][ii];

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
      for (unsigned int l = mg_dof_handlers.min_level();
           l <= mg_dof_handlers.max_level();
           ++l)
        valence[l].update_ghost_values();
    }

    void
    vmult(unsigned int           l,
          BlockVectorType       &dst,
          const BlockVectorType &src) const
    {
      TimerOutput::Scope scope(timer, "vanka");

      dst = 0.0;

      Vector<Number>     dst_local;
      Vector<Number>     src_local;
      auto const        &indices_l = indices[l];
      auto const        &blocks_l  = blocks[l];
      const unsigned int n_blocks  = src.n_blocks();
      for (unsigned int i = 0; i < n_blocks; ++i)
        src.block(i).update_ghost_values();

      for (unsigned int i = 0; i < blocks_l.size(); ++i)
        {
          // gather
          src_local.reinit(blocks_l[i].m());
          dst_local.reinit(blocks_l[i].m());

          for (unsigned int b = 0, c = 0; b < n_blocks; ++b)
            for (unsigned int j = 0; j < indices_l[i].size(); ++j, ++c)
              src_local[c] = src.block(b)[indices_l[i][j]];

          // patch solver
          blocks_l[i].vmult(dst_local, src_local);

          // scatter
          for (unsigned int b = 0, c = 0; b < n_blocks; ++b)
            for (unsigned int j = 0; j < indices_l[i].size(); ++j, ++c)
              {
                Number const weight = damp / valence[l][indices_l[i][j]];
                dst.block(b)[indices_l[i][j]] += weight * dst_local[c];
              }
        }

      for (unsigned int i = 0; i < n_blocks; ++i)
        src.block(i).zero_out_ghost_values();
      dst.compress(VectorOperation::add);
    }

    void
    clear() override final
    {}

    void
    smooth(unsigned int const     level,
           BlockVectorType       &u,
           BlockVectorType const &rhs) const override final
    {
      vmult(level, u, rhs);
    }



  private:
    TimerOutput &timer;

    AdditionalData additional_data;
    Number         damp = 1.0;

    MGLevelObject<std::vector<std::vector<types::global_dof_index>>> indices;
    MGLevelObject<VectorType>                                        valence;
    MGLevelObject<std::vector<FullMatrix<Number>>>                   blocks;
  };

  struct PreconditionerGMGAdditionalData
  {
    double       smoothing_range               = 20;
    unsigned int smoothing_degree              = 5;
    unsigned int smoothing_eig_cg_n_iterations = 20;

    unsigned int coarse_grid_smoother_sweeps = 1;
    unsigned int coarse_grid_n_cycles        = 1;
    std::string  coarse_grid_smoother_type   = "Smoother";

    unsigned int coarse_grid_maxiter = 1000;
    double       coarse_grid_abstol  = 1e-20;
    double       coarse_grid_reltol  = 1e-4;
  };
  template <int dim, typename LevelMatrixType>
  class GMG
  {
    using MGTransferType = MGTransferBlockGlobalCoarsening<dim, VectorType>;

  public:
    GMG(
      TimerOutput           &timer,
      const DoFHandler<dim> &dof_handler,
      const MGLevelObject<std::shared_ptr<const DoFHandler<dim>>>
        &mg_dof_handlers,
      const MGLevelObject<std::shared_ptr<const AffineConstraints<Number>>>
        &mg_constraints,
      const MGLevelObject<std::shared_ptr<const LevelMatrixType>> &mg_operators,
      const std::shared_ptr<MGSmootherBase<BlockVectorType>>      &mg_smoother_)
      : timer(timer)
      , dof_handler(dof_handler)
      , mg_dof_handlers(mg_dof_handlers)
      , mg_constraints(mg_constraints)
      , mg_operators(mg_operators)
      , min_level(mg_dof_handlers.min_level())
      , max_level(mg_dof_handlers.max_level())
      , transfers(min_level, max_level)
      , mg_smoother(mg_smoother_)
    {
      // setup transfer operators
      for (auto l = min_level; l < max_level; ++l)
        transfers[l + 1].reinit(*mg_dof_handlers[l + 1],
                                *mg_dof_handlers[l],
                                *mg_constraints[l + 1],
                                *mg_constraints[l]);
      transfer_scalar =
        std::make_unique<MGTransferGlobalCoarsening<dim, VectorType>>(
          transfers, [&](const auto l, auto &vec) {
            this->mg_operators[l]->initialize_dof_vector(vec);
          });

      transfer_block =
        std::make_unique<MGTransferBlockGlobalCoarsening<dim, VectorType>>(
          *transfer_scalar);
    }

    void
    reinit() const
    {
      PreconditionerGMGAdditionalData additional_data;

      // wrap level operators
      mg_matrix = mg::Matrix<BlockVectorType>(mg_operators);

      // setup smoothers on each level
      for (unsigned int level = min_level; level <= max_level; ++level)
        {
          BlockVectorType vec;
          mg_operators[level]->initialize_dof_vector(vec);
        }

      if (additional_data.coarse_grid_smoother_type != "Smoother")
        {
          // setup coarse-grid solver
          const auto coarse_comm =
            mg_dof_handlers[min_level]->get_communicator();
          if (coarse_comm != MPI_COMM_NULL)
            {
            }
          else
            {
            }
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
      preconditioner =
        std::make_unique<PreconditionMG<dim, BlockVectorType, MGTransferType>>(
          dof_handler, *mg, *transfer_block);
    }

    void
    vmult(BlockVectorType &dst, const BlockVectorType &src) const
    {
      TimerOutput::Scope scope(timer, "gmg");
      preconditioner->vmult(dst, src);
    }

    std::unique_ptr<const GMG<dim, LevelMatrixType>>
    clone() const
    {
      return std::make_unique<GMG<dim, LevelMatrixType>>(dof_handler,
                                                         mg_dof_handlers,
                                                         mg_constraints,
                                                         mg_operators);
    }

  private:
    TimerOutput &timer;

    // using SmootherPreconditionerType = DiagonalMatrix<VectorType>;
    const DoFHandler<dim>                                      &dof_handler;
    const MGLevelObject<std::shared_ptr<const DoFHandler<dim>>> mg_dof_handlers;
    const MGLevelObject<std::shared_ptr<const AffineConstraints<Number>>>
                                                                mg_constraints;
    const MGLevelObject<std::shared_ptr<const LevelMatrixType>> mg_operators;

    const unsigned int min_level;
    const unsigned int max_level;

    MGLevelObject<MGTwoLevelTransfer<dim, VectorType>> transfers;
    std::unique_ptr<MGTransferGlobalCoarsening<dim, VectorType>>
                                    transfer_scalar;
    std::unique_ptr<MGTransferType> transfer_block;

    mutable mg::Matrix<BlockVectorType>                      mg_matrix;
    mutable std::shared_ptr<MGSmootherBase<BlockVectorType>> mg_smoother;

    mutable std::unique_ptr<MGCoarseGridBase<BlockVectorType>> mg_coarse;
    mutable std::unique_ptr<Multigrid<BlockVectorType>>        mg;
    mutable std::unique_ptr<
      PreconditionMG<dim, BlockVectorType, MGTransferType>>
      preconditioner;
  };
} // namespace dealii
