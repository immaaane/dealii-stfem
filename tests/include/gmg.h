#pragma once

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/sparse_matrix_tools.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/multigrid.h>

#include "fe_time.h"
#include "types.h"
namespace dealii
{
  template <typename Number>
  class PreconditionVanka
  {
    using BlockVectorType = BlockVectorT<Number>;
    using VectorType      = VectorT<Number>;

  public:
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

      valence.reinit(dof_handler->locally_owned_dofs(),
                     locally_relevant_dofs,
                     dof_handler->get_communicator());

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


      SparseMatrixTools::restrict_to_full_matrices(*K_,
                                                   *SP_,
                                                   indices,
                                                   K_blocks);
      SparseMatrixTools::restrict_to_full_matrices(*M_,
                                                   *SP_,
                                                   indices,
                                                   M_blocks);

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

      valence.update_ghost_values();
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
            for (unsigned int j = 0; j < indices[i].size(); ++j, ++c)
              src_local[c] = src.block(b)[indices[i][j]];

          // patch solver
          blocks[i].vmult(dst_local, src_local);

          // scatter
          for (unsigned int b = 0, c = 0; b < n_blocks; ++b)
            for (unsigned int j = 0; j < indices[i].size(); ++j, ++c)
              {
                Number const weight = damp / valence[indices[i][j]];
                dst.block(b)[indices[i][j]] += weight * dst_local[c];
              }
        }

      for (unsigned int i = 0; i < n_blocks; ++i)
        src.block(i).zero_out_ghost_values();
      dst.compress(VectorOperation::add);
    }

    void
    clear()
    {}

    void
    smooth(BlockVectorType &u, BlockVectorType const &rhs) const
    {
      vmult(u, rhs);
    }



  private:
    TimerOutput &timer;

    Number damp = 1.0;

    std::vector<std::vector<types::global_dof_index>> indices;
    VectorType                                        valence;
    std::vector<FullMatrix<Number>>                   blocks;
  };

  struct PreconditionerGMGAdditionalData
  {
    double       smoothing_range               = 20;
    unsigned int smoothing_degree              = 5;
    unsigned int smoothing_eig_cg_n_iterations = 20;

    bool estimate_relaxation = false;

    unsigned int coarse_grid_smoother_sweeps = 1;
    unsigned int coarse_grid_n_cycles        = 1;
    std::string  coarse_grid_smoother_type   = "Smoother";

    unsigned int coarse_grid_maxiter = 1000;
    double       coarse_grid_abstol  = 1e-20;
    double       coarse_grid_reltol  = 1e-4;
  };
  template <int dim, typename Number, typename LevelMatrixType>
  class GMG
  {
    using BlockVectorType = BlockVectorT<Number>;
    using VectorType      = VectorT<Number>;

    using MGTransferType = MGTransferBlockGlobalCoarsening<dim, VectorType>;

    using SmootherPreconditionerType = PreconditionVanka<Number>;
    using SmootherType =
      PreconditionRelaxation<LevelMatrixType, SmootherPreconditionerType>;
    using MGSmootherType =
      MGSmootherPrecondition<LevelMatrixType, SmootherType, BlockVectorType>;

  public:
    GMG(
      TimerOutput           &timer,
      const DoFHandler<dim> &dof_handler,
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
      , src_(std::move(tmp1))
      , dst_(std::move(tmp2))
      , dof_handler(dof_handler)
      , mg_dof_handlers(mg_dof_handlers)
      , mg_constraints(mg_constraints)
      , mg_operators(mg_operators)
      , precondition_vanka(mg_smoother_)
      , min_level(mg_dof_handlers.min_level())
      , max_level(mg_dof_handlers.max_level())
      , transfers(min_level, max_level)
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

      MGLevelObject<typename SmootherType::AdditionalData> smoother_data(
        min_level, max_level);

      // setup smoothers on each level
      for (unsigned int level = min_level; level <= max_level; ++level)
        {
          smoother_data[level].preconditioner = precondition_vanka[level];
          smoother_data[level].n_iterations   = 1;
          smoother_data[level].relaxation =
            additional_data.estimate_relaxation ? estimate_relaxation(level) :
                                                  1.0;
        }
      mg_smoother = std::make_unique<MGSmootherType>(1, true, false, false);
      mg_smoother->initialize(mg_operators, smoother_data);

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

    double
    estimate_relaxation(unsigned level) const
    {
      if (level == 0)
        return 1;

      PreconditionerGMGAdditionalData additional_data;
      using ChebyshevPreconditionerType =
        PreconditionChebyshev<LevelMatrixType,
                              BlockVectorType,
                              SmootherPreconditionerType>;

      typename ChebyshevPreconditionerType::AdditionalData
        chebyshev_additional_data;
      chebyshev_additional_data.preconditioner = precondition_vanka[level];
      chebyshev_additional_data.smoothing_range =
        additional_data.smoothing_range;
      chebyshev_additional_data.degree = additional_data.smoothing_degree;
      chebyshev_additional_data.eig_cg_n_iterations =
        additional_data.smoothing_eig_cg_n_iterations;
      chebyshev_additional_data.eigenvalue_algorithm =
        ChebyshevPreconditionerType::AdditionalData::EigenvalueAlgorithm::
          power_iteration;
      chebyshev_additional_data.polynomial_type = ChebyshevPreconditionerType::
        AdditionalData::PolynomialType::fourth_kind;
      auto chebyshev = std::make_shared<ChebyshevPreconditionerType>();
      chebyshev->initialize(*mg_operators[level], chebyshev_additional_data);

      BlockVectorType vec;
      mg_operators[level]->initialize_dof_vector(vec);

      const auto evs = chebyshev->estimate_eigenvalues(vec);

      const double alpha = (chebyshev_additional_data.smoothing_range > 1. ?
                              evs.max_eigenvalue_estimate /
                                chebyshev_additional_data.smoothing_range :
                              std::min(0.9 * evs.max_eigenvalue_estimate,
                                       evs.min_eigenvalue_estimate));

      double omega = 2.0 / (alpha + evs.max_eigenvalue_estimate);
      deallog << "\n-Eigenvalue estimation level " << level
              << ":\n"
                 "    Relaxation parameter: "
              << omega
              << "\n"
                 "    Minimum eigenvalue: "
              << evs.min_eigenvalue_estimate
              << "\n"
                 "    Maximum eigenvalue: "
              << evs.max_eigenvalue_estimate << std::endl;

      return omega;
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

    std::unique_ptr<BlockVectorType> src_;
    std::unique_ptr<BlockVectorType> dst_;

    const DoFHandler<dim>                                      &dof_handler;
    const MGLevelObject<std::shared_ptr<const DoFHandler<dim>>> mg_dof_handlers;
    const MGLevelObject<std::shared_ptr<const AffineConstraints<Number>>>
                                                                mg_constraints;
    const MGLevelObject<std::shared_ptr<const LevelMatrixType>> mg_operators;
    const MGLevelObject<std::shared_ptr<SmootherPreconditionerType>>
      precondition_vanka;

    const unsigned int min_level;
    const unsigned int max_level;

    MGLevelObject<MGTwoLevelTransfer<dim, VectorType>> transfers;
    std::unique_ptr<MGTransferGlobalCoarsening<dim, VectorType>>
                                    transfer_scalar;
    std::unique_ptr<MGTransferType> transfer_block;

    mutable mg::Matrix<BlockVectorType> mg_matrix;


    mutable std::unique_ptr<MGSmootherType> mg_smoother;

    mutable std::unique_ptr<MGCoarseGridBase<BlockVectorType>> mg_coarse;
    mutable std::unique_ptr<Multigrid<BlockVectorType>>        mg;
    mutable std::unique_ptr<
      PreconditionMG<dim, BlockVectorType, MGTransferType>>
      preconditioner;
  };
} // namespace dealii
