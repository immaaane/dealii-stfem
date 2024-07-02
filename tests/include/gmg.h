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

#include "fe_time.h"
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
    std::vector<SmartPointer<const MGTwoLevelTransfer<dim, VectorType>>>
      transfer;

  public:
    MGTwoLevelBlockTransfer() = default;
    MGTwoLevelBlockTransfer(
      MGTwoLevelTransfer<dim, VectorType> const &transfer_)
      : transfer(1,
                 &const_cast<MGTwoLevelTransfer<dim, VectorType> &>(
                   static_cast<const MGTwoLevelTransfer<dim, VectorType> &>(
                     Utilities::get_underlying_value(transfer_))))
    {}
    MGTwoLevelBlockTransfer(MGTwoLevelBlockTransfer const &other)
      : transfer(other.transfer)
    {}

    MGTwoLevelBlockTransfer(
      std::vector<MGTwoLevelTransfer<dim, VectorType>> const &)
    {
      Assert(false, ExcInternalError()); // Implement for multiple variables
    }

    void
    prolongate_and_add(BlockVectorType &dst, const BlockVectorType &src) const
    {
      for (unsigned int i = 0; i < src.n_blocks(); ++i)
        transfer.front()->prolongate_and_add(dst.block(i), src.block(i));
    }

    void
    restrict_and_add(BlockVectorType &dst, const BlockVectorType &src) const
    {
      for (unsigned int i = 0; i < src.n_blocks(); ++i)
        transfer.front()->restrict_and_add(dst.block(i), src.block(i));
    }
  };

  template <typename Number>
  class MGTwoLevelTransferST
  {
    using BlockVectorType = BlockVectorT<Number>;
    FullMatrix<Number> prolongation_matrix;
    FullMatrix<Number> restriction_matrix;

  public:
    MGTwoLevelTransferST() = default;
    MGTwoLevelTransferST(TimeStepType const type,
                         unsigned int       r,
                         unsigned int       n_timesteps_at_once,
                         const bool restrict_is_transpose_prolongate = false,
                         TimeMGType mg_type = TimeMGType::tau)
    {
      bool k_mg = mg_type == TimeMGType::k;

      prolongation_matrix =
        k_mg ?
          get_time_projection_matrix<Number>(type,
                                             r - 1,
                                             r,
                                             n_timesteps_at_once) :
          get_time_prolongation_matrix<Number>(type, r, n_timesteps_at_once);
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
                                               r - 1,
                                               n_timesteps_at_once) :
            get_time_restriction_matrix<Number>(type, r, n_timesteps_at_once);
    }
    MGTwoLevelTransferST(MGTwoLevelTransferST<Number> const &other)
      : prolongation_matrix(other.prolongation_matrix)
      , restriction_matrix(other.restriction_matrix)
    {}

    void
    prolongate_and_add(BlockVectorType &dst, const BlockVectorType &src) const
    {
      AssertDimension(prolongation_matrix.n(), src.n_blocks());
      AssertDimension(prolongation_matrix.m(), dst.n_blocks());
      tensorproduct_add(dst, prolongation_matrix, src);
    }

    void
    restrict_and_add(BlockVectorType &dst, const BlockVectorType &src) const
    {
      AssertDimension(restriction_matrix.n(), src.n_blocks());
      AssertDimension(restriction_matrix.m(), dst.n_blocks());
      tensorproduct_add(dst, restriction_matrix, src);
    }
  };

  template <int dim, typename Number>
  class TwoLevelTransferOperator
  {
    using VectorType      = VectorT<Number>;
    using BlockVectorType = BlockVectorT<Number>;
    std::variant<MGTwoLevelBlockTransfer<dim, Number>,
                 MGTwoLevelTransferST<Number>>
      transfer_variant;

  public:
    TwoLevelTransferOperator(
      MGTwoLevelTransfer<dim, VectorType> const &transfer)
      : transfer_variant(MGTwoLevelBlockTransfer<dim, Number>(transfer))
    {}
    TwoLevelTransferOperator(MGTwoLevelTransferST<Number> const &transfer)
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
    STMGTransferBlockMatrixFree(
      std::vector<std::shared_ptr<MGTwoLevelTransfer<dim, VectorType>>> const
        &space_transfers_,
      MGLevelObject<TwoLevelTransferOperator<dim, Number>> const &transfers_,
      std::vector<unsigned int>                                   n_blocks_,
      std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>>
        partitioners_)
      : space_transfers(space_transfers_)
      , transfer(transfers_)
      , n_blocks(n_blocks_)
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
      if (dst.n_blocks() != n_blocks.back())
        dst.reinit(n_blocks.back());
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
      AssertDimension(n_blocks.back(), src.n_blocks());
      for (unsigned int level = dst.min_level(); level <= dst.max_level();
           ++level)
        {
          if (dst[level].n_blocks() != n_blocks[level])
            dst[level].reinit(n_blocks[level]);
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
          auto const &partitioner = partitioners[level - transfer.min_level()];
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

    bool same_for_all = true;
    std::vector<std::shared_ptr<MGTwoLevelTransfer<dim, VectorType>>>
                                                         space_transfers;
    MGLevelObject<TwoLevelTransferOperator<dim, Number>> transfer;
    std::vector<unsigned int>                            n_blocks;
    std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>>
      partitioners;
  };

  template <int dim, typename Number>
  auto
  build_stmg_transfers(
    TimeStepType const type,
    unsigned int       r,
    unsigned int       n_timesteps_at_once,
    const MGLevelObject<std::shared_ptr<const DoFHandler<dim>>>
      &mg_dof_handlers,
    const MGLevelObject<std::shared_ptr<const AffineConstraints<Number>>>
      &mg_constraints,
    const std::function<void(const unsigned int, VectorT<Number> &)>
                                  &initialize_dof_vector,
    const bool                     restrict_is_transpose_prolongate,
    const std::vector<TimeMGType> &mg_type_level)
  {
    using BlockVectorType        = BlockVectorT<Number>;
    using VectorType             = VectorT<Number>;
    unsigned int const min_level = mg_dof_handlers.min_level();
    unsigned int const max_level = mg_dof_handlers.max_level();
    unsigned int const n_levels  = mg_dof_handlers.n_levels();
    AssertDimension(n_levels - 1, mg_type_level.size());
    MGLevelObject<TwoLevelTransferOperator<dim, Number>> transfer(min_level,
                                                                  max_level);
    std::vector<unsigned int>                            n_blocks(n_levels);
    std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>>
      partitioners(n_levels);
    std::vector<std::shared_ptr<MGTwoLevelTransfer<dim, VectorType>>>
      space_transfers(n_levels);

    unsigned int n_k_levels = 0, n_tau_levels = 0;
    for (auto const &el : mg_type_level)
      if (el == TimeMGType::k)
        ++n_k_levels;
      else if (el == TimeMGType::tau)
        ++n_tau_levels;

    Assert((type == TimeStepType::DG ? r + 1 : r >= n_k_levels),
           ExcLowerRange(r, n_k_levels));
    unsigned int n_dofs_intvl   = (type == TimeStepType::DG) ? r + 1 : r;
    unsigned int n_blocks_level = n_dofs_intvl * n_timesteps_at_once;
    unsigned int i              = n_levels - 1;
    n_blocks[i]                 = n_blocks_level;

    for (unsigned int l = min_level; l <= max_level; ++l)
      {
        VectorType vector;
        initialize_dof_vector(l, vector);
        partitioners[l - min_level] = vector.get_partitioner();
      }
    for (auto mgt = mg_type_level.rbegin(); mgt != mg_type_level.rend();
         ++mgt, --i)
      {
        if (*mgt == TimeMGType::none)
          {
            space_transfers[i] =
              std::make_shared<MGTwoLevelTransfer<dim, VectorType>>();
            space_transfers[i]->reinit(*mg_dof_handlers[i],
                                       *mg_dof_handlers[i - 1],
                                       *mg_constraints[i],
                                       *mg_constraints[i - 1]);
            space_transfers[i]->enable_inplace_operations_if_possible(
              partitioners[i - 1], partitioners[i]);
            transfer[i] =
              TwoLevelTransferOperator<dim, Number>(*space_transfers[i]);
          }
        else
          transfer[i] = TwoLevelTransferOperator<dim, Number>(
            MGTwoLevelTransferST<Number>(type,
                                         r,
                                         n_timesteps_at_once,
                                         restrict_is_transpose_prolongate,
                                         *mgt));

        if (*mgt == TimeMGType::k)
          --r, --n_dofs_intvl;
        else if (*mgt == TimeMGType::tau)
          n_timesteps_at_once /= 2;
        n_blocks_level  = n_dofs_intvl * n_timesteps_at_once;
        n_blocks[i - 1] = n_blocks_level;
      }

    return std::make_unique<STMGTransferBlockMatrixFree<dim, Number>>(
      space_transfers, transfer, n_blocks, partitioners);
  }

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
    {
      indices.clear();
      valence.reinit(0);
      blocks.clear();
    }

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
    double       smoothing_range               = 1;
    unsigned int smoothing_degree              = 5;
    unsigned int smoothing_eig_cg_n_iterations = 20;
    unsigned int smoothing_steps               = 1;

    bool estimate_relaxation = true;

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
    bool         do_output               = false;
    bool         print_timing            = false;
    bool         space_time_mg           = true;
    bool         time_before_space       = false;
    TimeStepType type                    = TimeStepType::CGP;
    ProblemType  problem                 = ProblemType::wave;
    unsigned int n_timesteps_at_once     = 1;
    int          n_timesteps_at_once_min = -1;
    unsigned int fe_degree               = 1;
    int          fe_degree_min           = -1;
    unsigned int n_deg_cycles            = 1;
    unsigned int n_ref_cycles            = 1;
    double       frequency               = 1.0;
    int          refinement              = 2;
    bool         space_time_conv_test    = true;
    bool         extrapolate             = true;
    std::string  functional_file         = "functionals.txt";
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
      std::string              type_, problem_;
      dealii::ParameterHandler prm;
      prm.add_parameter("doOutput", do_output);
      prm.add_parameter("printTiming", print_timing);
      prm.add_parameter("spaceTimeMg", space_time_mg);
      prm.add_parameter("mgTimeBeforeSpace", time_before_space);
      prm.add_parameter("timeType", type_);
      prm.add_parameter("problemType", problem_);
      prm.add_parameter("nTimestepsAtOnce", n_timesteps_at_once);
      prm.add_parameter("nTimestepsAtOnceMin", n_timesteps_at_once_min);
      prm.add_parameter("feDegree", fe_degree);
      prm.add_parameter("feDegreeMin", fe_degree_min);
      prm.add_parameter("nDegCycles", n_deg_cycles);
      prm.add_parameter("nRefCycles", n_ref_cycles);
      prm.add_parameter("frequency", frequency);
      prm.add_parameter("refinement", refinement);
      prm.add_parameter("spaceTimeConvergenceTest", space_time_conv_test);
      prm.add_parameter("extrapolate", extrapolate);
      prm.add_parameter("functionalFile", functional_file);
      prm.add_parameter("hyperRectLowerLeft", hyperrect_lower_left);
      prm.add_parameter("hyperRectUpperRight", hyperrect_upper_right);
      prm.add_parameter("subdivisions", subdivisions);
      prm.add_parameter("distortGrid", distort_grid);
      prm.add_parameter("distortCoeff", distort_coeff);
      prm.add_parameter("sourcePoint", source);
      prm.add_parameter("endTime", end_time);

      prm.add_parameter("smoothingDegree", mg_data.smoothing_degree);
      prm.add_parameter("smoothingSteps", mg_data.smoothing_steps);
      prm.add_parameter("estimateRelaxation", mg_data.estimate_relaxation);
      prm.add_parameter("coarseGridSmootherType",
                        mg_data.coarse_grid_smoother_type);
      prm.add_parameter("coarseGridMaxiter", mg_data.coarse_grid_maxiter);
      prm.add_parameter("coarseGridAbstol", mg_data.coarse_grid_abstol);
      prm.add_parameter("coarseGridReltol", mg_data.coarse_grid_reltol);
      prm.add_parameter("restrictIsTransposeProlongate",
                        mg_data.restrict_is_transpose_prolongate);
      prm.add_parameter("variable", mg_data.variable);
      std::ifstream file;
      file.open(file_name);
      prm.parse_input_from_json(file, true);
      type    = str_to_time_type.at(type_);
      problem = str_to_problem_type.at(problem_);
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
    }
  };

  template <int dim, typename Number, typename LevelMatrixType>
  class GMG
  {
    using BlockVectorType = BlockVectorT<Number>;
    using VectorType      = VectorT<Number>;

    using MGTransferType = STMGTransferBlockMatrixFree<dim, Number>;

    using SmootherPreconditionerType = PreconditionVanka<Number>;
    using SmootherType =
      PreconditionRelaxation<LevelMatrixType, SmootherPreconditionerType>;
    using MGSmootherType =
      MGSmootherPrecondition<LevelMatrixType, SmootherType, BlockVectorType>;

  public:
    GMG(
      TimerOutput                   &timer,
      Parameters<dim> const         &parameters,
      unsigned int const             r,
      unsigned int const             n_timesteps_at_once,
      const std::vector<TimeMGType> &mg_type_level,
      const DoFHandler<dim>         &dof_handler,
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
      , src_(std::move(tmp1))
      , dst_(std::move(tmp2))
      , dof_handler(dof_handler)
      , mg_dof_handlers(mg_dof_handlers)
      , mg_constraints(mg_constraints)
      , mg_operators(mg_operators)
      , precondition_vanka(mg_smoother_)
      , min_level(mg_dof_handlers.min_level())
      , max_level(mg_dof_handlers.max_level())
    {
      transfer_block = build_stmg_transfers<dim, Number>(
        parameters.type,
        r,
        n_timesteps_at_once,
        mg_dof_handlers,
        mg_constraints,
        [&](const unsigned int l, VectorType &vec) {
          this->mg_operators[l]->initialize_dof_vector(vec);
        },
        additional_data.restrict_is_transpose_prolongate,
        mg_type_level);
    }

    void
    reinit() const
    {
      // wrap level operators
      mg_matrix = mg::Matrix<BlockVectorType>(mg_operators);

      MGLevelObject<typename SmootherType::AdditionalData> smoother_data(
        min_level, max_level);

      // setup smoothers on each level
      for (unsigned int level = min_level; level <= max_level; ++level)
        {
          smoother_data[level].preconditioner = precondition_vanka[level];
          smoother_data[level].n_iterations   = additional_data.smoothing_steps;
          smoother_data[level].relaxation =
            additional_data.estimate_relaxation ? estimate_relaxation(level) :
                                                  1.0;
        }
      mg_smoother = std::make_unique<MGSmootherType>(1,
                                                     additional_data.variable,
                                                     false,
                                                     false);
      mg_smoother->initialize(mg_operators, smoother_data);

      if (additional_data.coarse_grid_smoother_type != "Smoother")
        {
          solver_control_coarse = std::make_unique<ReductionControl>(
            additional_data.coarse_grid_maxiter,
            additional_data.coarse_grid_abstol,
            additional_data.coarse_grid_reltol,
            false,
            false);
          typename SolverGMRES<BlockVectorType>::AdditionalData const
            gmres_additional_data(10);
          gmres_coarse = std::make_unique<SolverGMRES<BlockVectorType>>(
            *solver_control_coarse, gmres_additional_data);

          auto diagonal_matrix =
            mg_operators[min_level]->get_matrix_diagonal_inverse();

          typename PreconditionRelaxation<
            LevelMatrixType,
            DiagonalMatrix<BlockVectorType>>::AdditionalData coarse_precon_data;
          coarse_precon_data.relaxation     = 0.9;
          coarse_precon_data.n_iterations   = 1;
          coarse_precon_data.preconditioner = diagonal_matrix;

          preconditioner_coarse = std::make_unique<
            PreconditionRelaxation<LevelMatrixType,
                                   DiagonalMatrix<BlockVectorType>>>();
          preconditioner_coarse->initialize(*(mg_operators[min_level]),
                                            coarse_precon_data);

          mg_coarse = std::make_unique<dealii::MGCoarseGridIterativeSolver<
            BlockVectorType,
            SolverGMRES<BlockVectorType>,
            LevelMatrixType,
            PreconditionRelaxation<LevelMatrixType,
                                   DiagonalMatrix<BlockVectorType>>>>(
            *gmres_coarse, *mg_operators[min_level], *preconditioner_coarse);
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

    PreconditionerGMGAdditionalData additional_data;

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

    std::unique_ptr<MGTransferType> transfer_block;

    mutable mg::Matrix<BlockVectorType> mg_matrix;


    mutable std::unique_ptr<MGSmootherType> mg_smoother;

    mutable std::unique_ptr<MGCoarseGridBase<BlockVectorType>> mg_coarse;
    mutable std::unique_ptr<SolverControl>                solver_control_coarse;
    mutable std::unique_ptr<SolverGMRES<BlockVectorType>> gmres_coarse;
    mutable std::unique_ptr<
      PreconditionRelaxation<LevelMatrixType, DiagonalMatrix<BlockVectorType>>>
      preconditioner_coarse;

    mutable std::unique_ptr<Multigrid<BlockVectorType>> mg;

    mutable std::unique_ptr<
      PreconditionMG<dim, BlockVectorType, MGTransferType>>
      preconditioner;
  };
} // namespace dealii
