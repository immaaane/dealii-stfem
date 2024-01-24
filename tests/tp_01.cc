#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/repartitioning_policy_tools.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_matrix_tools.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/vector_tools.h>

#include "include/exact_solution.h"
#include "include/fe_time.h"

using namespace dealii;
using dealii::numbers::PI;

using Number              = double;
using VectorType          = LinearAlgebra::distributed::Vector<double>;
using BlockVectorType     = LinearAlgebra::distributed::BlockVector<double>;
using SparseMatrixType    = TrilinosWrappers::SparseMatrix;
using SparsityPatternType = TrilinosWrappers::SparsityPattern;

template <typename Number, typename SystemMatrixType>
class SystemMatrix
{
public:
  SystemMatrix(const SystemMatrixType   &K,
               const SystemMatrixType   &M,
               const FullMatrix<Number> &Alpha_,
               const FullMatrix<Number> &Beta_)
    : K(K)
    , M(M)
    , Alpha(Alpha_)
    , Beta(Beta_)
    , alpha_is_zero(Alpha.all_zero())
    , beta_is_zero(Beta.all_zero())
  {}

  void
  vmult(BlockVectorType &dst, const BlockVectorType &src) const
  {
    const unsigned int n_blocks = src.n_blocks();

    dst = 0.0;
    VectorType tmp;
    K.initialize_dof_vector(tmp);
    for (unsigned int i = 0; i < n_blocks; ++i)
      {
        K.vmult(tmp, src.block(i));

        for (unsigned int j = 0; j < n_blocks; ++j)
          if (Alpha(j, i) != 0.0)
            dst.block(j).add(Alpha(j, i), tmp);
      }

    M.initialize_dof_vector(tmp);
    for (unsigned int i = 0; i < n_blocks; ++i)
      {
        M.vmult(tmp, src.block(i));

        for (unsigned int j = 0; j < n_blocks; ++j)
          if (Beta(j, i) != 0.0)
            dst.block(j).add(Beta(j, i), tmp);
      }
  }

  void
  Tvmult(BlockVectorType &dst, const BlockVectorType &src) const
  {
    const unsigned int n_blocks = src.n_blocks();

    dst = 0.0;
    VectorType tmp;
    K.initialize_dof_vector(tmp);
    for (unsigned int i = 0; i < n_blocks; ++i)
      {
        K.vmult(tmp, src.block(i));

        for (unsigned int j = 0; j < n_blocks; ++j)
          if (Alpha(i, j) != 0.0)
            dst.block(j).add(Alpha(i, j), tmp);
      }

    M.initialize_dof_vector(tmp);
    for (unsigned int i = 0; i < n_blocks; ++i)
      {
        M.vmult(tmp, src.block(i));

        for (unsigned int j = 0; j < n_blocks; ++j)
          if (Beta(i, j) != 0.0)
            dst.block(j).add(Beta(i, j), tmp);
      }
  }

  // Specialization for a nx1 matrix. Useful for rhs assembly
  void
  vmult(BlockVectorType &dst, const VectorType &src) const
  {
    const unsigned int n_blocks = dst.n_blocks();

    VectorType tmp;
    if (!alpha_is_zero)
      {
        K.initialize_dof_vector(tmp);
        K.vmult(tmp, src);
        for (unsigned int j = 0; j < n_blocks; ++j)
          if (Alpha(j, 0) != 0.0)
            dst.block(j).equ(Alpha(j, 0), tmp);
      }

    if (!beta_is_zero)
      {
        M.initialize_dof_vector(tmp);
        M.vmult(tmp, src);
        for (unsigned int j = 0; j < n_blocks; ++j)
          if (Beta(j, 0) != 0.0)
            dst.block(j).add(Beta(j, 0), tmp);
      }
  }

  void
  initialize_dof_vector(VectorType &vec) const
  {
    K.initialize_dof_vector(vec);
  }

  void
  initialize_dof_vector(BlockVectorType &vec) const
  {
    vec.reinit(Alpha.m());
    for (unsigned int i = 0; i < vec.n_blocks(); ++i)
      this->initialize_dof_vector(vec.block(i));
  }

private:
  const SystemMatrixType   &K;
  const SystemMatrixType   &M;
  const FullMatrix<Number> &Alpha;
  const FullMatrix<Number> &Beta;

  // Only used for nx1: small optimization to avoid unnecessary vmult
  bool alpha_is_zero;
  bool beta_is_zero;
};

template <typename Number>
class VankaSmoother : public MGSmoother<BlockVectorType>
{
public:
  struct AdditionalData
  {};
  template <int dim>
  VankaSmoother(
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
  vmult(unsigned int l, BlockVectorType &dst, const BlockVectorType &src) const
  {
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
  GMG(const DoFHandler<dim> &dof_handler,
      const MGLevelObject<std::shared_ptr<const DoFHandler<dim>>>
        &mg_dof_handlers,
      const MGLevelObject<std::shared_ptr<const AffineConstraints<Number>>>
        &mg_constraints,
      const MGLevelObject<std::shared_ptr<const LevelMatrixType>> &mg_operators,
      const std::shared_ptr<MGSmootherBase<BlockVectorType>>      &mg_smoother_)
    : dof_handler(dof_handler)
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
        const auto coarse_comm = mg_dof_handlers[min_level]->get_communicator();
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
  // using SmootherPreconditionerType = DiagonalMatrix<VectorType>;
  const DoFHandler<dim>                                      &dof_handler;
  const MGLevelObject<std::shared_ptr<const DoFHandler<dim>>> mg_dof_handlers;
  const MGLevelObject<std::shared_ptr<const AffineConstraints<Number>>>
                                                              mg_constraints;
  const MGLevelObject<std::shared_ptr<const LevelMatrixType>> mg_operators;

  const unsigned int min_level;
  const unsigned int max_level;

  MGLevelObject<MGTwoLevelTransfer<dim, VectorType>>           transfers;
  std::unique_ptr<MGTransferGlobalCoarsening<dim, VectorType>> transfer_scalar;
  std::unique_ptr<MGTransferType>                              transfer_block;

  mutable mg::Matrix<BlockVectorType>                      mg_matrix;
  mutable std::shared_ptr<MGSmootherBase<BlockVectorType>> mg_smoother;

  mutable std::unique_ptr<MGCoarseGridBase<BlockVectorType>> mg_coarse;
  mutable std::unique_ptr<Multigrid<BlockVectorType>>        mg;
  mutable std::unique_ptr<PreconditionMG<dim, BlockVectorType, MGTransferType>>
    preconditioner;
};

template <int dim, typename Number>
class MatrixFreeOperator
{
public:
  MatrixFreeOperator(const Mapping<dim>              &mapping,
                     const DoFHandler<dim>           &dof_handler,
                     const AffineConstraints<Number> &constraints,
                     const Quadrature<dim>           &quadrature,
                     const double                     mass_matrix_scaling,
                     const double                     laplace_matrix_scaling)
    : mass_matrix_scaling(mass_matrix_scaling)
    , laplace_matrix_scaling(laplace_matrix_scaling)
  {
    typename MatrixFree<dim, Number>::AdditionalData additional_data;
    additional_data.mapping_update_flags = update_values | update_gradients;

    matrix_free.reinit(
      mapping, dof_handler, constraints, quadrature, additional_data);
  }

  void
  initialize_dof_vector(VectorType &vec) const
  {
    matrix_free.initialize_dof_vector(vec);
  }

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    matrix_free.cell_loop(
      &MatrixFreeOperator::do_cell_integral_range, this, dst, src, true);
  }

  void
  compute_system_matrix(SparseMatrixType &sparse_matrix) const
  {
    MatrixFreeTools::compute_matrix(matrix_free,
                                    matrix_free.get_affine_constraints(),
                                    sparse_matrix,
                                    &MatrixFreeOperator::do_cell_integral_local,
                                    this);
  }

private:
  using FECellIntegrator = FEEvaluation<dim, -1, 0, 1, Number>;

  void
  do_cell_integral_range(
    const MatrixFree<dim, Number>               &matrix_free,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &range) const
  {
    FECellIntegrator integrator(matrix_free);

    for (unsigned int cell = range.first; cell < range.second; ++cell)
      {
        integrator.reinit(cell);

        // gather
        integrator.read_dof_values(src);

        do_cell_integral_local(integrator);

        // scatter
        integrator.distribute_local_to_global(dst);
      }
  }

  void
  do_cell_integral_local(FECellIntegrator &integrator) const
  {
    // evaluate
    if (mass_matrix_scaling != 0.0 && laplace_matrix_scaling != 0.0)
      integrator.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
    else if (mass_matrix_scaling != 0.0)
      integrator.evaluate(EvaluationFlags::values);
    else if (laplace_matrix_scaling != 0.0)
      integrator.evaluate(EvaluationFlags::gradients);

    // quadrature
    for (unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        if (mass_matrix_scaling != 0.0)
          integrator.submit_value(mass_matrix_scaling * integrator.get_value(q),
                                  q);
        if (laplace_matrix_scaling != 0.0)
          integrator.submit_gradient(laplace_matrix_scaling *
                                       integrator.get_gradient(q),
                                     q);
      }

    // integrate
    if (mass_matrix_scaling != 0.0 && laplace_matrix_scaling != 0.0)
      integrator.integrate(EvaluationFlags::values |
                           EvaluationFlags::gradients);
    else if (mass_matrix_scaling != 0.0)
      integrator.integrate(EvaluationFlags::values);
    else if (laplace_matrix_scaling != 0.0)
      integrator.integrate(EvaluationFlags::gradients);
  }

  MatrixFree<dim, Number> matrix_free;

  double mass_matrix_scaling;
  double laplace_matrix_scaling;
};

/** Time stepping by DG and CGP variational time discretizations
 *
 * This time integrator is suited for linear problems. For nonlinear problems we
 * would need a few extensions in order to integrate nonlinear terms accurately.
 */
template <int dim, typename Number>
class TimeIntegrator
{
public:
  TimeIntegrator(
    TimeStepType              type_,
    unsigned int              time_degree_,
    FullMatrix<Number> const &Alpha_,
    FullMatrix<Number> const &Beta_,
    FullMatrix<Number> const &Gamma_,
    FullMatrix<Number> const &Zeta_,
    double const              gmres_tolerance_,
    SystemMatrix<Number, MatrixFreeOperator<dim, Number>> const &matrix_,
    GMG<dim, SystemMatrix<Number, MatrixFreeOperator<dim, Number>>> const
      &preconditioner_,
    SystemMatrix<Number, MatrixFreeOperator<dim, Number>> const &rhs_matrix_,
    std::function<void(const double, VectorType &)> integrate_rhs_function)
    : type(type_)
    , time_degree(time_degree_)
    , Alpha(Alpha_)
    , Beta(Beta_)
    , Zeta(Zeta_)
    , Gamma(Gamma_)
    , solver_control(1000, 1.e-16, gmres_tolerance_, true, true)
    , solver(solver_control,
             dealii::SolverFGMRES<BlockVectorType>::AdditionalData{
               static_cast<unsigned int>(50)})
    , preconditioner(preconditioner_)
    , matrix(matrix_)
    , rhs_matrix(rhs_matrix_)
    , integrate_rhs_function(integrate_rhs_function)
  {
    if (type == TimeStepType::DG)
      quad_time =
        QGaussRadau<1>(time_degree + 1, QGaussRadau<1>::EndPoint::right);
    else if (type == TimeStepType::CGP)
      quad_time = QGaussLobatto<1>(time_degree + 1);
  }

  void
  solve(BlockVectorType                    &x,
        VectorType const                   &prev_x,
        [[maybe_unused]] const unsigned int timestep_number,
        const double                        time,
        const double                        time_step) const
  {
    BlockVectorType rhs(x.n_blocks());
    for (unsigned int j = 0; j < rhs.n_blocks(); ++j)
      {
        matrix.initialize_dof_vector(rhs.block(j));
      }
    rhs_matrix.vmult(rhs, prev_x);

    assemble_force(rhs, time, time_step);

    // constant extrapolation of solution from last time
    for (unsigned int j = 0; j < rhs.n_blocks(); ++j)
      {
        x.block(j) = prev_x;
      }
    try
      {
        solver.solve(matrix, x, rhs, preconditioner);
      }
    catch (const SolverControl::NoConvergence &e)
      {
        AssertThrow(false, ExcMessage(e.what()));
      }
  }
  void
  assemble_force(BlockVectorType &rhs,
                 double const     time,
                 double const     time_step) const
  {
    VectorType tmp;
    matrix.initialize_dof_vector(tmp);

    for (unsigned int j = 0; j < quad_time.size(); ++j)
      {
        double time_ = time + time_step * quad_time.point(j)[0];
        integrate_rhs_function(time_, tmp);
        for (unsigned int i = 0; i < rhs.n_blocks(); ++i)
          {
            if (type == TimeStepType::DG)
              {
                if (i == j)
                  rhs.block(i).add(Alpha(i, j), tmp);
              }
            else if (type == TimeStepType::CGP)
              {
                if (j == 0)
                  {
                    rhs.block(i).add(-Gamma(i, 0), tmp);
                  }
                else
                  {
                    if (i == j - 1)
                      rhs.block(i).add(Alpha(i, j - 1), tmp);
                  }
              }
          }
      }
  }

private:
  TimeStepType              type;
  unsigned int              time_degree;
  Quadrature<1>             quad_time;
  FullMatrix<Number> const &Alpha;
  FullMatrix<Number> const &Beta;
  FullMatrix<Number> const &Zeta;
  FullMatrix<Number> const &Gamma;

  mutable ReductionControl              solver_control;
  mutable SolverFGMRES<BlockVectorType> solver;
  GMG<dim, SystemMatrix<Number, MatrixFreeOperator<dim, Number>>> const
                                                              &preconditioner;
  SystemMatrix<Number, MatrixFreeOperator<dim, Number>> const &matrix;
  SystemMatrix<Number, MatrixFreeOperator<dim, Number>> const &rhs_matrix;
  std::function<void(const double, VectorType &)> integrate_rhs_function;
};

template <int dim>
void
test(dealii::ConditionalOStream const &pcout,
     MPI_Comm const                    comm_global,
     TimeStepType                      type)
{
  ConvergenceTable table;
  MappingQ1<dim>   mapping;


  auto convergence_test = [&](int const          refinement,
                              unsigned int const fe_degree,
                              bool               do_output) {
    const unsigned int n_blocks =
      type == TimeStepType::DG ? fe_degree + 1 : fe_degree;
    auto const  basis = get_time_basis<Number>(type, fe_degree);
    FE_Q<dim>   fe(fe_degree + 1);
    QGauss<dim> quad(fe.tensor_degree() + 1);

    parallel::distributed::Triangulation<dim> tria(comm_global);
    DoFHandler<dim>                           dof_handler(tria);

    GridGenerator::hyper_cube(tria);
    tria.refine_global(refinement);

    dof_handler.distribute_dofs(fe);

    AffineConstraints<Number> constraints;
    IndexSet                  locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    DoFTools::make_zero_boundary_constraints(dof_handler, constraints);
    constraints.close();
    pcout << ":: Number of active cells: " << tria.n_active_cells() << "\n"
          << ":: Number of degrees of freedom: " << dof_handler.n_dofs() << "\n"
          << std::endl;

    // create sparsity pattern
    SparsityPatternType sparsity_pattern(dof_handler.locally_owned_dofs(),
                                         dof_handler.locally_owned_dofs(),
                                         dof_handler.get_communicator());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    sparsity_pattern,
                                    constraints,
                                    false);
    sparsity_pattern.compress();

    // create scalar siffness matrix
    SparseMatrixType K;
    K.reinit(sparsity_pattern);

    // create scalar mass matrix
    SparseMatrixType M;
    M.reinit(sparsity_pattern);

    double time           = 0.;
    double time_step_size = 1.0 * pow(2.0, -(refinement + 1));
    double end_time       = 1.;

    // matrix-free operators
    MatrixFreeOperator<dim, Number> K_mf(
      mapping, dof_handler, constraints, quad, 0.0, 1.0);
    MatrixFreeOperator<dim, Number> M_mf(
      mapping, dof_handler, constraints, quad, 1.0, 0.0);

    if (false)
      {
        MatrixCreator::create_laplace_matrix<dim, dim>(
          mapping, dof_handler, quad, K, nullptr, constraints);
        MatrixCreator::create_mass_matrix<dim, dim>(
          mapping, dof_handler, quad, M, nullptr, constraints);
      }
    else
      {
        K_mf.compute_system_matrix(K);
        M_mf.compute_system_matrix(M);
      }

    auto [Alpha, Beta, Gamma, Zeta] =
      get_fe_time_weights<Number>(type, fe_degree);
    Alpha *= time_step_size;
    if (type == TimeStepType::CGP)
      Gamma *= time_step_size;

    SystemMatrix<Number, MatrixFreeOperator<dim, Number>> matrix(K_mf,
                                                                 M_mf,
                                                                 Alpha,
                                                                 Beta);
    SystemMatrix<Number, MatrixFreeOperator<dim, Number>> rhs_matrix(
      K_mf,
      M_mf,
      (type == TimeStepType::CGP) ? Gamma : Zeta,
      (type == TimeStepType::CGP) ? Zeta : Gamma);

    /// GMG
    RepartitioningPolicyTools::DefaultPolicy<dim>          policy(true);
    std::vector<std::shared_ptr<const Triangulation<dim>>> mg_triangulations =
      MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(
        tria, policy);
    const unsigned int min_level = 0;
    const unsigned int max_level = mg_triangulations.size() - 1;
    pcout << ":: Min Level " << min_level << "  Max Level " << max_level
          << std::endl;
    MGLevelObject<std::shared_ptr<const DoFHandler<dim>>> mg_dof_handlers(
      min_level, max_level);
    MGLevelObject<std::shared_ptr<const SparsityPatternType>>
      mg_sparsity_patterns(min_level, max_level);
    MGLevelObject<std::shared_ptr<const SparseMatrixType>> mg_M(min_level,
                                                                max_level);
    MGLevelObject<std::shared_ptr<const SparseMatrixType>> mg_K(min_level,
                                                                max_level);
    MGLevelObject<std::shared_ptr<const MatrixFreeOperator<dim, Number>>>
      mg_M_mf(min_level, max_level);
    MGLevelObject<std::shared_ptr<const MatrixFreeOperator<dim, Number>>>
      mg_K_mf(min_level, max_level);
    MGLevelObject<std::shared_ptr<const AffineConstraints<double>>>
      mg_constraints(min_level, max_level);
    MGLevelObject<std::shared_ptr<
      const SystemMatrix<Number, MatrixFreeOperator<dim, Number>>>>
      mg_operators(min_level, max_level);

    for (unsigned int l = min_level; l <= max_level; ++l)
      {
        auto dof_handler_ =
          std::make_shared<DoFHandler<dim>>(*mg_triangulations[l]);
        auto constraints_ = std::make_shared<AffineConstraints<double>>();
        dof_handler_->distribute_dofs(fe);

        IndexSet locally_relevant_dofs;
        DoFTools::extract_locally_relevant_dofs(*dof_handler_,
                                                locally_relevant_dofs);
        constraints_->reinit(locally_relevant_dofs);
        DoFTools::make_zero_boundary_constraints(*dof_handler_,
                                                 0,
                                                 *constraints_);
        constraints_->close();

        // matrix-free operators
        auto K_mf_ = std::make_shared<MatrixFreeOperator<dim, Number>>(
          mapping, *dof_handler_, *constraints_, quad, 0.0, 1.0);
        auto M_mf_ = std::make_shared<MatrixFreeOperator<dim, Number>>(
          mapping, *dof_handler_, *constraints_, quad, 1.0, 0.0);

        mg_operators[l] = std::make_shared<
          SystemMatrix<Number, MatrixFreeOperator<dim, Number>>>(*K_mf_,
                                                                 *M_mf_,
                                                                 Alpha,
                                                                 Beta);

        auto sparsity_pattern_ = std::make_shared<SparsityPatternType>(
          dof_handler_->locally_owned_dofs(),
          dof_handler_->locally_owned_dofs(),
          dof_handler_->get_communicator());
        DoFTools::make_sparsity_pattern(*dof_handler_,
                                        *sparsity_pattern_,
                                        *constraints_,
                                        false);
        sparsity_pattern_->compress();

        auto K_ = std::make_shared<SparseMatrixType>();
        K_->reinit(*sparsity_pattern_);
        auto M_ = std::make_shared<SparseMatrixType>();
        M_->reinit(*sparsity_pattern_);
        K_mf_->compute_system_matrix(*K_);
        M_mf_->compute_system_matrix(*M_);

        // matrix->attach(*mg_operators[l]);
        mg_sparsity_patterns[l] = sparsity_pattern_;
        mg_M_mf[l]              = M_mf_;
        mg_K_mf[l]              = K_mf_;
        mg_M[l]                 = M_;
        mg_K[l]                 = K_;
        mg_dof_handlers[l]      = dof_handler_;
        mg_constraints[l]       = constraints_;
      }

    std::shared_ptr<MGSmoother<BlockVectorType>> smoother =
      std::make_shared<VankaSmoother<Number>>(
        mg_K, mg_M, mg_sparsity_patterns, Alpha, Beta, mg_dof_handlers);
    // std::shared_ptr<MGSmootherBase<BlockVectorType>> smoother =
    //   std::make_shared<MGSmootherIdentity<BlockVectorType>>();
    auto preconditioner = std::make_unique<
      GMG<dim, SystemMatrix<Number, MatrixFreeOperator<dim, Number>>>>(
      dof_handler, mg_dof_handlers, mg_constraints, mg_operators, smoother);
    preconditioner->reinit();
    //
    /// GMG

    // Number                      frequency = 1.0;
    RHSFunction<dim, Number>   rhs_function;
    ExactSolution<dim, Number> exact_solution;

    auto integrate_rhs_function =
      [&mapping, &dof_handler, &quad, &rhs_function, &constraints](
        const double time, VectorType &rhs) -> void {
      rhs_function.set_time(time);
      rhs = 0.0;
      VectorTools::create_right_hand_side(
        mapping, dof_handler, quad, rhs_function, rhs, constraints);
    };
    [[maybe_unused]] auto evaluate_exact_solution =
      [&mapping, &dof_handler, &exact_solution](const double time,
                                                VectorType  &tmp) -> void {
      exact_solution.set_time(time);
      VectorTools::interpolate(mapping, dof_handler, exact_solution, tmp);
    };
    auto evaluate_numerical_solution =
      [&constraints, &basis, &type](const double           time,
                                    VectorType            &tmp,
                                    BlockVectorType const &x,
                                    VectorType const      &prev_x) -> void {
      int i = 0;
      tmp   = 0.0;
      for (auto const &el : basis)
        {
          if (double v = el.value(time); v != 0.0)
            {
              if (type == TimeStepType::DG)
                tmp.add(v, x.block(i));
              else
                tmp.add(v, (i == 0) ? prev_x : x.block(i - 1));
            }
          ++i;
        }
      constraints.distribute(tmp);
    };

    BlockVectorType x(n_blocks);
    for (unsigned int i = 0; i < n_blocks; ++i)
      matrix.initialize_dof_vector(x.block(i));
    VectorType prev_x;
    matrix.initialize_dof_vector(prev_x);
    VectorType exact;
    matrix.initialize_dof_vector(exact);
    VectorType numeric;
    matrix.initialize_dof_vector(numeric);

    unsigned int                 timestep_number = 0;
    ErrorCalculator<dim, Number> error_calculator(type,
                                                  fe_degree,
                                                  fe_degree,
                                                  mapping,
                                                  dof_handler,
                                                  exact_solution,
                                                  evaluate_numerical_solution);

    TimeIntegrator<dim, Number> step(type,
                                     fe_degree,
                                     Alpha,
                                     Beta,
                                     Gamma,
                                     Zeta,
                                     1.e-12,
                                     matrix,
                                     *preconditioner,
                                     rhs_matrix,
                                     integrate_rhs_function);
    // interpolate initial value
    evaluate_exact_solution(0, x.block(x.n_blocks() - 1));
    double l2 = 0., l8 = -1., h1_semi = 0.;
    while (time < end_time)
      {
        ++timestep_number;
        dealii::deallog << "Step " << timestep_number << " t = " << time
                        << std::endl;
        prev_x = x.block(x.n_blocks() - 1);
        step.solve(x, prev_x, timestep_number, time, time_step_size);
        for (unsigned int i = 0; i < n_blocks; ++i)
          constraints.distribute(x.block(i));

        auto error_on_In =
          error_calculator.evaluate_error(time, time_step_size, x, prev_x);
        time += time_step_size;

        l2 += error_on_In[VectorTools::L2_norm];
        l8 = std::max(error_on_In[VectorTools::Linfty_norm], l8);
        h1_semi += error_on_In[VectorTools::H1_seminorm];
        if (do_output)
          {
            numeric = 0.0;
            evaluate_numerical_solution(1.0, numeric, x, prev_x);
            DataOut<dim> data_out;
            data_out.attach_dof_handler(dof_handler);
            data_out.add_data_vector(numeric, "solution");
            data_out.build_patches();

            std::ofstream output("solution." +
                                 Utilities::int_to_string(timestep_number, 4) +
                                 ".vtu");
            data_out.write_vtu(output);
          }
        if (do_output)
          {
            exact = 0.0;
            evaluate_exact_solution(time, exact);
            DataOut<dim> data_out;
            data_out.attach_dof_handler(dof_handler);
            data_out.add_data_vector(exact, "solution");
            data_out.build_patches();
            std::ofstream output(
              "exact." + Utilities::int_to_string(timestep_number, 4) + ".vtu");
            data_out.write_vtu(output);
          }
      }

    unsigned int const n_active_cells = tria.n_active_cells();
    unsigned int const n_dofs         = dof_handler.n_dofs();
    table.add_value("cells", n_active_cells);
    table.add_value("s-dofs", n_dofs);
    table.add_value("t-dofs", n_blocks);
    table.add_value("st-dofs", n_dofs * n_blocks);
    table.add_value("L\u221E-L\u221E", l8);
    table.add_value("L2-L2", std::sqrt(l2));
    table.add_value("L2-H1_semi", std::sqrt(h1_semi));
  };
  for (int j = 0; j < 3; ++j)
    {
      for (int i = 2; i < 6; ++i)
        {
          convergence_test(i,
                           type == TimeStepType::DG ? j : j + 1,
                           j == 2 && i == 4);
        }
      table.set_precision("L\u221E-L\u221E", 5);
      table.set_precision("L2-L2", 5);
      table.set_precision("L2-H1_semi", 5);
      table.set_scientific("L\u221E-L\u221E", true);
      table.set_scientific("L2-L2", true);
      table.set_scientific("L2-H1_semi", true);
      table.evaluate_convergence_rates("L\u221E-L\u221E",
                                       ConvergenceTable::reduction_rate_log2);
      table.evaluate_convergence_rates("L2-L2",
                                       ConvergenceTable::reduction_rate_log2);
      table.evaluate_convergence_rates("L2-H1_semi",
                                       ConvergenceTable::reduction_rate_log2);
      if (pcout.is_active())
        table.write_text(pcout.get_stream());
      pcout << std::endl;
      table.clear();
    }
}



int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  dealii::ConditionalOStream       pcout(
    std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  std::string filename =
    "proc" +
    std::to_string(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)) +
    ".log";
  std::ofstream pout(filename);
  dealii::deallog.attach(pout);
  dealii::deallog.depth_console(0);
  MPI_Comm  comm = MPI_COMM_WORLD;
  const int dim  = 2;

  test<dim>(pcout, comm, TimeStepType::DG);
  test<dim>(pcout, comm, TimeStepType::CGP);
  dealii::deallog << std::endl;
  pcout << std::endl;
}
