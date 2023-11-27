#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_matrix_tools.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/vector_tools.h>

#include "fe_time.h"

using namespace dealii;
using dealii::numbers::PI;

template <int dim, typename Number>
class ExactSolution : public Function<dim, Number>
{
public:
  ExactSolution(double f_ = 1.0)
    : Function<dim, Number>()
    , f(f_)
  {}

  double
  value(Point<dim> const &x, unsigned int const) const override final
  {
    Number value = sin(2 * PI * f * this->get_time());
    for (unsigned int i = 0; i < dim; ++i)
      value *= sin(2 * PI * f * x[i]);
    return value;
  }
  Tensor<1, dim>
  gradient(const Point<dim> &x, const unsigned int) const override final
  {
    Tensor<1, dim> grad;
    for (unsigned int i = 0; i < dim; ++i)
      {
        grad[i] = 2 * PI * f * sin(2 * PI * f * this->get_time());
        for (unsigned int j = 0; j < dim; ++j)
          grad[i] *= (i == j ? cos(2 * PI * f * x[j]) : sin(2 * PI * f * x[j]));
      }
    return grad;
  }

private:
  double const f;
};

template <int dim, typename Number>
class RHSFunction : public Function<dim, Number>
{
public:
  RHSFunction(double f_ = 1.0)
    : Function<dim, Number>()
    , f(f_)
  {}

  double
  value(Point<dim> const &x, unsigned int const) const override final
  {
    Number value = 2 * PI * f * cos(2 * PI * f * this->get_time());
    for (unsigned int i = 0; i < dim; ++i)
      value *= sin(2 * PI * f * x[i]);
    return value;
  }

private:
  double const f;
};

template <typename Number, typename SystemMatrixType>
class SystemMatrix
{
public:
  using VectorType      = Vector<Number>;
  using BlockVectorType = BlockVector<Number>;

  SystemMatrix(const SystemMatrixType   &K,
               const SystemMatrixType   &M,
               const FullMatrix<Number> &Alpha_,
               const FullMatrix<Number> &Beta_)
    : K(K)
    , M(M)
    , Alpha(Alpha_)
    , Beta(Beta_)
  {}

  void
  vmult(BlockVectorType &dst, const BlockVectorType &src) const
  {
    const unsigned int n_blocks = src.n_blocks();

    VectorType tmp;
    tmp.reinit(src.block(0));
    for (unsigned int i = 0; i < n_blocks; ++i)
      {
        K.vmult(tmp, src.block(i));

        for (unsigned int j = 0; j < n_blocks; ++j)
          dst.block(j).add(Alpha(j, i), tmp);
      }

    for (unsigned int i = 0; i < n_blocks; ++i)
      {
        M.vmult(tmp, src.block(i));

        for (unsigned int j = 0; j < n_blocks; ++j)
          dst.block(j).add(Beta(j, i), tmp);
      }
  }

private:
  const SystemMatrixType   &K;
  const SystemMatrixType   &M;
  const FullMatrix<Number> &Alpha;
  const FullMatrix<Number> &Beta;
};



template <typename Number>
class Preconditioner
{
public:
  using VectorType      = Vector<Number>;
  using BlockVectorType = BlockVector<Number>;

  template <int dim>
  Preconditioner(const SparseMatrix<Number> &K,
                 const SparseMatrix<Number> &M,
                 const FullMatrix<Number>   &Alpha,
                 const FullMatrix<Number>   &Beta,
                 const DoFHandler<dim>      &dof_handler)
    : valence(dof_handler.n_dofs())
  {
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        std::vector<types::global_dof_index> my_indices(
          cell->get_fe().n_dofs_per_cell());
        cell->get_dof_indices(my_indices);
        for (auto const &dof_index : my_indices)
          valence(dof_index) += static_cast<Number>(1);

        indices.emplace_back(my_indices);
      }
    std::vector<FullMatrix<Number>> K_blocks, M_blocks;

    SparseMatrixTools::restrict_to_full_matrices(K,
                                                 K.get_sparsity_pattern(),
                                                 indices,
                                                 K_blocks);
    SparseMatrixTools::restrict_to_full_matrices(M,
                                                 M.get_sparsity_pattern(),
                                                 indices,
                                                 M_blocks);

    blocks.resize(K_blocks.size());

    for (unsigned int i = 0; i < blocks.size(); ++i)
      {
        const auto &K = K_blocks[i];
        const auto &M = M_blocks[i];
        auto       &B = blocks[i];

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

  void
  vmult(BlockVectorType &dst, const BlockVectorType &src) const
  {
    dst = 0.0;

    Vector<Number> dst_local;
    Vector<Number> src_local;

    const unsigned int n_blocks = src.n_blocks();

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
            { // TODO: weight before or after for better perf?
              Number const weight =
                static_cast<Number>(1) / valence[indices[i][j]];
              dst.block(b)[indices[i][j]] += weight * dst_local[c];
            }
      }
  }

private:
  std::vector<std::vector<types::global_dof_index>> indices;
  VectorType                                        valence;
  std::vector<FullMatrix<Number>>                   blocks;
};


template <int dim, typename Number>
class MatrixFreeOperator
{
private:
  using FECellIntegrator = FEEvaluation<dim, -1, 0, 1, Number>;

public:
  using VectorType = Vector<Number>;

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
  vmult(VectorType &dst, const VectorType &src) const
  {
    matrix_free.cell_loop(
      &MatrixFreeOperator::do_cell_integral_range, this, dst, src, true);
  }

  void
  compute_system_matrix(SparseMatrix<Number> &sparse_matrix) const
  {
    MatrixFreeTools::compute_matrix(matrix_free,
                                    matrix_free.get_affine_constraints(),
                                    sparse_matrix,
                                    &MatrixFreeOperator::do_cell_integral_local,
                                    this);
  }

private:
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

template <int dim, typename Number>
class ErrorCalculator
{
public:
  using VectorType      = Vector<Number>;
  using BlockVectorType = BlockVector<Number>;

  ErrorCalculator(TimeStepType               type_,
                  unsigned int               time_degree,
                  unsigned int               space_degree,
                  Mapping<dim> const        &mapping_,
                  DoFHandler<dim> const     &dof_handler_,
                  ExactSolution<dim, Number> exact_solution_,
                  std::function<void(const double, VectorType &)>
                    evaluate_numerical_solution_)
    : time_step_type(type_)
    , quad_cell(space_degree + 1)
    , quad_time(time_degree + 1)
    , mapping(mapping_)
    , dof_handler(dof_handler_)
    , exact_solution(exact_solution_)
    , evaluate_numerical_solution(evaluate_numerical_solution_)
  {}

  std::unordered_map<VectorTools::NormType, double>
  evaluate_error(const double time,
                 const double time_step,
                 BlockVectorType const &) const
  {
    std::unordered_map<VectorTools::NormType, double> error;
    auto const &tq = quad_time.get_points();
    auto const &tw = quad_time.get_weights();

    VectorType     numeric(dof_handler.n_dofs());
    Vector<double> differences_per_cell(
      dof_handler.get_triangulation().n_active_cells());

    error[VectorTools::L2_norm]     = 0.0;
    error[VectorTools::Linfty_norm] = -1.;
    error[VectorTools::H1_seminorm] = 0.0;

    double time_, ref_t;
    for (unsigned q = 0; q < quad_time.size(); ++q)
      {
        ref_t = tq[q][0];
        time_ = time + ref_t * time_step;
        exact_solution.set_time(time_);
        evaluate_numerical_solution(time_, numeric);

        dealii::VectorTools::integrate_difference(mapping,
                                                  dof_handler,
                                                  numeric,
                                                  exact_solution,
                                                  differences_per_cell,
                                                  quad_cell,
                                                  dealii::VectorTools::L2_norm);
        double l2 = dealii::VectorTools::compute_global_error(
          dof_handler.get_triangulation(),
          differences_per_cell,
          dealii::VectorTools::L2_norm);
        error[VectorTools::L2_norm] += time_step * tw[q] * l2 * l2;

        dealii::VectorTools::integrate_difference(
          mapping,
          dof_handler,
          numeric,
          exact_solution,
          differences_per_cell,
          quad_cell,
          dealii::VectorTools::Linfty_norm);
        double l8 = dealii::VectorTools::compute_global_error(
          dof_handler.get_triangulation(),
          differences_per_cell,
          dealii::VectorTools::Linfty_norm);
        error[VectorTools::Linfty_norm] =
          std::max(l8, error[VectorTools::Linfty_norm]);

        dealii::VectorTools::integrate_difference(
          mapping,
          dof_handler,
          numeric,
          exact_solution,
          differences_per_cell,
          quad_cell,
          dealii::VectorTools::H1_seminorm);
        double h1 = dealii::VectorTools::compute_global_error(
          dof_handler.get_triangulation(),
          differences_per_cell,
          dealii::VectorTools::H1_seminorm);
        error[VectorTools::H1_seminorm] += time_step * tw[q] * h1 * h1;
      }
    return error;
  }

private:
  TimeStepType                                    time_step_type;
  QGauss<dim> const                               quad_cell;
  QGauss<dim> const                               quad_time;
  const Mapping<dim>                             &mapping;
  const DoFHandler<dim>                          &dof_handler;
  mutable ExactSolution<dim, Number>              exact_solution;
  std::function<void(const double, VectorType &)> evaluate_numerical_solution;
};

template <int dim, typename Number>
class TimeIntegrator
{
public:
  using VectorType      = Vector<Number>;
  using BlockVectorType = BlockVector<Number>;

  TimeIntegrator(
    double const gmres_tolerance_,
    SystemMatrix<Number, MatrixFreeOperator<dim, Number>> const &matrix_,
    Preconditioner<Number> const                        &preconditioner_,
    std::function<void(const double, BlockVectorType &)> integrate_rhs_function)
    : solver_control(100, 1.e-16, gmres_tolerance_)
    , solver(solver_control)
    , preconditioner(preconditioner_)
    , matrix(matrix_)
    , integrate_rhs_function(integrate_rhs_function)
  {}

  void
  solve(BlockVectorType                    &x,
        [[maybe_unused]] const unsigned int timestep_number,
        const double                        time,
        [[maybe_unused]] const double       time_step) const
  {
    BlockVectorType rhs(x);
    rhs = 0.;

    integrate_rhs_function(time, rhs);
    try
      {
        solver.solve(matrix, x, rhs, preconditioner);
      }
    catch (const SolverControl::NoConvergence &e)
      {
        AssertThrow(false, ExcMessage(e.what()));
      }
  }


private:
  mutable ReductionControl                                     solver_control;
  mutable SolverGMRES<BlockVectorType>                         solver;
  Preconditioner<Number> const                                &preconditioner;
  SystemMatrix<Number, MatrixFreeOperator<dim, Number>> const &matrix;

  std::function<void(const double, BlockVectorType &)> integrate_rhs_function;
};

template <int dim>
void
test()
{
  using Number          = double;
  using BlockVectorType = BlockVector<Number>;
  using VectorType      = Vector<Number>;

  TimeStepType type = TimeStepType::DG;

  const unsigned int fe_degree = 2;
  const unsigned int n_blocks =
    type == TimeStepType::DG ? fe_degree + 1 : fe_degree;

  MappingQ1<dim> mapping;

  FE_Q<dim>   fe(fe_degree);
  QGauss<dim> quad(fe_degree + 1);

  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(1);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<Number> constraints;
  DoFTools::make_zero_boundary_constraints(dof_handler, constraints);
  constraints.close();

  // create sparsity pattern
  SparsityPattern sparsity_pattern(dof_handler.n_dofs(), dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, sparsity_pattern, constraints);

  // create scalar siffness matrix
  SparseMatrix<Number> K;
  K.reinit(sparsity_pattern);

  // create scalar mass matrix
  SparseMatrix<Number> M;
  M.reinit(sparsity_pattern);

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

  auto const [Alpha, Beta, Gamma, Zeta] =
    get_fe_time_weights<Number>(type, fe_degree);

  SystemMatrix<Number, MatrixFreeOperator<dim, Number>> matrix(K_mf,
                                                               M_mf,
                                                               Alpha,
                                                               Beta);

  Preconditioner<Number> preconditioner(K, M, Alpha, Beta, dof_handler);


  Number                     frequency = 1.0;
  RHSFunction<dim, Number>   rhs_function(frequency);
  ExactSolution<dim, Number> exact_solution(frequency);

  auto integrate_rhs_function =
    [&mapping, &dof_handler, &quad, &rhs_function, &constraints](
      const double time, BlockVectorType &tmp) -> void {
    rhs_function.set_time(time);
    VectorTools::create_right_hand_side(
      mapping, dof_handler, quad, rhs_function, tmp, constraints);
  };
  [[maybe_unused]] auto evaluate_exact_solution =
    [&mapping, &dof_handler, &exact_solution](const double time,
                                              VectorType  &tmp) -> void {
    exact_solution.set_time(time);
    VectorTools::interpolate(mapping, dof_handler, exact_solution, tmp);
  };
  auto evaluate_numerical_solution =
    [&mapping, &dof_handler, &quad, &exact_solution, &constraints](
      const double time, VectorType &tmp) -> void {
    (void)time;
    tmp = 0.;
  };

  BlockVectorType x(n_blocks);
  for (unsigned int i = 0; i < n_blocks; ++i)
    x.block(i).reinit(dof_handler.n_dofs());

  double time           = 0.;
  double time_step_size = 1.;
  double end_time       = 1.;
  double l2 = 0., l8 = -1., h1_semi = 0.;

  unsigned int                 timestep_number = 0;
  ErrorCalculator<dim, Number> error_calculator(type,
                                                fe_degree,
                                                fe_degree,
                                                mapping,
                                                dof_handler,
                                                exact_solution,
                                                evaluate_numerical_solution);

  TimeIntegrator<dim, Number> step(1.e-12,
                                   matrix,
                                   preconditioner,
                                   integrate_rhs_function);


  while (time > end_time + 0.5 * time_step_size)
    {
      time += time_step_size;
      ++timestep_number;

      step.solve(x, timestep_number, time, time_step_size);
      constraints.distribute(x);

      auto error_on_In = error_calculator.evaluate_error(0, 1.0, x);
      l2 += error_on_In[VectorTools::L2_norm];
      l8 = std::max(error_on_In[VectorTools::Linfty_norm], l8);
      h1_semi += error_on_In[VectorTools::H1_seminorm];
      DataOut<dim> data_out;
    }
}



int
main()
{
  return 0;

  const int dim = 2;

  test<dim>();
}
