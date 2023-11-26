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

#include "fe_time.h"

using namespace dealii;
using dealii::numbers::PI;

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

  ErrorCalculator(TimeStepType           type_,
                  unsigned int           time_order,
                  unsigned int           space_order,
                  Mapping<dim> const    &mapping,
                  DoFHandler<dim> const &dof_handler,
                  std::function<void(const double, VectorType &)>
                    &const evaluate_exact_solution_,
                  std::function<void(const double, VectorType &)>
                    &const evaluate_numerical_solution_)
    : time_step_type(type_)
    , quad_cell(space_order + 1)
    , quad_time(time_order + 1)
    , evaluate_exact_solution(evaluate_exact_solution_)
    , evaluate_numerical_solution(evaluate_numerical_solution_)
  {}

  std::unordored_map<dealii::VectorTools::NormType, double>
  evaluate_error(const double time,
                 const double time_step,
                 BlockVectorType const &)
  {
    std::unordored_map<dealii::VectorTools::NormType, double> error;

    double l2, l8, h1, time_;
    for (unsigned q = 0; q < quad_time.size(); ++q)
      {
        ref_t = tq[q][0];
        time_ = time + ref_t * time_step;

        error[dealii::VectorTools::L2_norm] += l2;
        error[dealii::VectorTools::Linfty_norm] += l8;
        error[dealii::VectorTools::H1_norm] += h1;
      }
  }

private:
  TimeStepType           time_step_type;
  QGauss<dim> const      quad_cell;
  QGauss<dim> const      quad_time;
  const Mapping<dim>    &mapping;
  const DoFHandler<dim> &dof_handler;
  std::function<void(const double, VectorType &)>
    &const evaluate_exact_solution;
  std::function<void(const double, VectorType &)>
    &const evaluate_numerical_solution;
};

template <int dim, typename Number>
class time_step
{
public:
  using VectorType      = Vector<Number>;
  using BlockVectorType = BlockVector<Number>;

  time_step(
    TimeStepType       type_,
    unsigned const int time_order_,
    double const       gmres_tolerance_,
    std::function<void(const double, VectorType &)> const
      &integrate_rhs_function,
    std::function<void(const double, VectorType &)> const &evaluate_solution)
    : time_step_type(type_)
    , time_order(time_order_)
    , solver_control(100, 1.e-16, gmres_tolerance_)
    , solver(solver_control)
  {
    {
      auto const [Alpha_, Beta_, Gamma_, Zeta_] =
        get_fe_time_weights<Number>(type, degree);
      Alpha = Alpha_;
      Beta  = Beta_;
      Gamma = Gamma_;
      Zeta  = Zeta_;
    }
    this->matrix = SystemMatrix<Number, MatrixFreeOperator<dim, Number>>(K_mf,
                                                                         M_mf,
                                                                         Alpha,
                                                                         Beta);
    this->preconditioner =
      Preconditioner<Number>(K, M, Alpha, Beta, dof_handler);
  }

  void
  solve(BlockVectorType   &solution,
        const unsigned int timestep_number,
        const double       time,
        const double       time_step)
  {
    BlockVectorType rhs(n_blocks);
    for (unsigned int i = 0; i < n_blocks; ++i)
      rhs.block(i).reinit(dof_handler.n_dofs());

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
  TimeStepType       time_step_type;
  unsigned int       time_order;
  FullMatrix<Number> Alpha;
  FullMatrix<Number> Beta;
  FullMatrix<Number> Gamma;
  FullMatrix<Number> Zeta;

  ReductionControl                                      solver_control;
  SolverGMRES<BlockVectorType>                          solver;
  Preconditioner<Number>                                preconditioner;
  SystemMatrix<Number, MatrixFreeOperator<dim, Number>> matrix;

  std::function<void(const double, VectorType &)> const integrate_rhs_function;
};

template <int dim, typename Number>
class ExactSolution : Function<dim, Number>
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
  gradient(const dealii::Point<dim> &x, const unsigned int) const override final
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
class RHSFunction : Function<dim, Number>
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

template <int dim>
void
test()
{
  using Number          = double;
  using BlockVectorType = BlockVector<Number>;

  const unsigned int fe_degree = 2;
  const unsigned int n_blocks  = 3;

  MappingQ1<dim> mapping;

  FE_Q<dim>   fe(fe_degree);
  QGauss<dim> quad(fe_degree + 1);

  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(4);

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

  Number                     frequency = 1.0;
  RHSFunction<dim, Number>   rhs_function(frequency);
  ExactSolution<dim, Number> exact_solution(frequency);

  const auto integrate_rhs_function =
    [&dof_handler, &quad, &rhs_function, &constraints](const double time,
                                                       VectorType  &tmp) {
      rhs_function.set_time(time);
      VectorTools::create_right_hand_side(
        dof_handler, quad, rhs_function, tmp, constraints);
    };
  const auto evaluate_exact_solution =
    [&dof_handler, &quad, &exact_solution, &constraints](const double time,
                                                         VectorType  &tmp) {
      exact_solution.set_time(time);
      VectorTools::interpolate(
        dof_handler, quad, exact_solution, tmp, constraints);
    };


  TimeStepType type = TimeStepType::DG;
  auto const [Alpha, Beta, Gamma, Zeta] =
    get_fe_time_weights<Number>("DG", fe_degree);

  BlockVectorType x(n_blocks);
  for (unsigned int i = 0; i < n_blocks; ++i)
    x.block(i).reinit(dof_handler.n_dofs());

  ErrorCalculator<dim, Number> error_calculator(type,
                                                fe_degree,
                                                fe_degree,
                                                mapping,
                                                dof_handler,
                                                evaluate_exact_solution,
                                                integrate_rhs_function);

  time_step<dim, Number> step(type, fe_degree, 1.e-12);


  step.solve(x, 0, 0.0, 1.0, integrate_rhs_function);
  error_calculator.evaluate_error(0, 1.0, x);


  DataOut<dim> data_out;
}



int
main()
{
  return 0;

  const int dim = 2;

  test<dim>();
}
