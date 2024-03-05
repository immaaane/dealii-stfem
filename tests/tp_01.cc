#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/repartitioning_policy_tools.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_creator.h>

#include "include/exact_solution.h"
#include "include/fe_time.h"
#include "include/gmg.h"
#include "include/operators.h"
#include "include/time_integrators.h"

using namespace dealii;
using dealii::numbers::PI;

enum class ProblemType : unsigned int
{
  heat = 1,
  wave = 2,
};

template <typename Number_dst, typename Number_src>
FullMatrix<Number_dst>
convert_to(FullMatrix<Number_src> const &in)
{
  FullMatrix<Number_dst> out(in.m(), in.n());
  out.copy_from(in);
  return out;
}

template <int dim, typename Number, typename NumberPreconditioner = Number>
void
test(dealii::ConditionalOStream &pcout,
     MPI_Comm const              comm_global,
     TimeStepType const          type,
     ProblemType const           problem,
     unsigned int                n_timesteps_at_once = 1)
{
  ConvergenceTable table;
  MappingQ1<dim>   mapping;

  const bool print_timing = false;

  auto convergence_test = [&](int const          refinement,
                              unsigned int const fe_degree,
                              bool               do_output,
                              unsigned int       n_timesteps_at_once) {
    using VectorType      = VectorT<Number>;
    using BlockVectorType = BlockVectorT<Number>;
    const unsigned int n_blocks =
      (type == TimeStepType::DG ? fe_degree + 1 : fe_degree) *
      n_timesteps_at_once;
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
    Number frequency      = 1.0;

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

    // We need the case n_timesteps_at_once=1 matrices always for the
    // integration of the source f
    auto [Alpha_1, Beta_1, Gamma_1, Zeta_1] =
      get_fe_time_weights<Number>(type, fe_degree, time_step_size, 1);
    auto [Alpha, Beta, Gamma, Zeta] = get_fe_time_weights<Number>(
      type, fe_degree, time_step_size, n_timesteps_at_once);

    TimerOutput timer(pcout,
                      TimerOutput::never,
                      TimerOutput::cpu_and_wall_times);

    FullMatrix<Number> lhs_uK, lhs_uM, rhs_uK, rhs_uM, rhs_vM,
      zero(Gamma.m(), Gamma.n());
    FullMatrix<NumberPreconditioner> lhs_uK_p, lhs_uM_p, rhs_uK_p, rhs_uM_p,
      rhs_vM_p, zero_p(Gamma.m(), Gamma.n());
    std::unique_ptr<SystemMatrix<Number, MatrixFreeOperator<dim, Number>>>
      rhs_matrix, rhs_matrix_v, matrix;
    if (problem == ProblemType::wave)
      {
        auto [Alpha_lhs, Beta_lhs, rhs_uK_, rhs_uM_, rhs_vM_] =
          get_fe_time_weights_wave(
            type, Alpha_1, Beta_1, Gamma_1, Zeta_1, n_timesteps_at_once);

        lhs_uK = Alpha_lhs;
        lhs_uM = Beta_lhs;
        rhs_uK = rhs_uK_;
        rhs_uM = rhs_uM_;
        rhs_vM = rhs_vM_;

        rhs_vM_p     = convert_to<NumberPreconditioner>(rhs_vM);
        rhs_matrix_v = std::make_unique<
          SystemMatrix<Number, MatrixFreeOperator<dim, Number>>>(
          timer, K_mf, M_mf, zero, rhs_vM);
      }
    else
      {
        lhs_uK = Alpha;
        lhs_uM = Beta;
        rhs_uK = (type == TimeStepType::CGP) ? Gamma : zero;
        rhs_uM = (type == TimeStepType::CGP) ? Zeta : Gamma;
      }
    lhs_uK_p = convert_to<NumberPreconditioner>(lhs_uK);
    lhs_uM_p = convert_to<NumberPreconditioner>(lhs_uM);
    rhs_uK_p = convert_to<NumberPreconditioner>(rhs_uK);
    rhs_uM_p = convert_to<NumberPreconditioner>(rhs_uM);
    matrix =
      std::make_unique<SystemMatrix<Number, MatrixFreeOperator<dim, Number>>>(
        timer, K_mf, M_mf, lhs_uK, lhs_uM);
    rhs_matrix =
      std::make_unique<SystemMatrix<Number, MatrixFreeOperator<dim, Number>>>(
        timer, K_mf, M_mf, rhs_uK, rhs_uM);

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
    MGLevelObject<
      std::shared_ptr<const MatrixFreeOperator<dim, NumberPreconditioner>>>
      mg_M_mf(min_level, max_level);
    MGLevelObject<
      std::shared_ptr<const MatrixFreeOperator<dim, NumberPreconditioner>>>
      mg_K_mf(min_level, max_level);
    MGLevelObject<
      std::shared_ptr<const AffineConstraints<NumberPreconditioner>>>
      mg_constraints(min_level, max_level);
    MGLevelObject<std::shared_ptr<
      const SystemMatrix<NumberPreconditioner,
                         MatrixFreeOperator<dim, NumberPreconditioner>>>>
      mg_operators(min_level, max_level);
    MGLevelObject<std::shared_ptr<PreconditionVanka<NumberPreconditioner>>>
      precondition_vanka(min_level, max_level);

    for (unsigned int l = min_level; l <= max_level; ++l)
      {
        auto dof_handler_ =
          std::make_shared<DoFHandler<dim>>(*mg_triangulations[l]);
        auto constraints_ =
          std::make_shared<AffineConstraints<NumberPreconditioner>>();
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
        auto K_mf_ =
          std::make_shared<MatrixFreeOperator<dim, NumberPreconditioner>>(
            mapping, *dof_handler_, *constraints_, quad, 0.0, 1.0);
        auto M_mf_ =
          std::make_shared<MatrixFreeOperator<dim, NumberPreconditioner>>(
            mapping, *dof_handler_, *constraints_, quad, 1.0, 0.0);


        mg_operators[l] = std::make_shared<
          SystemMatrix<NumberPreconditioner,
                       MatrixFreeOperator<dim, NumberPreconditioner>>>(
          timer, *K_mf_, *M_mf_, lhs_uK_p, lhs_uM_p);

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
        precondition_vanka[l] =
          std::make_shared<PreconditionVanka<NumberPreconditioner>>(
            timer,
            mg_K[l],
            mg_M[l],
            mg_sparsity_patterns[l],
            lhs_uK_p,
            lhs_uM_p,
            mg_dof_handlers[l]);
      }



    // std::shared_ptr<MGSmootherBase<BlockVectorType>> smoother =
    //   std::make_shared<MGSmootherIdentity<BlockVectorType>>();
    std::unique_ptr<BlockVectorT<NumberPreconditioner>> tmp1, tmp2;
    if (!std::is_same_v<Number, NumberPreconditioner>)
      {
        tmp1 = std::make_unique<BlockVectorT<NumberPreconditioner>>();
        tmp2 = std::make_unique<BlockVectorT<NumberPreconditioner>>();
        matrix->initialize_dof_vector(*tmp1);
        matrix->initialize_dof_vector(*tmp2);
      }
    auto preconditioner = std::make_unique<
      GMG<dim,
          NumberPreconditioner,
          SystemMatrix<NumberPreconditioner,
                       MatrixFreeOperator<dim, NumberPreconditioner>>>>(
      timer,
      dof_handler,
      mg_dof_handlers,
      mg_constraints,
      mg_operators,
      precondition_vanka,
      std::move(tmp1),
      std::move(tmp2));
    preconditioner->reinit();
    //
    /// GMG

    // Number                      frequency = 1.0;
    std::unique_ptr<Function<dim, Number>> rhs_function;
    std::unique_ptr<Function<dim, Number>>
      exact_solution = std::make_unique<ExactSolution<dim, Number>>(frequency),
      exact_solution_v;
    if (problem == ProblemType::wave)
      {
        rhs_function =
          std::make_unique<wave::RHSFunction<dim, Number>>(frequency);
        exact_solution_v =
          std::make_unique<wave::ExactSolutionV<dim, Number>>(frequency);
      }
    else
      {
        rhs_function = std::make_unique<RHSFunction<dim, Number>>(frequency);
      }
    auto integrate_rhs_function =
      [&mapping, &dof_handler, &quad, &rhs_function, &constraints](
        const double time, VectorType &rhs) -> void {
      rhs_function->set_time(time);
      rhs = 0.0;
      VectorTools::create_right_hand_side(
        mapping, dof_handler, quad, *rhs_function, rhs, constraints);
    };
    [[maybe_unused]] auto evaluate_exact_solution =
      [&mapping, &dof_handler, &exact_solution](const double time,
                                                VectorType  &tmp) -> void {
      exact_solution->set_time(time);
      VectorTools::interpolate(mapping, dof_handler, *exact_solution, tmp);
    };
    [[maybe_unused]] auto evaluate_exact_v_solution =
      [&mapping, &dof_handler, &exact_solution_v](const double time,
                                                  VectorType  &tmp) -> void {
      exact_solution_v->set_time(time);
      VectorTools::interpolate(mapping, dof_handler, *exact_solution_v, tmp);
    };
    auto evaluate_numerical_solution =
      [&constraints, &basis, &type](const double           time,
                                    VectorType            &tmp,
                                    BlockVectorType const &x,
                                    VectorType const      &prev_x,
                                    unsigned block_offset = 0) -> void {
      int i = 0;
      tmp   = 0.0;
      for (auto const &el : basis)
        {
          if (double v = el.value(time); v != 0.0)
            {
              if (type == TimeStepType::DG)
                tmp.add(v, x.block(block_offset + i));
              else
                tmp.add(v,
                        (block_offset + i == 0) ?
                          prev_x :
                          x.block(block_offset + i - 1));
            }
          ++i;
        }
      constraints.distribute(tmp);
    };

    BlockVectorType x(n_blocks), v(n_blocks);
    for (unsigned int i = 0; i < n_blocks; ++i)
      matrix->initialize_dof_vector(x.block(i));
    VectorType prev_x, prev_v;
    matrix->initialize_dof_vector(prev_x);
    if (problem == ProblemType::wave)
      {
        matrix->initialize_dof_vector(prev_v);
        for (unsigned int i = 0; i < n_blocks; ++i)
          matrix->initialize_dof_vector(v.block(i));
      }

    VectorType exact;
    matrix->initialize_dof_vector(exact);
    VectorType numeric;
    matrix->initialize_dof_vector(numeric);

    unsigned int                 timestep_number = 0;
    ErrorCalculator<dim, Number> error_calculator(type,
                                                  fe_degree,
                                                  fe_degree,
                                                  mapping,
                                                  dof_handler,
                                                  *exact_solution,
                                                  evaluate_numerical_solution);

    std::unique_ptr<TimeIntegrator<dim, Number, NumberPreconditioner>> step;
    if (problem == ProblemType::heat)
      step =
        std::make_unique<TimeIntegratorHeat<dim, Number, NumberPreconditioner>>(
          type,
          fe_degree,
          Alpha_1,
          Gamma_1,
          1.e-12,
          *matrix,
          *preconditioner,
          *rhs_matrix,
          integrate_rhs_function,
          n_timesteps_at_once);
    else
      step =
        std::make_unique<TimeIntegratorWave<dim, Number, NumberPreconditioner>>(
          type,
          fe_degree,
          Alpha_1,
          Beta_1,
          Gamma_1,
          Zeta_1,
          1.e-12,
          *matrix,
          *preconditioner,
          *rhs_matrix,
          *rhs_matrix_v,
          integrate_rhs_function,
          n_timesteps_at_once);

    // interpolate initial value
    auto nt_dofs = static_cast<unsigned int>(n_blocks / n_timesteps_at_once);
    evaluate_exact_solution(0, x.block(x.n_blocks() - 1));
    if (problem == ProblemType::wave)
      evaluate_exact_v_solution(0, v.block(v.n_blocks() - 1));
    double l2 = 0., l8 = -1., h1_semi = 0.;
    int    i = 0;
    while (time < end_time)
      {
        TimerOutput::Scope scope(timer, "step");

        ++timestep_number;
        dealii::deallog << "Step " << timestep_number << " t = " << time
                        << std::endl;
        prev_x = x.block(x.n_blocks() - 1);
        if (problem == ProblemType::heat)
          static_cast<
            TimeIntegratorHeat<dim, Number, NumberPreconditioner> const *>(
            step.get())
            ->solve(x, prev_x, timestep_number, time, time_step_size);
        else
          {
            prev_v = v.block(v.n_blocks() - 1);
            static_cast<
              TimeIntegratorWave<dim, Number, NumberPreconditioner> const *>(
              step.get())
              ->solve(
                x, v, prev_x, prev_v, timestep_number, time, time_step_size);
          }
        for (unsigned int i = 0; i < n_blocks; ++i)
          constraints.distribute(x.block(i));
        auto error_on_In = error_calculator.evaluate_error(
          time, time_step_size, x, prev_x, n_timesteps_at_once);
        time += n_timesteps_at_once * time_step_size;

        l2 += error_on_In[VectorTools::L2_norm];
        l8 = std::max(error_on_In[VectorTools::Linfty_norm], l8);
        h1_semi += error_on_In[VectorTools::H1_seminorm];
        ++i;
        if (do_output)
          {
            numeric = 0.0;
            evaluate_numerical_solution(
              1.0, numeric, x, prev_x, (n_timesteps_at_once - 1) * nt_dofs);
            DataOut<dim> data_out;
            data_out.attach_dof_handler(dof_handler);
            data_out.add_data_vector(numeric, "solution");
            data_out.build_patches();

            std::ofstream output("solution." +
                                 Utilities::int_to_string(timestep_number, 4) +
                                 ".vtu");
            data_out.write_vtu(output);
          }
#ifdef DEBUG
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
#endif
      }

    if (print_timing)
      timer.print_wall_time_statistics(MPI_COMM_WORLD);

    unsigned int const n_active_cells = tria.n_active_cells();
    unsigned int const n_dofs         = dof_handler.n_dofs();
    table.add_value("cells", n_active_cells);
    table.add_value("s-dofs", n_dofs);
    table.add_value("t-dofs", n_blocks);
    table.add_value("st-dofs", i * n_dofs * n_blocks);
    table.add_value("L\u221E-L\u221E", l8);
    table.add_value("L2-L2", std::sqrt(l2));
    table.add_value("L2-H1_semi", std::sqrt(h1_semi));
  };
  for (int j = 0; j < 3; ++j)
    {
      for (int i = 2; i < 6; ++i)
        convergence_test(i,
                         type == TimeStepType::DG ? j : j + 1,
                         false,
                         n_timesteps_at_once);

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
// #define PRECONDITIONER_FLOAT
#ifdef PRECONDITIONER_FLOAT
#  define test test<dim, double, float>
#else
#  define test test<dim, double, double>
#endif
  dealii::deallog << "HEAT 2 steps at once" << std::endl;
  test(pcout, comm, TimeStepType::DG, ProblemType::heat, 2);
  test(pcout, comm, TimeStepType::CGP, ProblemType::heat, 2);

  dealii::deallog << "HEAT single step" << std::endl;
  test(pcout, comm, TimeStepType::DG, ProblemType::heat);
  test(pcout, comm, TimeStepType::CGP, ProblemType::heat);

  dealii::deallog << "WAVE 4 steps at once" << std::endl;
  test(pcout, comm, TimeStepType::DG, ProblemType::wave, 4);
  test(pcout, comm, TimeStepType::CGP, ProblemType::wave, 4);

  dealii::deallog << "WAVE single step" << std::endl;
  test(pcout, comm, TimeStepType::DG, ProblemType::wave);
  test(pcout, comm, TimeStepType::CGP, ProblemType::wave);
#undef test

  dealii::deallog << std::endl;
  pcout << std::endl;
}
