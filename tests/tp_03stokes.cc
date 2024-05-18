#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/repartitioning_policy_tools.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_dgp.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_creator.h>

#include "include/exact_solution.h"
#include "include/fe_time.h"
#include "include/getopt++.h"
#include "include/gmg.h"
#include "include/operators.h"
#include "include/time_integrators.h"

using namespace dealii;
using dealii::numbers::PI;

template <typename Number, typename NumberPreconditioner = Number>
void
test(dealii::ConditionalOStream &pcout,
     MPI_Comm const              comm_global,
     std::string                 file_name,
     int                         dim)
{
  std::variant<Parameters<2>, Parameters<3>> parameters;
  if (dim == 2)
    parameters = Parameters<2>();
  else
    parameters = Parameters<3>();
  std::visit([&](auto &p) { p.parse(file_name); }, parameters);
  ConvergenceTable table;
  ConvergenceTable itable;

  auto convergence_test = [&]<int dim>(int const              refinement,
                                       int const              fe_degree,
                                       Parameters<dim> const &parameters) {
    const bool print_timing      = parameters.print_timing;
    const bool space_time_mg     = parameters.space_time_mg;
    const bool time_before_space = parameters.time_before_space;
    const bool is_cgp            = parameters.type == TimeStepType::CGP;
    Assert(parameters.fe_degree >= (is_cgp ? 1 : 0),
           ExcLowerRange(parameters.fe_degree, (is_cgp ? 1 : 0)));
    Assert(parameters.refinement >= 1, ExcLowerRange(parameters.refinement, 1));

    MappingQ1<dim>     mapping;
    bool const         do_output           = parameters.do_output;
    unsigned int const n_timesteps_at_once = parameters.n_timesteps_at_once;

    using VectorType            = VectorT<Number>;
    using BlockVectorType       = BlockVectorT<Number>;
    const unsigned int nt_dofs  = is_cgp ? fe_degree : fe_degree + 1;
    const unsigned int n_blocks = nt_dofs * n_timesteps_at_once;
    BlockSlice         blk_slice(n_timesteps_at_once, 2, nt_dofs);

    auto const    basis = get_time_basis(parameters.type, fe_degree);
    FESystem<dim> fe_u(FE_Q<dim>(fe_degree + 1), dim);
    FE_Q<dim>     fe_p(fe_degree);
    QGauss<dim>   quad_u(fe_u.tensor_degree() + 1);
    QGauss<dim>   quad_p(fe_p.tensor_degree() + 1);
    std::vector<Quadrature<dim>> quads{quad_u, quad_p};
    std::vector<Quadrature<dim>> quads_u{quad_u, quad_u};

    parallel::distributed::Triangulation<dim> tria(comm_global);
    DoFHandler<dim>                           dof_handler_u(tria);
    DoFHandler<dim>                           dof_handler_p(tria);
    std::vector<const DoFHandler<dim> *>      dof_handlers(
      {&dof_handler_u, &dof_handler_p});

    GridGenerator::subdivided_hyper_rectangle(tria,
                                              parameters.subdivisions,
                                              parameters.hyperrect_lower_left,
                                              parameters.hyperrect_upper_right);
    double spc_step  = GridTools::minimal_cell_diameter(tria) / std::sqrt(dim);
    double viscosity = parameters.viscosity;
    tria.refine_global(refinement);
    if (parameters.distort_grid != 0.0)
      GridTools::distort_random(parameters.distort_grid, tria);
    dof_handler_u.distribute_dofs(fe_u);
    dof_handler_p.distribute_dofs(fe_p);


    dealii::AffineConstraints<Number>                      constraints_u;
    dealii::AffineConstraints<Number>                      constraints_p;
    std::vector<const dealii::AffineConstraints<Number> *> constraints{
      &constraints_u, &constraints_p};

    IndexSet locally_relevant_dofs_u;
    IndexSet locally_relevant_dofs_p;
    DoFTools::extract_locally_relevant_dofs(dof_handler_u,
                                            locally_relevant_dofs_u);
    DoFTools::extract_locally_relevant_dofs(dof_handler_p,
                                            locally_relevant_dofs_p);
    constraints_u.reinit(locally_relevant_dofs_u);
    constraints_p.reinit(locally_relevant_dofs_p);
    DoFTools::make_hanging_node_constraints(dof_handler_u, constraints_u);
    DoFTools::make_zero_boundary_constraints(dof_handler_u, constraints_u);
    DoFTools::make_hanging_node_constraints(dof_handler_p, constraints_p);
    constraints_u.close();
    constraints_p.close();
    pcout << "\n:: Number of active cells: " << tria.n_global_active_cells()
          << "\n:: Number of u degrees of freedom: " << dof_handler_u.n_dofs()
          << "\n:: Number of p degrees of freedom: " << dof_handler_p.n_dofs()
          << "\n:: Number of u degrees of freedom per cell: "
          << dof_handler_u.get_fe().n_dofs_per_cell()
          << "\n:: Number of p degrees of freedom per cell: "
          << dof_handler_p.get_fe().n_dofs_per_cell() << "\n";

    double       time     = 0.;
    double       time_len = parameters.end_time - time;
    unsigned int n_steps  = static_cast<unsigned int>((time_len) / spc_step);
    double time_step_size = time_len * pow(2.0, -(refinement + 1)) / n_steps;

    // matrix-free operators
    StokesMatrixFreeOperator<dim, Number> Stokes_mf(
      mapping, dof_handlers, constraints, quads_u, viscosity);
    MatrixFreeOperatorVector<dim, Number> M_mf(
      mapping, dof_handler_u, constraints_u, quad_u, 1.0, 0.0);

    // We need the case n_timesteps_at_once=1 matrices always for the
    // integration of the source f

    auto [Alpha_1, Beta_1, Gamma_1, Zeta_1] =
      get_fe_time_weights_stokes<Number>(parameters.type,
                                         fe_degree,
                                         time_step_size,
                                         1);
    auto [Alpha, Beta, Gamma, Zeta] = get_fe_time_weights_stokes<Number>(
      parameters.type, fe_degree, time_step_size, n_timesteps_at_once);

    TimerOutput timer(pcout,
                      TimerOutput::never,
                      TimerOutput::cpu_and_wall_times);

    FullMatrix<Number> lhs_uK, lhs_uM, rhs_uK, rhs_uM,
      zero(Gamma.m(), Gamma.n());

    using SystemN = SystemMatrixStokes<Number,
                                       StokesMatrixFreeOperator<dim, Number>,
                                       MatrixFreeOperatorVector<dim, Number>>;
    using SystemNP =
      SystemMatrixStokes<NumberPreconditioner,
                         StokesMatrixFreeOperator<dim, NumberPreconditioner>,
                         MatrixFreeOperatorVector<dim, NumberPreconditioner>>;
    std::unique_ptr<SystemN> rhs_matrix, matrix;
    lhs_uK = Alpha;
    lhs_uM = Beta;
    rhs_uK = is_cgp ? Gamma : zero;
    rhs_uM = is_cgp ? Zeta : Gamma;
    matrix = std::make_unique<SystemN>(
      timer, Stokes_mf, M_mf, lhs_uK, lhs_uM, blk_slice);
    rhs_matrix = std::make_unique<SystemN>(
      timer, Stokes_mf, M_mf, rhs_uK, rhs_uM, blk_slice);


    /// GMG
    RepartitioningPolicyTools::DefaultPolicy<dim>          policy(true);
    std::vector<std::shared_ptr<const Triangulation<dim>>> mg_triangulations =
      MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(
        tria, policy);
    unsigned int fe_degree_min =
      space_time_mg ? parameters.fe_degree_min : fe_degree;
    unsigned int n_timesteps_min =
      space_time_mg ? std::max(parameters.n_timesteps_at_once_min, 1) :
                      n_timesteps_at_once;
    auto poly_time_sequence = get_poly_mg_sequence(fe_degree,
                                                   fe_degree_min,
                                                   parameters.poly_coarsening);

    std::vector<TimeMGType> mg_type_level =
      get_time_mg_sequence(mg_triangulations.size(),
                           poly_time_sequence,
                           n_timesteps_at_once,
                           n_timesteps_min,
                           TimeMGType::k,
                           time_before_space);
    mg_triangulations =
      get_space_time_triangulation(mg_type_level, mg_triangulations);


    const unsigned int min_level = 0;
    const unsigned int max_level = mg_triangulations.size() - 1;
    pcout << ":: Min Level " << min_level << "  Max Level " << max_level
          << "\n";
    MGLevelObject<std::shared_ptr<const DoFHandler<dim>>> mg_dof_handlers_u(
      min_level, max_level);
    MGLevelObject<std::shared_ptr<const DoFHandler<dim>>> mg_dof_handlers_p(
      min_level, max_level);
    MGLevelObject<std::vector<const DoFHandler<dim> *>> mg_dof_handlers(
      min_level, max_level);

    MGLevelObject<std::shared_ptr<const BlockSparsityPatternType>>
      mg_sparsity_patterns(min_level, max_level);
    MGLevelObject<std::shared_ptr<const BlockSparseMatrixType>> mg_M(min_level,
                                                                     max_level);
    MGLevelObject<std::shared_ptr<const BlockSparseMatrixType>> mg_Stokes(
      min_level, max_level);

    MGLevelObject<std::shared_ptr<
      const MatrixFreeOperatorVector<dim, NumberPreconditioner>>>
      mg_M_mf(min_level, max_level);
    MGLevelObject<std::shared_ptr<
      const StokesMatrixFreeOperator<dim, NumberPreconditioner>>>
      mg_Stokes_mf(min_level, max_level);

    MGLevelObject<
      std::shared_ptr<const AffineConstraints<NumberPreconditioner>>>
      mg_constraints_u(min_level, max_level);
    MGLevelObject<
      std::shared_ptr<const AffineConstraints<NumberPreconditioner>>>
      mg_constraints_p(min_level, max_level);
    MGLevelObject<
      std::vector<const dealii::AffineConstraints<NumberPreconditioner> *>>
      mg_constraints(min_level, max_level);

    MGLevelObject<std::shared_ptr<const SystemNP>> mg_operators(min_level,
                                                                max_level);
    MGLevelObject<std::shared_ptr<PreconditionVanka<NumberPreconditioner>>>
      precondition_vanka(min_level, max_level);
    std::vector<std::array<FullMatrix<NumberPreconditioner>, 4>> fetw;
    fetw = get_fe_time_weights<NumberPreconditioner>(
      parameters.type,
      time_step_size,
      n_timesteps_at_once,
      mg_type_level,
      poly_time_sequence,
      get_fe_time_weights_stokes<NumberPreconditioner>);
    Table<2, bool> K_mask;
    Table<2, bool> M_mask(2, 2);
    M_mask.fill(false);
    M_mask(0, 0)                   = true;
    unsigned int const n_levels    = mg_dof_handlers.n_levels();
    unsigned int const n_variables = dof_handlers.size();

    std::vector<BlockSlice> blk_indices = get_blk_indices(parameters.type,
                                                          n_timesteps_at_once,
                                                          n_variables,
                                                          n_levels,
                                                          mg_type_level,
                                                          poly_time_sequence);

    for (unsigned int l = min_level, i = 0; l <= max_level; ++l, ++i)
      {
        auto dof_handler_p_ =
          std::make_shared<DoFHandler<dim>>(*mg_triangulations[l]);
        auto dof_handler_u_ =
          std::make_shared<DoFHandler<dim>>(*mg_triangulations[l]);

        auto constraints_p_ =
          std::make_shared<AffineConstraints<NumberPreconditioner>>();
        auto constraints_u_ =
          std::make_shared<AffineConstraints<NumberPreconditioner>>();

        dof_handler_p_->distribute_dofs(fe_p);
        dof_handler_u_->distribute_dofs(fe_u);

        IndexSet locally_relevant_dofs_u_;
        IndexSet locally_relevant_dofs_p_;
        DoFTools::extract_locally_relevant_dofs(*dof_handler_u_,
                                                locally_relevant_dofs_u_);
        DoFTools::extract_locally_relevant_dofs(*dof_handler_p_,
                                                locally_relevant_dofs_p_);
        constraints_u_->reinit(locally_relevant_dofs_u_);
        constraints_p_->reinit(locally_relevant_dofs_p_);
        DoFTools::make_hanging_node_constraints(*dof_handler_u_,
                                                *constraints_u_);
        DoFTools::make_zero_boundary_constraints(*dof_handler_u_,
                                                 *constraints_u_);
        DoFTools::make_hanging_node_constraints(*dof_handler_p_,
                                                *constraints_p_);
        constraints_u_->close();
        constraints_p_->close();

        // matrix-free operators
        mg_dof_handlers_u[l] = dof_handler_u_;
        mg_dof_handlers_p[l] = dof_handler_p_;
        mg_dof_handlers[l]   = {dof_handler_u_.get(), dof_handler_p_.get()};
        mg_constraints_u[l]  = constraints_u_;
        mg_constraints_p[l]  = constraints_p_;
        mg_constraints[l]    = {constraints_u_.get(), constraints_p_.get()};

        auto Stokes_mf_ =
          std::make_shared<StokesMatrixFreeOperator<dim, NumberPreconditioner>>(
            mapping, mg_dof_handlers[l], mg_constraints[l], quads_u, viscosity);
        auto M_mf_ =
          std::make_shared<MatrixFreeOperatorVector<dim, NumberPreconditioner>>(
            mapping,
            *mg_dof_handlers_u[l],
            *mg_constraints_u[l],
            quad_u,
            1.0,
            0.0);

        auto const &lhs_uK_p = fetw[l][0];
        auto const &lhs_uM_p = fetw[l][1];

        mg_operators[l] = std::make_shared<SystemNP>(
          timer, *Stokes_mf_, *M_mf_, lhs_uK_p, lhs_uM_p, blk_indices[i]);

        auto sparsity_pattern = std::make_shared<BlockSparsityPatternType>();
        auto Stokes           = std::make_shared<BlockSparseMatrixType>();
        auto M                = std::make_shared<BlockSparseMatrixType>();

        // create sparsity pattern
        {
          std::vector<IndexSet> row_partitioning{
            dof_handler_u_->locally_owned_dofs(),
            dof_handler_p_->locally_owned_dofs()};
          std::vector<IndexSet> column_partitioning{
            dof_handler_u_->locally_owned_dofs(),
            dof_handler_p_->locally_owned_dofs()};
          std::vector<IndexSet> writeable_rows{locally_relevant_dofs_u_,
                                               locally_relevant_dofs_p_};
          auto const subdomain = dealii::Utilities::MPI::this_mpi_process(
            dof_handler_u.get_communicator());
          sparsity_pattern->reinit(row_partitioning,
                                   column_partitioning,
                                   writeable_rows,
                                   dof_handler_u.get_communicator());
          make_block_sparsity_pattern_block(*dof_handler_u_,
                                            *dof_handler_u_,
                                            sparsity_pattern->block(0, 0),
                                            *constraints_u_,
                                            *constraints_u_,
                                            true,
                                            subdomain);
          make_block_sparsity_pattern_block(*dof_handler_u_,
                                            *dof_handler_p_,
                                            sparsity_pattern->block(0, 1),
                                            *constraints_u_,
                                            *constraints_p_,
                                            true,
                                            subdomain);
          make_block_sparsity_pattern_block(*dof_handler_p_,
                                            *dof_handler_u_,
                                            sparsity_pattern->block(1, 0),
                                            *constraints_p_,
                                            *constraints_u_,
                                            true,
                                            subdomain);
          sparsity_pattern->compress();
        }
        // create Stokes matrix
        Stokes->reinit(*sparsity_pattern);
        Stokes_mf_->compute_system_matrix(*Stokes);
        //  create vector-valued mass matrix
        M->reinit(sparsity_pattern->n_block_rows(),
                  sparsity_pattern->n_block_cols());
        M->block(0, 0).reinit(sparsity_pattern->block(0, 0));
        M_mf_->compute_system_matrix(M->block(0, 0));

        // matrix->attach(*mg_operators[l]);
        mg_sparsity_patterns[l] = sparsity_pattern;
        mg_M_mf[l]              = M_mf_;
        mg_Stokes_mf[l]         = Stokes_mf_;
        mg_M[l]                 = M;
        mg_Stokes[l]            = Stokes;
        precondition_vanka[l] =
          std::make_shared<PreconditionVanka<NumberPreconditioner>>(
            timer,
            mg_Stokes[l],
            mg_M[l],
            mg_sparsity_patterns[l],
            lhs_uK_p,
            lhs_uM_p,
            mg_dof_handlers[l],
            blk_indices[i],
            K_mask,
            M_mask);
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
    using Preconditioner = GMG<dim, NumberPreconditioner, SystemNP>;
    std::unique_ptr<Preconditioner> preconditioner =
      std::make_unique<Preconditioner>(timer,
                                       parameters,
                                       fe_degree,
                                       n_timesteps_at_once,
                                       mg_type_level,
                                       poly_time_sequence,
                                       dof_handlers,
                                       mg_dof_handlers,
                                       mg_constraints,
                                       mg_operators,
                                       precondition_vanka,
                                       std::move(tmp1),
                                       std::move(tmp2));
    preconditioner->reinit();
    //
    /// GMG

    std::unique_ptr<Function<dim, Number>> rhs_function_u, rhs_function_p;
    std::unique_ptr<Function<dim, Number>> exact_solution_u, exact_solution_p;
    if (parameters.space_time_conv_test)
      {
        exact_solution_u =
          std::make_unique<stokes::ExactSolutionU<dim, Number>>();
        exact_solution_p =
          std::make_unique<stokes::ExactSolutionP<dim, Number>>();
        rhs_function_u =
          std::make_unique<stokes::RHSFunction<dim, Number>>(viscosity, false);
        rhs_function_p =
          std::make_unique<Functions::ZeroFunction<dim, Number>>(1);
      }
    else
      {
        exact_solution_u =
          std::make_unique<Functions::CutOffFunctionCinfty<dim>>(
            1.e-2, parameters.source, 1, numbers::invalid_unsigned_int, true);
        rhs_function_u =
          std::make_unique<Functions::ZeroFunction<dim, Number>>(dim);
        rhs_function_p =
          std::make_unique<Functions::ZeroFunction<dim, Number>>(1);
      }
    auto integrate_rhs_function_u = [&mapping,
                                     &dof_handlers,
                                     &quads,
                                     &rhs_function_u,
                                     &rhs_function_p,
                                     &constraints,
                                     &parameters](const double time,
                                                  VectorType  &rhs) -> void {
      rhs_function_u->set_time(time);
      rhs = 0.0;
      if (parameters.space_time_conv_test)
        VectorTools::create_right_hand_side(mapping,
                                            *dof_handlers[0],
                                            quads[0],
                                            *rhs_function_u,
                                            rhs,
                                            *constraints[0]);
    };
    auto integrate_rhs_function_p = [&mapping,
                                     &dof_handlers,
                                     &quads,
                                     &rhs_function_u,
                                     &rhs_function_p,
                                     &constraints,
                                     &parameters](const double time,
                                                  VectorType  &rhs) -> void {
      rhs_function_p->set_time(time);
      rhs = 0.0;
      if (parameters.space_time_conv_test)
        VectorTools::create_right_hand_side(mapping,
                                            *dof_handlers[1],
                                            quads[1],
                                            *rhs_function_p,
                                            rhs,
                                            *constraints[1]);
    };
    auto evaluate_exact_solution_u = [&mapping,
                                      &dof_handlers,
                                      &exact_solution_u,
                                      &exact_solution_p,
                                      &parameters](const double time,
                                                   VectorType  &tmp) -> void {
      exact_solution_u->set_time(time);
      VectorTools::interpolate(mapping,
                               *dof_handlers[0],
                               *exact_solution_u,
                               tmp);
    };
    auto evaluate_exact_solution_p = [&mapping,
                                      &dof_handlers,
                                      &exact_solution_u,
                                      &exact_solution_p,
                                      &parameters](const double time,
                                                   VectorType  &tmp) -> void {
      exact_solution_p->set_time(time);
      VectorTools::interpolate(mapping,
                               *dof_handlers[1],
                               *exact_solution_p,
                               tmp);
    };
    auto evaluate_numerical_solution =
      [&constraints, &basis, &is_cgp, &blk_slice](int          variable,
                                                  const double time,
                                                  VectorType  &tmp,
                                                  BlockVectorType const &x,
                                                  VectorType const      &prev_x,
                                                  unsigned blk_offset) -> void {
      int i      = 0;
      tmp        = 0.0;
      auto slice = blk_slice.get_time(x, variable);
      auto td    = slice.begin() + blk_offset;
      for (auto const &el : basis)
        {
          if (double v = el.value(time); v != 0.0)
            {
              if (!is_cgp)
                tmp.add(v, td->get());
              else
                tmp.add(v, blk_offset + i == 0 ? prev_x : std::prev(td)->get());
            }
          ++td;
          ++i;
        }
      constraints[variable]->distribute(tmp);
    };
    auto evaluate_numerical_solution_u = [&](const double           time,
                                             VectorType            &tmp,
                                             BlockVectorType const &x,
                                             VectorType const      &prev_x,
                                             unsigned blk_offset = 0) -> void {
      evaluate_numerical_solution(0, time, tmp, x, prev_x, blk_offset);
    };

    auto evaluate_numerical_solution_p = [&](const double           time,
                                             VectorType            &tmp,
                                             BlockVectorType const &x,
                                             VectorType const      &prev_x,
                                             unsigned blk_offset = 0) -> void {
      evaluate_numerical_solution(1, time, tmp, x, prev_x, blk_offset);
    };

    BlockVectorType x, v;
    matrix->initialize_dof_vector(x);
    BlockVectorType prev_x(2);
    matrix->initialize_dof_vector(prev_x.block(0), 0);
    matrix->initialize_dof_vector(prev_x.block(1), 1);

    // Point eval
    auto real_points = dim == 2 ?
                         std::vector<Point<dim, Number>>{{0.75, 0}} :
                         std::vector<Point<dim, Number>>{{0.75, 0, 0},
                                                         {0, 0, 0.75},
                                                         {0.75, 0.1, 0.75}};

    Utilities::MPI::RemotePointEvaluation<dim, dim> rpe;
    rpe.reinit(real_points, tria, mapping);
    unsigned int i_eval_f          = 0;
    auto const   evaluate_function = [&](const ArrayView<Number> &values,
                                       const auto              &cell_data) {
      unsigned int              ii = i_eval_f;
      FEPointEvaluation<1, dim> fe_point(mapping, fe_p, update_values);
      std::vector<Number>       local_values;
      for (const auto cell : cell_data.cell_indices())
        {
          auto const cell_dofs =
            cell_data.get_active_cell_iterator(cell)->as_dof_handler_iterator(
              dof_handler_p);
          auto const unit_points = cell_data.get_unit_points(cell);
          auto const local_value = cell_data.get_data_view(cell, values);
          local_values.resize(cell_dofs->get_fe().n_dofs_per_cell());
          cell_dofs->get_dof_values(x.block(ii),
                                    local_values.begin(),
                                    local_values.end());

          fe_point.reinit(cell_dofs, unit_points);
          fe_point.evaluate(local_values, EvaluationFlags::values);

          for (unsigned int q = 0; q < unit_points.size(); ++q)
            local_value[q] = fe_point.get_value(q);
        }
    };


#ifdef DEBUG
    BlockVectorType exact(2);
    matrix->initialize_dof_vector(exact.block(0), 0);
    matrix->initialize_dof_vector(exact.block(1), 1);
#endif
    BlockVectorType numeric(2);
    matrix->initialize_dof_vector(numeric.block(0), 0);
    matrix->initialize_dof_vector(numeric.block(1), 1);

    unsigned int                 timestep_number = 0;
    ErrorCalculator<dim, Number> error_calculator_u(
      parameters.type,
      fe_degree,
      fe_degree,
      mapping,
      dof_handler_u,
      *exact_solution_u,
      evaluate_numerical_solution_u);
    ErrorCalculator<dim, Number> error_calculator_p(
      parameters.type,
      fe_degree,
      fe_degree,
      mapping,
      dof_handler_p,
      *exact_solution_p,
      evaluate_numerical_solution_p);

    std::vector<std::function<void(const double, VectorType &)>>
      integrate_rhs_function_{integrate_rhs_function_u,
                              integrate_rhs_function_p};
    std::unique_ptr<TimeIntegratorFO<dim, Number, Preconditioner, SystemN>>
      step = std::make_unique<
        TimeIntegratorFO<dim, Number, Preconditioner, SystemN>>(
        parameters.type,
        fe_degree,
        Alpha_1,
        Gamma_1,
        1.e-12,
        *matrix,
        *preconditioner,
        *rhs_matrix,
        integrate_rhs_function_,
        n_timesteps_at_once,
        parameters.extrapolate);

    // interpolate initial value

    evaluate_exact_solution_u(
      0, x.block(blk_slice.index(n_timesteps_at_once - 1, 0, nt_dofs - 1)));

    double           l2 = 0., l8 = -1., h1_semi = 0.;
    double           l2_p = 0., l8_p = -1., h1_semi_p = 0.;
    constexpr double qNaN           = std::numeric_limits<double>::quiet_NaN();
    bool const       st_convergence = parameters.space_time_conv_test;
    int              i = 0, total_gmres_iterations = 0;

    unsigned int samples_per_interval = (fe_degree + 1) * (fe_degree + 1);
    double       sample_step          = 1.0 / (samples_per_interval - 1);
    i_eval_f = blk_slice.index(n_timesteps_at_once - 1, 1, nt_dofs - 1);
    x.block(i_eval_f).update_ghost_values();
    std::vector<Number> output_point_evaluation =
      rpe.template evaluate_and_process<Number>(evaluate_function);
    x.block(i_eval_f).zero_out_ghost_values();
    if (!rpe.is_map_unique())
      {
        auto const         &point_indices = rpe.get_point_ptrs();
        std::vector<Number> new_output;
        new_output.reserve(point_indices.size() - 1);
        for (auto el : point_indices)
          if (el < output_point_evaluation.size())
            new_output.push_back(output_point_evaluation[el]);

        output_point_evaluation.swap(new_output);
      }

    std::vector<Number> prev_output_pt_eval = output_point_evaluation;
    FullMatrix<Number>  output_pt_eval(fe_degree + 1, real_points.size());
    FullMatrix<Number>  time_evaluator =
      get_time_evaluation_matrix<Number>(basis, samples_per_interval);
    FullMatrix<Number> output_pt_eval_res(samples_per_interval,
                                          real_points.size());
    auto const         do_point_evaluation = [&]() {
      Assert(output_pt_eval.n() >= prev_output_pt_eval.size(),
             ExcLowerRange(output_pt_eval.n(), prev_output_pt_eval.size()));
      for (unsigned int it = 0; it < n_timesteps_at_once; ++it)
        {
          if (is_cgp)
            std::copy_n(prev_output_pt_eval.begin(),
                        prev_output_pt_eval.size(),
                        output_pt_eval.begin(0));
          for (unsigned int t_dof = 0; t_dof < nt_dofs; ++t_dof)
            {
              i_eval_f = blk_slice.index(it, 1, t_dof);
              output_point_evaluation =
                rpe.template evaluate_and_process<Number>(evaluate_function);
              if (!rpe.is_map_unique())
                {
                  auto const         &point_indices = rpe.get_point_ptrs();
                  std::vector<Number> new_output;
                  new_output.reserve(point_indices.size() - 1);
                  for (auto el : point_indices)
                    if (el < output_point_evaluation.size())
                      new_output.push_back(output_point_evaluation[el]);

                  output_point_evaluation.swap(new_output);
                }
              Assert(output_pt_eval.m() > t_dof + is_cgp,
                     ExcLowerRange(output_pt_eval.m(), t_dof + is_cgp));
              std::copy_n(output_point_evaluation.begin(),
                          output_point_evaluation.size(),
                          output_pt_eval.begin(t_dof + is_cgp));
            }
          time_evaluator.mmult(output_pt_eval_res, output_pt_eval);
          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            {
              std::ofstream file(parameters.functional_file, std::ios::app);
              for (unsigned int row = 0; row < output_pt_eval_res.m(); ++row)
                {
                  double t_ = time + time_step_size * (it + row * sample_step);
                  file << std::setw(16) << std::scientific << t_;
                  for (unsigned int c = 0; c < output_pt_eval_res.n(); ++c)
                    file << std::setw(16) << std::scientific << " "
                         << output_pt_eval_res(row, c);
                  file << '\n';
                }
              file << '\n';
            }

          prev_output_pt_eval = output_point_evaluation;
        }
    };
    auto const data_output =
      [&](VectorType const &v, VectorType const &p, std::string const &name) {
        DataOut<dim> data_out;
        data_out.add_data_vector(dof_handler_u, v, "u");
        data_out.add_data_vector(dof_handler_p, p, "p");
        data_out.build_patches();
        data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));
        data_out.write_vtu_with_pvtu_record(
          "./", name, timestep_number, tria.get_communicator(), 4);
      };

    while (time < parameters.end_time)
      {
        TimerOutput::Scope scope(timer, "step");

        ++timestep_number;
        dealii::deallog << "Step " << timestep_number << " t = " << time
                        << std::endl;

        equ(prev_x,
            blk_slice.get_variable(x, n_timesteps_at_once - 1, nt_dofs - 1));
        step->solve(x, prev_x, timestep_number, time, time_step_size);
        total_gmres_iterations += step->last_step();

        for (unsigned int v = 0; v < 2; ++v)
          {
            auto var = blk_slice.get_time(x, v);
            for (auto &vec : var)
              constraints[v]->distribute(vec.get());
          }
        x.update_ghost_values();
        if (parameters.mean_pressure == true)
          {
            auto p_tc = blk_slice.get_time(x, 1);
            for (auto &p : p_tc)
              {
                double const mean_pressure = VectorTools::compute_mean_value(
                  mapping, dof_handler_p, quad_p, p.get(), 0);
                p.get().add(-mean_pressure);
              }
          }


        if (st_convergence)
          {
            auto error_on_In_u = error_calculator_u.evaluate_error(
              time, time_step_size, x, prev_x.block(0), n_timesteps_at_once);
            l2 += error_on_In_u[VectorTools::L2_norm];
            l8 = std::max(error_on_In_u[VectorTools::Linfty_norm], l8);
            h1_semi += error_on_In_u[VectorTools::H1_seminorm];

            auto error_on_In_p = error_calculator_p.evaluate_error(
              time, time_step_size, x, prev_x.block(1), n_timesteps_at_once);
            l2_p += error_on_In_p[VectorTools::L2_norm];
            l8_p = std::max(error_on_In_p[VectorTools::Linfty_norm], l8);
            h1_semi_p += error_on_In_p[VectorTools::H1_seminorm];
          }
        else
          do_point_evaluation();

        x.zero_out_ghost_values();
        time += n_timesteps_at_once * time_step_size;
        ++i;

        if (do_output)
          {
            numeric = 0.0;
            evaluate_numerical_solution_u(1.0,
                                          numeric.block(0),
                                          x,
                                          prev_x.block(1),
                                          (n_timesteps_at_once - 1) * nt_dofs);
            evaluate_numerical_solution_p(1.0,
                                          numeric.block(1),
                                          x,
                                          prev_x.block(1),
                                          (n_timesteps_at_once - 1) * nt_dofs);
            data_output(numeric.block(0), numeric.block(1), "solution");
          }
#ifdef DEBUG
        if (do_output && st_convergence)
          {
            exact = 0.0;
            evaluate_exact_solution_u(time, exact.block(0));
            evaluate_exact_solution_p(time, exact.block(1));
            data_output(exact.block(0), exact.block(1), "exact");
          }
#endif
      }
    double average_gmres_iter = static_cast<double>(total_gmres_iterations) /
                                static_cast<double>(timestep_number);
    pcout << "Average GMRES iterations " << average_gmres_iter << " ("
          << total_gmres_iterations << " gmres_iterations / " << timestep_number
          << " timesteps)\n"
          << std::endl;
    if (print_timing)
      timer.print_wall_time_statistics(MPI_COMM_WORLD);

    auto const   n_active_cells = tria.n_global_active_cells();
    size_t const n_dofs  = static_cast<size_t>(dof_handlers[0]->n_dofs() +
                                              dof_handlers[1]->n_dofs());
    size_t const st_dofs = i * n_dofs * n_blocks;
    size_t const work    = n_dofs * n_blocks * total_gmres_iterations;
    table.add_value("cells", n_active_cells);
    table.add_value("s-dofs", n_dofs);
    table.add_value("t-dofs", n_blocks);
    table.add_value("st-dofs", st_dofs);
    table.add_value("work", work);
    table.add_value("L\u221E-L\u221E(u)", st_convergence ? l8 : qNaN);
    table.add_value("L2-L2(u)", st_convergence ? std::sqrt(l2) : qNaN);
    table.add_value("L2-H1_semi(u)",
                    st_convergence ? std::sqrt(h1_semi) : qNaN);
    table.add_value("L\u221E-L\u221E(p)", st_convergence ? l8_p : qNaN);
    table.add_value("L2-L2(p)", st_convergence ? std::sqrt(l2_p) : qNaN);
    table.add_value("L2-H1_semi(p)",
                    st_convergence ? std::sqrt(h1_semi_p) : qNaN);
    itable.add_value(std::to_string(refinement), average_gmres_iter);
  };
  auto const [k, d_cyc, r_cyc, r] = std::visit(
    [](auto const &p) {
      return std::make_tuple(p.fe_degree,
                             p.n_deg_cycles,
                             p.n_ref_cycles,
                             p.refinement);
    },
    parameters);

  for (unsigned int j = k; j < k + d_cyc; ++j)
    {
      itable.add_value("k \\ r", j);
      for (unsigned int i = r; i < r + r_cyc; ++i)
        if (dim == 2)
          convergence_test(i, j, std::get<Parameters<2>>(parameters));
        else
          convergence_test(i, j, std::get<Parameters<3>>(parameters));

      table.set_precision("L\u221E-L\u221E(u)", 5);
      table.set_precision("L2-L2(u)", 5);
      table.set_precision("L2-H1_semi(u)", 5);
      table.set_precision("L\u221E-L\u221E(p)", 5);
      table.set_precision("L2-L2(p)", 5);
      table.set_precision("L2-H1_semi(p)", 5);
      table.set_scientific("L\u221E-L\u221E(u)", true);
      table.set_scientific("L2-L2(u)", true);
      table.set_scientific("L2-H1_semi(u)", true);
      table.set_scientific("L\u221E-L\u221E(p)", true);
      table.set_scientific("L2-L2(p)", true);
      table.set_scientific("L2-H1_semi(p)", true);
      table.evaluate_convergence_rates("L\u221E-L\u221E(u)",
                                       ConvergenceTable::reduction_rate_log2);
      table.evaluate_convergence_rates("L2-L2(u)",
                                       ConvergenceTable::reduction_rate_log2);
      table.evaluate_convergence_rates("L2-H1_semi(u)",
                                       ConvergenceTable::reduction_rate_log2);
      table.evaluate_convergence_rates("L\u221E-L\u221E(p)",
                                       ConvergenceTable::reduction_rate_log2);
      table.evaluate_convergence_rates("L2-L2(p)",
                                       ConvergenceTable::reduction_rate_log2);
      table.evaluate_convergence_rates("L2-H1_semi(p)",
                                       ConvergenceTable::reduction_rate_log2);
      pcout << "Convergence table k=" << j << std::endl;
      if (pcout.is_active())
        table.write_text(pcout.get_stream());
      pcout << std::endl;
      table.clear();
    }
  pcout << "Iteration count table\n";
  if (pcout.is_active())
    itable.write_text(pcout.get_stream());
  pcout << std::endl;
}



int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  dealii::ConditionalOStream       pcout(
    std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  MPI_Comm    comm               = MPI_COMM_WORLD;
  std::string file               = "default";
  std::string log_file_prefix    = "proc";
  int         dim                = 2;
  bool        precondition_float = true;
  {
    namespace arg_t = util::arg_type;
    util::cl_options clo(argc, argv);
    clo.insert(file, "file", arg_t::required, 'f', "Path to parameterfile");
    clo.insert(dim, "dim", arg_t::required, 'd', "Spatial dimensions");
    clo.insert(precondition_float, "precondition_float", arg_t::none, 'p');
    clo.insert(log_file_prefix,
               "log_prefix",
               util::arg_type::required,
               'l',
               "prefix of the log file, default is 'proc'");
  }
  std::string filename =
    log_file_prefix + file +
    std::to_string(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)) +
    ".log";
  std::ofstream pout(filename);
  dealii::deallog.attach(pout);
  dealii::deallog.depth_console(0);

  auto tst = [&](std::string file_name) {
    if (precondition_float)
      test<double, float>(pcout, comm, file_name, dim);
    else
      test<double, double>(pcout, comm, file_name, dim);
  };
  if (file == "default")
    {
      std::vector<std::pair<std::string, std::string>> tests = {
        {"Stokes\n", "tests/json/tfstokes.json"}};
      for (const auto &[header, file_name] : tests)
        {
          dealii::deallog << header;
          tst(file_name);
        }
    }
  else
    tst(file);

  dealii::deallog << std::endl;
  pcout << std::endl;
}
