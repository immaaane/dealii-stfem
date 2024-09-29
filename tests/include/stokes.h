// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by Nils Margenberg and Peter Munch

#pragma once

#include "compute_block_matrix.h"

namespace stokes
{
  using namespace dealii;
  struct Parameters
  {
    bool         compute_drag_lift       = true;
    double       rho                     = 1.0;
    double       characteristic_diameter = 0.1;
    double       u_mean                  = 1.0;
    double       viscosity               = 1.0;
    double       penalty1                = 20;
    double       penalty2                = 10;
    bool         mean_pressure           = true;
    bool         dg_pressure             = true;
    unsigned int dirichlet_boundary      = 1;
    unsigned int dfg_benchmark           = 0;
    double       height                  = 0.41;

    void
    parse(const std::string file_name)
    {
      ParameterHandler prm;
      prm.add_parameter("computeDragLift", compute_drag_lift);
      prm.add_parameter("rho", rho);
      prm.add_parameter("characteristicDiam", characteristic_diameter);
      prm.add_parameter("uMean", u_mean);
      prm.add_parameter("viscosity", viscosity);
      prm.add_parameter("penalty1", penalty1);
      prm.add_parameter("penalty2", penalty2);
      prm.add_parameter("meanPressure", mean_pressure);
      prm.add_parameter("dGPressure", dg_pressure);
      prm.add_parameter("dfgBenchmark", dfg_benchmark);
      std::ifstream file;
      file.open(file_name);
      prm.parse_input_from_json(file, true);
    }
  };

  static constexpr double dirichlet_factor = 10;
  template <int dim, typename Number>
  struct InflowDfg : public Function<dim, Number>
  {
    InflowDfg(stokes::Parameters const &stokes_parameters)
      : Function<dim, Number>(dim)
      , is_dfg3(stokes_parameters.dfg_benchmark == 3)
      , u_max(stokes_parameters.u_mean * (dim == 2 ? 1.5 : 2.25))
    {}

    Number
    value(Point<dim> const  &x,
          unsigned int const component) const override final
    {
      auto t = this->get_time();
      using dealii::numbers::PI;
      auto const factor = is_dfg3 ?
                            sin(PI * t / 8.0) :
                            ((t < 1. / dirichlet_factor) ?
                               0.5 - 0.5 * cos(dirichlet_factor * PI * t) :
                               1.0);
      if (component == 0)
        {
          if (dim == 3)
            return 16. * u_max * factor * x(2) * (x(2) - 0.41) * x(1) *
                   (x(1) - 0.41) / pow(0.41, 4);
          else
            return ((4. * u_max * factor * x(1) * (0.41 - x(1))) /
                    (pow(0.41, 2)));
        }
      return 0.;
    }

  private:
    bool   is_dfg3;
    double u_max;
  };

  template <int dim, typename Number>
  struct LidDriven : public Function<dim, Number>
  {
    LidDriven(stokes::Parameters const &stokes_parameters)
      : Function<dim, Number>(dim)
      , u_max(stokes_parameters.u_mean)
    {}

    Number
    value(Point<dim> const &, unsigned int const component) const override final
    {
      auto       t      = this->get_time();
      auto const factor = (t < 1. / dirichlet_factor) ?
                            0.5 - 0.5 * cos(dirichlet_factor * PI * t) :
                            1.0;
      if (component == 1)
        return factor * u_max;
      return 0.;
    }

  private:
    double u_max;
  };

  template <int dim, typename Number>
  auto
  get_sparsity_pattern(const dealii::DoFHandler<dim> &dof_handler_u_,
                       const dealii::DoFHandler<dim> &dof_handler_p_,
                       const dealii::IndexSet        &locally_relevant_dofs_u_,
                       const dealii::IndexSet        &locally_relevant_dofs_p_,
                       const AffineConstraints<Number> &constraints_u,
                       const AffineConstraints<Number> &constraints_p)
  {
    auto sparsity_pattern = std::make_shared<BlockSparsityPatternType>();
    // Set up the row and column partitioning for the block system
    std::vector<dealii::IndexSet> row_partitioning{
      dof_handler_u_.locally_owned_dofs(), dof_handler_p_.locally_owned_dofs()};

    std::vector<dealii::IndexSet> column_partitioning{
      dof_handler_u_.locally_owned_dofs(), dof_handler_p_.locally_owned_dofs()};

    // Define the writable rows (locally relevant degrees of freedom)
    std::vector<dealii::IndexSet> writeable_rows{locally_relevant_dofs_u_,
                                                 locally_relevant_dofs_p_};

    // Get the MPI subdomain
    auto const subdomain = dealii::Utilities::MPI::this_mpi_process(
      dof_handler_u_.get_communicator());

    // Reinitialize the sparsity pattern
    sparsity_pattern->reinit(row_partitioning,
                             column_partitioning,
                             writeable_rows,
                             dof_handler_u_.get_communicator());

    // Create the block sparsity pattern for each block
    DoFTools::make_block_sparsity_pattern_block(dof_handler_u_,
                                                dof_handler_u_,
                                                sparsity_pattern->block(0, 0),
                                                constraints_u,
                                                constraints_u,
                                                true,
                                                subdomain);

    DoFTools::make_block_sparsity_pattern_block(dof_handler_u_,
                                                dof_handler_p_,
                                                sparsity_pattern->block(0, 1),
                                                constraints_u,
                                                constraints_p,
                                                true,
                                                subdomain);

    DoFTools::make_block_sparsity_pattern_block(dof_handler_p_,
                                                dof_handler_u_,
                                                sparsity_pattern->block(1, 0),
                                                constraints_p,
                                                constraints_u,
                                                true,
                                                subdomain);
    // Compress the sparsity pattern
    sparsity_pattern->compress();
    return sparsity_pattern;
  }

} // namespace stokes
