#include <deal.II/grid/grid_generator.h>

#include "stmg.h"
#include "stokes.h"
#include "types.h"
namespace dealii
{
  template <typename Number, int dim>
  void
  setup_constraints_up(dealii::AffineConstraints<Number> &constraints_u,
                       dealii::AffineConstraints<Number> &constraints_p,
                       DoFHandler<dim> const             &dof_handler_u,
                       DoFHandler<dim> const             &dof_handler_p,
                       Parameters<dim> const             &parameters,
                       stokes::Parameters const          &stokes_parameters)
  {
    DoFTools::make_hanging_node_constraints(dof_handler_u, constraints_u);
    if (parameters.space_time_conv_test)
      {
        if (!parameters.nitsche_boundary)
          DoFTools::make_zero_boundary_constraints(dof_handler_u,
                                                   constraints_u);
      }
    else
      {
        DoFTools::make_zero_boundary_constraints(dof_handler_u,
                                                 3,
                                                 constraints_u);
        DoFTools::make_zero_boundary_constraints(dof_handler_u,
                                                 2,
                                                 constraints_u);
        if (stokes_parameters.dfg_benchmark == 0)
          {
            DoFTools::make_zero_boundary_constraints(dof_handler_u,
                                                     0,
                                                     constraints_u);
            if (dim == 3)
              {
                DoFTools::make_zero_boundary_constraints(dof_handler_u,
                                                         4,
                                                         constraints_u);
                DoFTools::make_zero_boundary_constraints(dof_handler_u,
                                                         5,
                                                         constraints_u);
              }
          }
      }
    DoFTools::make_hanging_node_constraints(dof_handler_p, constraints_p);
  }

  template <typename Number, int dim>
  std::pair<std::vector<std::unique_ptr<Function<dim, Number>>>,
            std::map<types::boundary_id, dealii::Function<dim> *>>
  get_dirichlet_function(Parameters<dim> const    &parameters,
                         stokes::Parameters const &stokes_parameters)
  {
    std::pair<std::vector<std::unique_ptr<Function<dim, Number>>>,
              std::map<types::boundary_id, dealii::Function<dim> *>>
      ret;

    if (parameters.problem == ProblemType::stokes)
      {
        ret.first.reserve(1);
        if (!parameters.space_time_conv_test)
          {
            if (stokes_parameters.dfg_benchmark != 0)
              {
                ret.first.emplace_back(
                  std::make_unique<stokes::InflowDfg<dim, Number>>(
                    stokes_parameters));
                ret.second.emplace(0, ret.first.front().get());
              }
            else
              {
                ret.first.emplace_back(
                  std::make_unique<stokes::LidDriven<dim, Number>>(
                    stokes_parameters));
                ret.second.emplace(1, ret.first.front().get());
              }
          }
        else
          {
            ret.first.emplace_back(
              std::make_unique<Functions::ZeroFunction<dim, Number>>(dim));
            ret.second.emplace(0, ret.first.front().get());
          }
      }

    return ret;
  }

  template <int dim>
  void
  setup_triangulation(Triangulation<dim>    &tria,
                      Parameters<dim> const &parameters)
  {
    // for wave and heat and stokes convergence
    if (parameters.grid_descriptor == "hyperRectangle")
      GridGenerator::subdivided_hyper_rectangle(
        tria,
        parameters.subdivisions,
        parameters.hyperrect_lower_left,
        parameters.hyperrect_upper_right,
        parameters.colorize_boundary);
    else if (parameters.grid_descriptor == "dfgBenchmark")
      {
        GridGenerator::channel_with_cylinder(tria, 0.045, 3, 1.0, true);
      }
    else if (parameters.grid_descriptor == "bendingPipe")
      {
      }
  }
} // namespace dealii
