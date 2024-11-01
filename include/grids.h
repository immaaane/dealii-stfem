#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

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
        if constexpr (dim == 2)
          GridGenerator::channel_with_cylinder(tria, 0.025, 1, 1.0, true);
        else
          {
            dealii::Triangulation<3> tmp;
            GridGenerator::channel_with_cylinder(tmp, 0.025, 1, 1.0, true);
            tmp.reset_all_manifolds();
            Tensor<1, 3> shift;
            shift[0] = 0.3;
            GridTools::shift(shift, tmp);
            Triangulation<3> front_tria;
            GridGenerator::subdivided_hyper_rectangle(front_tria,
                                                      {3u, 4u, 4u},
                                                      Point<3>(0.0, 0.0, 0.0),
                                                      Point<3>(0.3,
                                                               0.41,
                                                               0.41));
            GridGenerator::merge_triangulations(
              front_tria, tmp, tria, 1.e-3, true, false);
            const Point<3>     axial_point(0.5, 0.2, 0.0);
            const Tensor<1, 3> direction{{0.0, 0.0, 1.0}};
            tria.set_manifold(cylindrical_manifold_id, FlatManifold<3>());
            tria.set_manifold(tfi_manifold_id, FlatManifold<3>());
            const CylindricalManifold<3>        cylindrical_manifold(direction,
                                                              axial_point);
            TransfiniteInterpolationManifold<3> inner_manifold;
            inner_manifold.initialize(tria);
            tria.set_manifold(cylindrical_manifold_id, cylindrical_manifold);
            tria.set_manifold(tfi_manifold_id, inner_manifold);

            for (const auto &face : tria.active_face_iterators())
              if (face->at_boundary())
                {
                  auto const &center = face->center();
                  if (std::abs(center[0] - 0.0) < 1e-8)
                    face->set_boundary_id(0);
                  else if (std::abs(center[0] - 2.5) < 1e-8)
                    face->set_boundary_id(1);
                  else if (face->manifold_id() == cylindrical_manifold_id)
                    face->set_boundary_id(2);
                  else
                    face->set_boundary_id(3);
                }
          }
      }
    else if (parameters.grid_descriptor == "bendingPipe")
      {
      }
  }
} // namespace dealii
