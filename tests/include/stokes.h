// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by Nils Margenberg and Peter Munch

#pragma once

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
      if (component == 0 && x(0) < 1.e-10)
        {
          if (dim == 3)
            return -16. * u_max * factor * x(1) * (x(2) - 0.41 / 2.) *
                   (0.41 - x(1)) * (x(2) + 0.41 / 2.) / pow(0.41, 4);
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
} // namespace stokes
