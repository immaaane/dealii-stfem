// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by Nils Margenberg and Peter Munch

#pragma once

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/numerics/vector_tools.h>

#include "fe_time.h"
#include "types.h"

using namespace dealii;
using dealii::numbers::PI;

template <int dim, typename Number>
class ExactSolution : public Function<dim, Number>
{
public:
  ExactSolution(Number f_ = 1.0)
    : Function<dim, Number>()
    , f(f_)
  {}

  Number
  value(Point<dim> const &x, unsigned int const) const override final
  {
    Number value = sin(2 * PI * f * this->get_time());
    for (unsigned int i = 0; i < dim; ++i)
      value *= sin(2 * PI * f * x[i]);
    return value;
  }
  Tensor<1, dim, Number>
  gradient(const Point<dim> &x, const unsigned int) const override final
  {
    Tensor<1, dim, Number> grad;
    Number                 tv = 2 * PI * f * sin(2 * PI * f * this->get_time());
    for (unsigned int i = 0; i < dim; ++i)
      {
        grad[i] = tv;
        for (unsigned int j = 0; j < dim; ++j)
          grad[i] *= (i == j ? cos(2 * PI * f * x[j]) : sin(2 * PI * f * x[j]));
      }
    return grad;
  }

private:
  Number const f;
};

template <int dim, typename Number>
class RHSFunction : public Function<dim, Number>
{
public:
  RHSFunction(Number f_ = 1.0)
    : Function<dim, Number>()
    , f(f_)
  {}

  Number
  value(Point<dim> const &x, unsigned int const) const override final
  {
    Number value =
      dim * 4 * PI * PI * f * f * sin(2 * PI * f * this->get_time()) +
      2 * PI * f * cos(2 * PI * f * this->get_time());
    for (unsigned int i = 0; i < dim; ++i)
      value *= sin(2 * PI * f * x[i]);
    return value;
  }

private:
  Number const f;
};

template <int dim, typename Number>
class RHSFunction2 : public Function<dim, Number>
{
public:
  RHSFunction2()
    : Function<dim>()
  {}

  virtual Number
  value(const Point<dim> &p, const unsigned int) const override
  {
    const Number x = p[0];
    const Number y = dim >= 2 ? p[1] : 0.0;
    const Number z = dim >= 3 ? p[2] : 0.0;
    const Number t = this->get_time();

    if (dim == 2)
      return sin(2 * PI * x) * sin(2 * PI * y) *
             (PI * cos(PI * t) - 0.5 * (sin(PI * t) + 1) +
              (2 * 2 + 2 * 2) * PI * PI * (sin(PI * t) + 1)) *
             exp(-0.5 * t);
    else if (dim == 3)
      return sin(2 * PI * x) * sin(2 * PI * y) * sin(2 * PI * z) *
             (PI * cos(PI * t) - 0.5 * (sin(PI * t) + 1) +
              (2 * 2 + 2 * 2 + 2 * 2) * PI * PI * (sin(PI * t) + 1)) *
             exp(-0.5 * t);
    return 0.0;
  }
};

template <int dim, typename Number>
class ExactSolution2 : public Function<dim, Number>
{
public:
  ExactSolution2()
    : Function<dim>()
  {}

  virtual Number
  value(const Point<dim> &p, const unsigned int) const override
  {
    const Number x = p[0];
    const Number y = dim >= 2 ? p[1] : 0.0;
    const Number z = dim >= 3 ? p[2] : 0.0;
    const Number t = this->get_time();

    if (dim == 2)
      return sin(2 * PI * x) * sin(2 * PI * y) * (1 + sin(PI * t)) *
             exp(-0.5 * t);
    else if (dim == 3)
      return sin(2 * PI * x) * sin(2 * PI * y) * sin(2 * PI * z) *
             (1 + sin(PI * t)) * exp(-0.5 * t);

    Assert(false, ExcNotImplemented());

    return 0.0;
  }
};

// Analytic solutions for linear acoustic waves
namespace wave
{
  template <int dim, typename Number>
  using ExactSolution = ExactSolution<dim, Number>;

  template <int dim, typename Number>
  class ExactSolutionV : public Function<dim, Number>
  {
  public:
    ExactSolutionV(Number f_ = 1.0)
      : Function<dim, Number>()
      , f(f_)
    {}

    Number
    value(Point<dim> const &x, unsigned int const) const override final
    {
      Number value = 2 * PI * f * cos(2 * PI * f * this->get_time());
      for (unsigned int i = 0; i < dim; ++i)
        value *= sin(2 * PI * f * x[i]);
      return value;
    }

  private:
    Number const f;
  };

  template <int dim, typename Number>
  class RHSFunction : public Function<dim, Number>
  {
  public:
    RHSFunction(Number f_ = 1.0)
      : Function<dim, Number>()
      , f(f_)
    {}

    Number
    value(Point<dim> const &x, unsigned int const) const override final
    {
      Number value =
        pow(2.0, dim) * pow(PI * f, 2) * sin(2 * PI * f * this->get_time());
      for (unsigned int i = 0; i < dim; ++i)
        value *= sin(2 * PI * f * x[i]);
      return value;
    }

  private:
    Number const f;
  };

} // namespace wave

namespace stokes
{
  template <int dim, typename Number>
  class ExactSolutionU : public Function<dim, Number>
  {
  public:
    ExactSolutionU()
      : Function<dim, Number>(2)
    {}

    Number
    value(Point<dim> const  &x,
          unsigned int const component) const override final
    {
      Number sin_t    = sin(this->get_time());
      Number sin_PI_x = sin(PI * x(0)), sin_PI_y = sin(PI * x(1));
      Number cos_PI_x = cos(PI * x(0)), cos_PI_y = cos(PI * x(1));
      if (component == 0)
        return cos_PI_y * sin_t * sin_PI_x * sin_PI_x * sin_PI_y;
      if (component == 1)
        return -cos_PI_x * sin_t * sin_PI_x * sin_PI_y * sin_PI_y;
      return 0.0;
    }
    Tensor<1, dim, Number>
    gradient(const Point<dim>  &x,
             const unsigned int component) const override final
    {
      Tensor<1, dim, Number> grad;
      Number                 sin_t = sin(this->get_time());
      Number sin_PI_x = sin(PI * x(0)), sin_PI_y = sin(PI * x(1));
      Number cos_PI_x = cos(PI * x(0)), cos_PI_y = cos(PI * x(1));
      Number PI_sin_t = PI * sin_t;
      if (component == 0)
        {
          grad[0] = 2 * PI_sin_t * cos_PI_x * sin_PI_x * cos_PI_y * sin_PI_y;
          grad[1] = PI_sin_t * (sin_PI_x * sin_PI_x * cos_PI_y * cos_PI_y -
                                sin_PI_x * sin_PI_x * sin_PI_y * sin_PI_y);
        }
      else if (component == 1)
        {
          grad[0] = PI_sin_t * (sin_PI_x * sin_PI_x - cos_PI_x * cos_PI_x) *
                    sin_PI_y * sin_PI_y;
          grad[1] = -2 * PI_sin_t * cos_PI_x * sin_PI_x * cos_PI_y * sin_PI_y;
        }
      return grad;
    }
  };


  template <int dim, typename Number>
  class ExactSolutionP : public Function<dim, Number>
  {
  public:
    ExactSolutionP()
      : Function<dim, Number>()
    {}

    Number
    value(Point<dim> const &x, unsigned int const) const override final
    {
      Number sin_t    = sin(this->get_time());
      Number sin_PI_x = sin(PI * x(0)), sin_PI_y = sin(PI * x(1)),
             cos_PI_x = cos(PI * x(0)), cos_PI_y = cos(PI * x(1));
      return cos_PI_x * cos_PI_y * sin_t * sin_PI_x * sin_PI_y;
    }
    Tensor<1, dim, Number>
    gradient(const Point<dim> &x, const unsigned int) const override final
    {
      Tensor<1, dim, Number> grad;
      Number sin_PI_x = sin(PI * x(0)), sin_PI_y = sin(PI * x(1)),
             cos_PI_x = cos(PI * x(0)), cos_PI_y = cos(PI * x(1)),
             PI_sin_t = PI * sin(this->get_time());
      grad[0]         = PI_sin_t * (cos_PI_x * cos_PI_x - sin_PI_x * sin_PI_x) *
                cos_PI_y * sin_PI_y;
      grad[1] = PI_sin_t * (cos_PI_y * cos_PI_y - sin_PI_y * sin_PI_y) *
                cos_PI_x * sin_PI_x;
      return grad;
    }
  };

  template <int dim, typename Number>
  class RHSFunction : public Function<dim, Number>
  {
  public:
    RHSFunction(Number viscosity_, bool navier = false)
      : Function<dim, Number>(2)
      , viscosity(viscosity_)
      , nonlinear_factor(navier ? 1.0 : 0.0)
    {}

    Number
    value(Point<dim> const  &x,
          unsigned int const component) const override final
    {
      Number sin_t    = sin(this->get_time());
      Number cos_t    = cos(this->get_time());
      Number sin_PI_x = sin(PI * x(0));
      Number sin_PI_y = sin(PI * x(1));
      Number cos_PI_x = cos(PI * x(0));
      Number cos_PI_y = cos(PI * x(1));

      if (component == 0)
        return sin_PI_y *
               (PI * (1.0 - 2.0 * PI * viscosity) * cos_PI_x * cos_PI_x *
                  cos_PI_y * sin_t +
                cos_PI_y *
                  (cos_t + PI * (-1.0 + 6.0 * PI * viscosity) * sin_t) *
                  sin_PI_x * sin_PI_x +
                nonlinear_factor * PI * cos_PI_x * sin_t * sin_t * sin_PI_x *
                  sin_PI_x * sin_PI_x * sin_PI_y);
      else if (component == 1)
        return sin_PI_x * (nonlinear_factor * PI * cos_PI_y * sin_t * sin_t *
                             sin_PI_x * sin_PI_y * sin_PI_y * sin_PI_y +
                           cos_PI_x * (PI *
                                         (-2.0 * PI * viscosity +
                                          (1.0 + 4.0 * PI * viscosity) *
                                            cos(2.0 * PI * x(1))) *
                                         sin_t -
                                       cos_t * sin_PI_y * sin_PI_y));
      return 0;
    }

  private:
    Number viscosity        = 1.;
    Number nonlinear_factor = 1.;
  };
} // namespace stokes


namespace stokespoly
{
  template <int dim, typename Number>
  class ExactSolutionU : public Function<dim, Number>
  {
  public:
    ExactSolutionU()
      : Function<dim, Number>(2)
    {}

    Number
    value(Point<dim> const  &x,
          unsigned int const component) const override final
    {
      using std::pow;
      const Number t = this->get_time();
      if (component == 0)
        return 1000 * t * pow(x(0), 2) * pow(1 - x(0), 4) * pow(x(1), 2) *
               (1 - x(1)) * (3 - 5 * x(1));
      if (component == 1)
        return 1000 * t * -2 * x(0) * pow(1 - x(0), 3) * (1 - 3 * x(0)) *
               pow(x(1), 3) * pow(1 - x(1), 2);
      return 0.0;
    }
    Tensor<1, dim, Number>
    gradient(const Point<dim>  &x,
             const unsigned int component) const override final
    {
      Tensor<1, dim, Number> grad;
      using std::pow;
      const Number t = this->get_time();
      if (component == 0)
        {
          grad[0] = 2000 * t * pow(1 - x(0), 4) * x(0) * (3 - 5 * x(1)) *
                      (1 - x(1)) * pow(x(1), 2) -
                    4000 * t * pow(1 - x(0), 3) * pow(x(0), 2) *
                      (3 - 5 * x(1)) * (1 - x(1)) * pow(x(1), 2);
          grad[1] = 2000 * t * pow(1 - x(0), 4) * pow(x(0), 2) *
                      (3 - 5 * x(1)) * (1 - x(1)) * x(1) -
                    1000 * t * pow(1 - x(0), 4) * pow(x(0), 2) *
                      (3 - 5 * x(1)) * pow(x(1), 2) -
                    5000 * t * pow(1 - x(0), 4) * pow(x(0), 2) * (1 - x(1)) *
                      pow(x(1), 2);
        }
      else if (component == 1)
        {
          grad[0] = -2000 * t * (1 - 3 * x(0)) * pow(1 - x(0), 3) *
                      pow(1 - x(1), 2) * pow(x(1), 3) +
                    6000 * t * (1 - 3 * x(0)) * pow(1 - x(0), 2) * x(0) *
                      pow(1 - x(1), 2) * pow(x(1), 3) +
                    6000 * t * pow(1 - x(0), 3) * x(0) * pow(1 - x(1), 2) *
                      pow(x(1), 3);
          grad[1] = -6000 * t * (1 - 3 * x(0)) * pow(1 - x(0), 3) * x(0) *
                      pow(1 - x(1), 2) * pow(x(1), 2) +
                    4000 * t * (1 - 3 * x(0)) * pow(1 - x(0), 3) * x(0) *
                      (1 - x(1)) * pow(x(1), 3);
        }
      return grad;
    }
  };


  template <int dim, typename Number>
  class ExactSolutionP : public Function<dim, Number>
  {
  public:
    ExactSolutionP()
      : Function<dim, Number>()
    {}

    Number
    value(Point<dim> const &x, unsigned int const) const override final
    {
      const Number t = this->get_time();
      return t * (-4 * 4 * x(0) * (1 - x(0)) * x(1) * (1 - x(1)) + 4. / 9.);
    }
    Tensor<1, dim, Number>
    gradient(const Point<dim> &x, const unsigned int) const override final
    {
      const Number           t = this->get_time();
      Tensor<1, dim, Number> grad;
      grad[0] = -16 * t * (1 - 2 * x(0)) * x(1) * (1 - x(1));
      grad[1] = -16 * t * x(0) * (1 - x(0)) * (1 - 2 * x(1));
      return grad;
    }
  };

  template <int dim, typename Number>
  class RHSFunction : public Function<dim, Number>
  {
  public:
    RHSFunction(Number viscosity_, bool)
      : Function<dim, Number>(2)
      , viscosity(viscosity_)
    {}

    Number
    value(Point<dim> const  &x,
          unsigned int const component) const override final
    {
      const Number t = this->get_time();
      if (component == 0)
        return (5000 * pow(x(0), 6) - 20000 * pow(x(0), 5) +
                (30000 - 150000 * viscosity * t) * pow(x(0), 4) +
                (400000 * viscosity * t - 20000) * pow(x(0), 3) +
                (5000 - 360000 * viscosity * t) * pow(x(0), 2) +
                120000 * viscosity * t * x(0) - 10000 * viscosity * t) *
                 pow(x(1), 4) +
               (-(8000 * pow(x(0), 6)) + 32000 * pow(x(0), 5) +
                (240000 * viscosity * t - 48000) * pow(x(0), 4) +
                (32000 - 640000 * viscosity * t) * pow(x(0), 3) +
                (576000 * viscosity * t - 8000) * pow(x(0), 2) -
                192000 * viscosity * t * x(0) + 16000 * viscosity * t) *
                 pow(x(1), 3) +
               ((3000 - 60000 * viscosity * t) * pow(x(0), 6) +
                (240000 * viscosity * t - 12000) * pow(x(0), 5) +
                (18000 - 450000 * viscosity * t) * pow(x(0), 4) +
                (480000 * viscosity * t - 12000) * pow(x(0), 3) +
                (3000 - 276000 * viscosity * t) * pow(x(0), 2) +
                (72000 * viscosity - 32) * t * x(0) +
                (16 - 6000 * viscosity) * t) *
                 pow(x(1), 2) +
               (48000 * viscosity * t * pow(x(0), 6) -
                192000 * viscosity * t * pow(x(0), 5) +
                288000 * viscosity * t * pow(x(0), 4) -
                192000 * viscosity * t * pow(x(0), 3) +
                48000 * viscosity * t * pow(x(0), 2) + 32 * t * x(0) - 16 * t) *
                 x(1) -
               6000 * viscosity * t * pow(x(0), 6) +
               24000 * viscosity * t * pow(x(0), 5) -
               36000 * viscosity * t * pow(x(0), 4) +
               24000 * viscosity * t * pow(x(0), 3) -
               6000 * viscosity * t * pow(x(0), 2);
      else if (component == 1)
        return (-(6000 * pow(x(0), 5)) + 20000 * pow(x(0), 4) +
                (120000 * viscosity * t - 24000) * pow(x(0), 3) +
                (12000 - 240000 * viscosity * t) * pow(x(0), 2) +
                (144000 * viscosity * t - 2000) * x(0) -
                24000 * viscosity * t) *
                 pow(x(1), 5) +
               (12000 * pow(x(0), 5) - 40000 * pow(x(0), 4) +
                (48000 - 240000 * viscosity * t) * pow(x(0), 3) +
                (480000 * viscosity * t - 24000) * pow(x(0), 2) +
                (4000 - 288000 * viscosity * t) * x(0) +
                48000 * viscosity * t) *
                 pow(x(1), 4) +
               ((120000 * viscosity * t - 6000) * pow(x(0), 5) +
                (20000 - 400000 * viscosity * t) * pow(x(0), 4) +
                (600000 * viscosity * t - 24000) * pow(x(0), 3) +
                (12000 - 480000 * viscosity * t) * pow(x(0), 2) +
                (184000 * viscosity * t - 2000) * x(0) -
                24000 * viscosity * t) *
                 pow(x(1), 3) +
               (-(144000 * viscosity * t * pow(x(0), 5)) +
                480000 * viscosity * t * pow(x(0), 4) -
                576000 * viscosity * t * pow(x(0), 3) +
                288000 * viscosity * t * pow(x(0), 2) -
                48000 * viscosity * t * x(0)) *
                 pow(x(1), 2) +
               (36000 * viscosity * t * pow(x(0), 5) -
                120000 * viscosity * t * pow(x(0), 4) +
                144000 * viscosity * t * pow(x(0), 3) +
                (-(72000 * viscosity) - 32) * t * pow(x(0), 2) +
                (12000 * viscosity + 32) * t * x(0)) *
                 x(1) +
               16 * t * pow(x(0), 2) - 16 * t * x(0);
      return 0;
    }

  private:
    Number viscosity = 1.;
  };
} // namespace stokespoly

template <int dim, typename Number>
class ErrorCalculator
{
public:
  using VectorType      = VectorT<Number>;
  using BlockVectorType = BlockVectorT<Number>;

  ErrorCalculator(
    TimeStepType                      type_,
    unsigned int                      time_degree,
    unsigned int                      space_degree,
    Mapping<dim> const               &mapping_,
    DoFHandler<dim> const            &dof_handler_,
    Function<dim, Number>            &exact_solution_,
    std::function<void(const double,
                       VectorType &,
                       BlockVectorType const &,
                       VectorType const &,
                       unsigned int)> evaluate_numerical_solution_)
    : time_step_type(type_)
    , nt_dofs((time_step_type == TimeStepType::DG) ? time_degree + 1 :
                                                     time_degree)
    , quad_cell(space_degree + 1)
    , quad_time(time_degree + 1)
    , mapping(mapping_)
    , dof_handler(dof_handler_)
    , exact_solution(exact_solution_)
    , evaluate_numerical_solution(evaluate_numerical_solution_)
  {}

  std::unordered_map<VectorTools::NormType, double>
  evaluate_error(const double           time,
                 const double           time_step,
                 BlockVectorType const &x,
                 VectorType const      &prev_x,
                 unsigned int           n_time_steps_at_once,
                 bool                   calculate_divergence = false) const
  {
    std::unordered_map<VectorTools::NormType, double> error{
      {VectorTools::L2_norm, 0.0},
      {VectorTools::Linfty_norm, -1.0},
      {VectorTools::H1_seminorm, 0.0}};
    if (calculate_divergence)
      error.emplace(VectorTools::Hdiv_seminorm, 0.0);
    auto const &tq = quad_time.get_points();
    auto const &tw = quad_time.get_weights();

    VectorType     numeric(prev_x);
    Vector<double> differences_per_cell(
      dof_handler.get_triangulation().n_active_cells());
    for (unsigned int i = 0; i < x.n_blocks(); ++i)
      x.block(i).update_ghost_values();
    prev_x.update_ghost_values();
    double time_;
    for (unsigned int it = 0; it < n_time_steps_at_once; ++it)
      for (unsigned q = 0; q < quad_time.size(); ++q)
        {
          time_ = time + time_step * it + tq[q][0] * time_step;
          exact_solution.set_time(time_);
          auto const &current_prev_x =
            it == 0 ? prev_x : x.block(nt_dofs * it - 1);
          evaluate_numerical_solution(
            tq[q][0], numeric, x, current_prev_x, nt_dofs * it);

          numeric.update_ghost_values();
          dealii::VectorTools::integrate_difference(
            mapping,
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
          if (calculate_divergence)
            {
              dealii::VectorTools::integrate_difference(
                mapping,
                dof_handler,
                numeric,
                exact_solution,
                differences_per_cell,
                quad_cell,
                dealii::VectorTools::Hdiv_seminorm);
              double hdiv = dealii::VectorTools::compute_global_error(
                dof_handler.get_triangulation(),
                differences_per_cell,
                dealii::VectorTools::Hdiv_seminorm);
              error[VectorTools::Hdiv_seminorm] +=
                time_step * tw[q] * hdiv * hdiv;
            }
          numeric.zero_out_ghost_values();
        }
    for (unsigned int i = 0; i < x.n_blocks(); ++i)
      x.block(i).zero_out_ghost_values();
    prev_x.zero_out_ghost_values();
    return error;
  }

private:
  TimeStepType           time_step_type;
  unsigned int           nt_dofs;
  QGauss<dim> const      quad_cell;
  QGauss<1> const        quad_time;
  const Mapping<dim>    &mapping;
  const DoFHandler<dim> &dof_handler;
  Function<dim, Number> &exact_solution;
  std::function<void(const double,
                     VectorType &,
                     BlockVectorType const &,
                     VectorType const &,
                     unsigned int)>
    evaluate_numerical_solution;
};
