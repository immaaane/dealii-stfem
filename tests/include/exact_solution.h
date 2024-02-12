#pragma once

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include "fe_time.h"

using namespace dealii;
using dealii::numbers::PI;

using Number          = double;
using VectorType      = LinearAlgebra::distributed::Vector<double>;
using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

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
    double         tv = 2 * PI * f * sin(2 * PI * f * this->get_time());
    for (unsigned int i = 0; i < dim; ++i)
      {
        grad[i] = tv;
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
    Number value = 8 * PI * PI * f * f * sin(2 * PI * f * this->get_time()) +
                   2 * PI * f * cos(2 * PI * f * this->get_time());
    for (unsigned int i = 0; i < dim; ++i)
      value *= sin(2 * PI * f * x[i]);
    return value;
  }

private:
  double const f;
};

template <int dim, typename Number>
class RHSFunction2 : public Function<dim, Number>
{
public:
  RHSFunction2()
    : Function<dim>()
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int) const override
  {
    const double x = p[0];
    const double y = dim >= 2 ? p[1] : 0.0;
    const double z = dim >= 3 ? p[2] : 0.0;
    const double t = this->get_time();

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

  virtual double
  value(const Point<dim> &p, const unsigned int) const override
  {
    const double x = p[0];
    const double y = dim >= 2 ? p[1] : 0.0;
    const double z = dim >= 3 ? p[2] : 0.0;
    const double t = this->get_time();

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
    ExactSolutionV(double f_ = 1.0)
      : Function<dim, Number>()
      , f(f_)
    {}

    double
    value(Point<dim> const &x, unsigned int const) const override final
    {
      double value = 2 * PI * f * cos(2 * PI * f * this->get_time());
      for (unsigned int i = 0; i < dim; ++i)
        value *= sin(2 * PI * f * x[i]);
      return value;
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
      double value =
        pow(2.0, dim) * pow(PI * f, 2) * sin(2 * PI * f * this->get_time());
      for (unsigned int i = 0; i < dim; ++i)
        value *= sin(2 * PI * f * x[i]);
      return value;
    }

  private:
    double const f;
  };

} // namespace wave

template <int dim, typename Number>
class ErrorCalculator
{
public:
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
                 unsigned int           n_time_steps_at_once) const
  {
    std::unordered_map<VectorTools::NormType, double> error{
      {VectorTools::L2_norm, 0.0},
      {VectorTools::Linfty_norm, -1.0},
      {VectorTools::H1_seminorm, 0.0}};
    auto const &tq = quad_time.get_points();
    auto const &tw = quad_time.get_weights();

    VectorType     numeric(prev_x);
    Vector<double> differences_per_cell(
      dof_handler.get_triangulation().n_active_cells());
    for (unsigned int i = 0; i < x.n_blocks(); ++i)
      x.block(i).update_ghost_values();
    prev_x.update_ghost_values();
    auto nt_dofs =
      static_cast<unsigned int>(x.n_blocks() / n_time_steps_at_once);
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
          numeric.zero_out_ghost_values();
        }
    for (unsigned int i = 0; i < x.n_blocks(); ++i)
      x.block(i).zero_out_ghost_values();
    prev_x.zero_out_ghost_values();
    return error;
  }

private:
  TimeStepType           time_step_type;
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
