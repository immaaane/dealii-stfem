#pragma once

#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>

#include "gmg.h"
#include "operators.h"
#include "types.h"

namespace dealii
{
  /** Time stepping by DG and CGP variational time discretizations
   *
   * This time integrator is suited for linear problems. For nonlinear problems
   * we would need a few extensions in order to integrate nonlinear terms
   * accurately.
   */
  template <int dim, typename Number, typename NumberPreconditioner>
  class TimeIntegrator
  {
  public:
    using GMGPreconditioner =
      GMG<dim,
          NumberPreconditioner,
          SystemMatrix<NumberPreconditioner,
                       MatrixFreeOperator<dim, NumberPreconditioner>>>;
    using VectorType      = VectorT<Number>;
    using BlockVectorType = BlockVectorT<Number>;

    TimeIntegrator(
      TimeStepType              type_,
      unsigned int              time_degree_,
      FullMatrix<Number> const &Alpha_,
      FullMatrix<Number> const &Gamma_,
      double const              gmres_tolerance_,
      SystemMatrix<Number, MatrixFreeOperator<dim, Number>> const &matrix_,
      GMGPreconditioner const &preconditioner_,
      SystemMatrix<Number, MatrixFreeOperator<dim, Number>> const &rhs_matrix_,
      std::function<void(const double, VectorType &)> integrate_rhs_function,
      unsigned int                                    n_timesteps_at_once_)
      : type(type_)
      , time_degree(time_degree_)
      , Alpha(Alpha_)
      , Gamma(Gamma_)
      , solver_control(200, 1.e-12, gmres_tolerance_, false, true)
      , solver(solver_control)
      , preconditioner(preconditioner_)
      , matrix(matrix_)
      , rhs_matrix(rhs_matrix_)
      , integrate_rhs_function(integrate_rhs_function)
      , n_timesteps_at_once(n_timesteps_at_once_)
    {
      if (type == TimeStepType::DG)
        quad_time =
          QGaussRadau<1>(time_degree + 1, QGaussRadau<1>::EndPoint::right);
      else if (type == TimeStepType::CGP)
        quad_time = QGaussLobatto<1>(time_degree + 1);
    }

    void
    assemble_force(BlockVectorType &rhs,
                   double const     time,
                   double const     time_step) const
    {
      VectorType tmp;
      matrix.initialize_dof_vector(tmp);

      for (unsigned int it = 0; it < n_timesteps_at_once; ++it)
        for (unsigned int j = 0; j < quad_time.size(); ++j)
          {
            double time_ =
              time + time_step * it + time_step * quad_time.point(j)[0];
            integrate_rhs_function(time_, tmp);
            /// Here we exploit that Alpha is a diagonal matrix
            if (type == TimeStepType::DG)
              rhs.block(j + it * Alpha.m()).add(Alpha(j, j), tmp);
            else
              {
                if (j == 0)
                  for (unsigned int i = 0; i < Gamma.m(); ++i)
                    rhs.block(i + it * Alpha.m()).add(-Gamma(i, 0), tmp);
                else
                  rhs.block(j - 1 + it * Alpha.m())
                    .add(Alpha(j - 1, j - 1), tmp);
              }
          }
    }

  protected:
    TimeStepType              type;
    unsigned int              time_degree;
    Quadrature<1>             quad_time;
    FullMatrix<Number> const &Alpha;
    FullMatrix<Number> const &Gamma;

    mutable ReductionControl                                     solver_control;
    mutable SolverFGMRES<BlockVectorType>                        solver;
    GMGPreconditioner const                                     &preconditioner;
    SystemMatrix<Number, MatrixFreeOperator<dim, Number>> const &matrix;
    SystemMatrix<Number, MatrixFreeOperator<dim, Number>> const &rhs_matrix;
    std::function<void(const double, VectorType &)> integrate_rhs_function;
    unsigned int                                    n_timesteps_at_once;
  };


  template <int dim, typename Number, typename NumberPreconditioner = Number>
  class TimeIntegratorHeat
    : public TimeIntegrator<dim, Number, NumberPreconditioner>
  {
  public:
    using GMGPreconditioner =
      typename TimeIntegrator<dim, Number, NumberPreconditioner>::
        GMGPreconditioner;
    using VectorType =
      typename TimeIntegrator<dim, Number, NumberPreconditioner>::VectorType;
    using BlockVectorType =
      typename TimeIntegrator<dim, Number, NumberPreconditioner>::
        BlockVectorType;

    TimeIntegratorHeat(
      TimeStepType              type_,
      unsigned int              time_degree_,
      FullMatrix<Number> const &Alpha_,
      FullMatrix<Number> const &Gamma_,
      double const              gmres_tolerance_,
      SystemMatrix<Number, MatrixFreeOperator<dim, Number>> const &matrix_,
      GMGPreconditioner const &preconditioner_,
      SystemMatrix<Number, MatrixFreeOperator<dim, Number>> const &rhs_matrix_,
      std::function<void(const double, VectorType &)> integrate_rhs_function,
      unsigned int                                    n_timesteps_at_once_)
      : TimeIntegrator<dim, Number, NumberPreconditioner>(
          type_,
          time_degree_,
          Alpha_,
          Gamma_,
          gmres_tolerance_,
          matrix_,
          preconditioner_,
          rhs_matrix_,
          integrate_rhs_function,
          n_timesteps_at_once_)
    {}

    void
    solve(BlockVectorType                    &x,
          VectorType const                   &prev_x,
          [[maybe_unused]] const unsigned int timestep_number,
          const double                        time,
          const double                        time_step) const
    {
      BlockVectorType rhs(x.n_blocks());
      for (unsigned int j = 0; j < rhs.n_blocks(); ++j)
        this->matrix.initialize_dof_vector(rhs.block(j));
      this->rhs_matrix.vmult(rhs, prev_x);

      this->assemble_force(rhs, time, time_step);

      // constant extrapolation of solution from last time
      for (unsigned int j = 0; j < rhs.n_blocks(); ++j)
        x.block(j) = prev_x;
      try
        {
          this->solver.solve(this->matrix, x, rhs, this->preconditioner);
        }
      catch (const SolverControl::NoConvergence &e)
        {
          AssertThrow(false, ExcMessage(e.what()));
        }
    }
  };

  template <int dim, typename Number, typename NumberPreconditioner = Number>
  class TimeIntegratorWave
    : public TimeIntegrator<dim, Number, NumberPreconditioner>
  {
  public:
    using GMGPreconditioner =
      typename TimeIntegrator<dim, Number, NumberPreconditioner>::
        GMGPreconditioner;
    using VectorType =
      typename TimeIntegrator<dim, Number, NumberPreconditioner>::VectorType;
    using BlockVectorType =
      typename TimeIntegrator<dim, Number, NumberPreconditioner>::
        BlockVectorType;

    TimeIntegratorWave(
      TimeStepType              type_,
      unsigned int              time_degree_,
      FullMatrix<Number> const &Alpha_,
      FullMatrix<Number> const &Beta_,
      FullMatrix<Number> const &Gamma_,
      FullMatrix<Number> const &Zeta_,
      double const              gmres_tolerance_,
      SystemMatrix<Number, MatrixFreeOperator<dim, Number>> const &matrix_,
      GMGPreconditioner const &preconditioner_,
      SystemMatrix<Number, MatrixFreeOperator<dim, Number>> const &rhs_matrix_,
      SystemMatrix<Number, MatrixFreeOperator<dim, Number>> const
                                                     &rhs_matrix_v_,
      std::function<void(const double, VectorType &)> integrate_rhs_function,
      unsigned int                                    n_timesteps_at_once_)
      : TimeIntegrator<dim, Number, NumberPreconditioner>(
          type_,
          time_degree_,
          Alpha_,
          Gamma_,
          gmres_tolerance_,
          matrix_,
          preconditioner_,
          rhs_matrix_,
          integrate_rhs_function,
          n_timesteps_at_once_)
      , rhs_matrix_v(rhs_matrix_v_)
      , Beta(Beta_)
      , Zeta(Zeta_)
      , Alpha_inv(this->Alpha)
      , AixB(this->Alpha.m(), this->Alpha.n())
      , AixG(this->Alpha.m(), this->Gamma.n())
      , AixZ(this->Alpha.m(), this->Zeta.n())
    {
      Alpha_inv.gauss_jordan();
      Alpha_inv.mmult(this->AixB, this->Beta);
      Alpha_inv.mmult(this->AixG, this->Gamma);
      Alpha_inv.mmult(this->AixZ, this->Zeta);
      if (this->type == TimeStepType::DG)
        AixG *= -1.0;
      else
        AixZ *= -1.0;
    }

    void
    solve(BlockVectorType                    &u,
          BlockVectorType                    &v,
          VectorType const                   &prev_u,
          VectorType const                   &prev_v,
          [[maybe_unused]] const unsigned int timestep_number,
          const double                        time,
          const double                        time_step) const
    {
      BlockVectorType rhs(u.n_blocks());
      for (unsigned int j = 0; j < rhs.n_blocks(); ++j)
        this->matrix.initialize_dof_vector(rhs.block(j));

      this->rhs_matrix.vmult(rhs, prev_u);
      this->rhs_matrix_v.vmult_add(rhs, prev_v);
      this->assemble_force(rhs, time, time_step);

      // constant extrapolation of solution from last time
      for (unsigned int j = 0; j < rhs.n_blocks(); ++j)
        u.block(j) = prev_u;
      try
        {
          this->solver.solve(this->matrix, u, rhs, this->preconditioner);
        }
      catch (const SolverControl::NoConvergence &e)
        {
          AssertThrow(false, ExcMessage(e.what()));
        }
      unsigned int nt_dofs = AixB.m();
      v                    = 0.0;
      for (unsigned int it = 0; it < this->n_timesteps_at_once; ++it)
        {
          VectorType const &prev_u_ =
            it == 0 ? prev_u : u.block(it * nt_dofs - 1);
          tensorproduct_add(v, AixB, u, it * nt_dofs);
          if (this->type == TimeStepType::DG)
            tensorproduct_add(v, AixG, prev_u_, it * nt_dofs);
          else
            {
              VectorType const &prev_v_ =
                it == 0 ? prev_v : v.block(it * nt_dofs - 1);
              tensorproduct_add(v, AixG, prev_v_, it * nt_dofs);
              tensorproduct_add(v, AixZ, prev_u_, it * nt_dofs);
            }
        }
    }

  private:
    SystemMatrix<Number, MatrixFreeOperator<dim, Number>> const &rhs_matrix_v;

    FullMatrix<Number> const &Beta;
    FullMatrix<Number> const &Zeta;

    FullMatrix<Number> Alpha_inv;
    FullMatrix<Number> AixB;
    FullMatrix<Number> AixG;
    FullMatrix<Number> AixZ;
  };
} // namespace dealii
