// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by Nils Margenberg and Peter Munch

#pragma once

#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>

#include "gmg.h"
#include "operators.h"
#include "types.h"

namespace dealii
{
  struct EmptyNitsche
  {};

  /** Time stepping by DG and CGP variational time discretizations
   *
   * This time integrator is suited for linear problems. For nonlinear problems
   * we would need a few extensions in order to integrate nonlinear terms
   * accurately.
   */
  template <int dim,
            typename Number,
            typename Preconditioner,
            typename System,
            typename NitscheIntegrator = EmptyNitsche>
  class TimeIntegrator
  {
  public:
    using VectorType      = VectorT<Number>;
    using BlockVectorType = BlockVectorT<Number>;

    TimeIntegrator(TimeStepType              type_,
                   unsigned int              time_degree_,
                   FullMatrix<Number> const &Alpha_,
                   FullMatrix<Number> const &Gamma_,
                   double const              gmres_tolerance_,
                   System const             &matrix_,
                   Preconditioner const     &preconditioner_,
                   System const             &rhs_matrix_,
                   std::vector<std::function<void(const double, VectorType &)>>
                                            integrate_rhs_function,
                   unsigned int             n_timesteps_at_once_,
                   bool                     extrapolate_,
                   const NitscheIntegrator &nitsche_ = NitscheIntegrator(),
                   double                   abstol   = 1.e-12)
      : type(type_)
      , time_degree(time_degree_)
      , quad_time(get_time_quad(type, time_degree))
      , Alpha(Alpha_)
      , Gamma(Gamma_)
      , solver_control(200, abstol, gmres_tolerance_, false, true)
      , solver(solver_control,
               typename SolverFGMRES<
                 BlockVectorType>::AdditionalData::AdditionalData(100))
      , preconditioner(preconditioner_)
      , matrix(matrix_)
      , rhs_matrix(rhs_matrix_)
      , nitsche(nitsche_)
      , integrate_rhs_function(integrate_rhs_function)
      , n_timesteps_at_once(n_timesteps_at_once_)
      , idx(n_timesteps_at_once,
            integrate_rhs_function.size(),
            (type == TimeStepType::DG ? time_degree + 1 : time_degree))
      , dirichlet_color_to_fun(idx.n_variables())
      , do_extrapolate(extrapolate_)
    {}

    void
    assemble_force(BlockVectorType &rhs,
                   double const     time,
                   double const     time_step) const
    {
      AssertDimension(rhs.n_blocks(), idx.n_blocks());
      BlockVectorType tmp(integrate_rhs_function.size());
      for (unsigned int i = 0; i < integrate_rhs_function.size(); ++i)
        matrix.initialize_dof_vector(tmp.block(i), i);

      for (unsigned int it = 0; it < idx.n_timesteps_at_once(); ++it)
        for (unsigned int j = 0; j < quad_time.size(); ++j)
          {
            double time_ =
              time + time_step * it + time_step * quad_time.point(j)[0];
            for (unsigned int iv = 0; iv < idx.n_variables(); ++iv)
              {
                integrate_rhs_function[iv](time_, tmp.block(iv));
                /// Here we exploit that Alpha is a diagonal matrix
                if (type == TimeStepType::DG)
                  rhs.block(idx.index(it, iv, j))
                    .add(Alpha(idx.index(0, iv, j), idx.index(0, iv, j)),
                         tmp.block(iv));
                else
                  {
                    if (j == 0)
                      for (unsigned int i = 0; i < idx.n_timedofs(); ++i)
                        rhs.block(idx.index(it, iv, i))
                          .add(-Gamma(idx.index(0, iv, i), 0), tmp.block(iv));
                    else
                      rhs.block(idx.index(it, iv, j - 1))
                        .add(Alpha(idx.index(0, iv, j - 1),
                                   idx.index(0, iv, j - 1)),
                             tmp.block(iv));
                  }
              }
          }
    }

    void
    add_dirichlet_function(unsigned int           var,
                           types::boundary_id     b_id,
                           Function<dim, Number> *f) const
    {
      dirichlet_color_to_fun[var].emplace(b_id, f);
      dirichlet_function_added = true;
    }

    template <typename T = NitscheIntegrator>
    typename std::enable_if<std::is_same<T, EmptyNitsche>::value>::type
    assemble_nitsche(BlockVectorType &, double const, double const) const
    {}

    template <typename T = NitscheIntegrator>
    typename std::enable_if<!std::is_same<T, EmptyNitsche>::value>::type
    assemble_nitsche(BlockVectorType &rhs,
                     double const     time_,
                     double const     time_step) const
    {
      if (!dirichlet_function_added)
        return;
      AssertDimension(rhs.n_blocks(), idx.n_blocks());
      BlockVectorType tmp(idx.n_variables());
      nitsche.initialize_dof_vector(tmp);
      for (unsigned int it = 0; it < idx.n_timesteps_at_once(); ++it)
        for (unsigned int j = 0; j < quad_time.size(); ++j)
          {
            double time =
              time_ + time_step * it + time_step * quad_time.point(j)[0];
            for (unsigned int iv = 0; iv < idx.n_variables(); ++iv)
              {
                if (dirichlet_color_to_fun[iv].empty())
                  continue;
                for (auto &[bid, f] : dirichlet_color_to_fun[iv])
                  f->set_time(time);
                nitsche.set_dirichlet_functions(dirichlet_color_to_fun[iv]);

                nitsche.vmult(tmp);
                if (type == TimeStepType::DG)
                  for (unsigned int jv = 0; jv < idx.n_variables(); ++jv)
                    rhs.block(idx.index(it, jv, j))
                      .add(Alpha(idx.index(0, iv, j), idx.index(0, jv, j)),
                           tmp.block(jv));
                else
                  for (unsigned int jv = 0; jv < idx.n_variables(); ++jv)
                    {
                      if (j == 0)
                        for (unsigned int i = 0; i < idx.n_timedofs(); ++i)
                          rhs.block(idx.index(it, jv, i))
                            .add(-Gamma(idx.index(0, iv, i), 0), tmp.block(jv));
                      else
                        rhs.block(idx.index(it, jv, j - 1))
                          .add(Alpha(idx.index(0, iv, j - 1),
                                     idx.index(0, iv, j - 1)),
                               tmp.block(jv));
                    }
              }
          }
    }

    unsigned int
    last_step() const
    {
      return solver_control.last_step();
    }

  protected:
    void
    extrapolate(BlockVectorType &x, BlockVectorType const &prev_x) const
    {
      for (unsigned int it = 0; it < idx.n_timesteps_at_once(); ++it)
        for (unsigned int i = 0; i < idx.n_variables(); ++i)
          for (unsigned int j = 0; j < idx.n_timedofs(); ++j)
            if (do_extrapolate)
              x.block(idx.index(it, i, j)) = prev_x.block(i);
            else
              x.block(idx.index(it, i, j)) = 0.0;
    }

    TimeStepType              type;
    unsigned int              time_degree;
    Quadrature<1>             quad_time;
    FullMatrix<Number> const &Alpha;
    FullMatrix<Number> const &Gamma;

    mutable ReductionControl              solver_control;
    mutable SolverFGMRES<BlockVectorType> solver;
    Preconditioner const                 &preconditioner;
    System const                         &matrix;
    System const                         &rhs_matrix;
    NitscheIntegrator const              &nitsche;
    std::vector<std::function<void(const double, VectorType &)>>
                 integrate_rhs_function;
    unsigned int n_timesteps_at_once;
    BlockSlice   idx;

    mutable std::vector<
      std::map<types::boundary_id, dealii::Function<dim, Number> *>>
                 dirichlet_color_to_fun;
    mutable bool dirichlet_function_added = false;

    bool do_extrapolate;
  };


  template <int dim,
            typename Number,
            typename Preconditioner,
            typename System,
            typename NitscheIntegrator = EmptyNitsche>
  class TimeIntegratorFO : public TimeIntegrator<dim,
                                                 Number,
                                                 Preconditioner,
                                                 System,
                                                 NitscheIntegrator>
  {
  public:
    using VectorType =
      typename TimeIntegrator<dim, Number, Preconditioner, System>::VectorType;
    using BlockVectorType =
      typename TimeIntegrator<dim, Number, Preconditioner, System>::
        BlockVectorType;

    TimeIntegratorFO(
      TimeStepType              type_,
      unsigned int              time_degree_,
      FullMatrix<Number> const &Alpha_,
      FullMatrix<Number> const &Gamma_,
      double const              gmres_tolerance_,
      System const             &matrix_,
      Preconditioner const     &preconditioner_,
      System const             &rhs_matrix_,
      std::vector<std::function<void(const double, VectorType &)>>
                               integrate_rhs_function,
      unsigned int             n_timesteps_at_once_,
      bool                     extrapolate = true,
      const NitscheIntegrator &nitsche     = NitscheIntegrator(),
      double                   abstol      = 1.e-12)
      : TimeIntegrator<dim, Number, Preconditioner, System, NitscheIntegrator>(
          type_,
          time_degree_,
          Alpha_,
          Gamma_,
          gmres_tolerance_,
          matrix_,
          preconditioner_,
          rhs_matrix_,
          integrate_rhs_function,
          n_timesteps_at_once_,
          extrapolate,
          nitsche,
          abstol)
    {}

    void
    solve(BlockVectorType   &x,
          const VectorType  &prev_x,
          const unsigned int timestep_number,
          const double       time,
          const double       time_step) const
    {
      AssertDimension(this->idx.n_variables(), 1);
      BlockVectorType prev_x_(1);
      prev_x_.block(0).swap(const_cast<VectorType &>(prev_x));
      this->solve(x, prev_x_, timestep_number, time, time_step);
      prev_x_.block(0).swap(const_cast<VectorType &>(prev_x));
    }

    void
    solve(BlockVectorType &x,
          BlockVectorType &prev_x,
          BlockVectorType &rhs,
          const unsigned int,
          const double time,
          const double time_step) const
    {
      this->rhs_matrix.vmult_slice(rhs, prev_x);
      this->assemble_nitsche(rhs, time, time_step);
      this->assemble_force(rhs, time, time_step);

      this->extrapolate(x, prev_x);
      try
        {
          this->solver.solve(this->matrix, x, rhs, this->preconditioner);
        }
      catch (const SolverControl::NoConvergence &e)
        {
          AssertThrow(false, ExcMessage(e.what()));
        }
    }


    void
    solve(BlockVectorType   &x,
          BlockVectorType   &prev_x,
          const unsigned int timestep_number,
          const double       time,
          const double       time_step) const
    {
      BlockVectorType rhs(x.n_blocks());
      for (unsigned int j = 0; j < rhs.n_blocks(); ++j)
        rhs.block(j).reinit(x.block(j).get_partitioner());
      this->solve(x, prev_x, rhs, timestep_number, time, time_step);
    }
  };

  template <int dim, typename Number, typename Preconditioner, typename System>
  class TimeIntegratorWave
    : public TimeIntegrator<dim, Number, Preconditioner, System>
  {
  public:
    using VectorType =
      typename TimeIntegrator<dim, Number, Preconditioner, System>::VectorType;
    using BlockVectorType =
      typename TimeIntegrator<dim, Number, Preconditioner, System>::
        BlockVectorType;

    TimeIntegratorWave(
      TimeStepType              type_,
      unsigned int              time_degree_,
      FullMatrix<Number> const &Alpha_,
      FullMatrix<Number> const &Beta_,
      FullMatrix<Number> const &Gamma_,
      FullMatrix<Number> const &Zeta_,
      double const              gmres_tolerance_,
      System const             &matrix_,
      Preconditioner const     &preconditioner_,
      System const             &rhs_matrix_,
      System const             &rhs_matrix_v_,
      std::vector<std::function<void(const double, VectorType &)>>
                   integrate_rhs_function,
      unsigned int n_timesteps_at_once_,
      bool         extrapolate = true)
      : TimeIntegrator<dim, Number, Preconditioner, System>(
          type_,
          time_degree_,
          Alpha_,
          Gamma_,
          gmres_tolerance_,
          matrix_,
          preconditioner_,
          rhs_matrix_,
          integrate_rhs_function,
          n_timesteps_at_once_,
          extrapolate)
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
    solve(BlockVectorType  &u,
          BlockVectorType  &v,
          VectorType const &prev_u,
          VectorType const &prev_v,
          const unsigned int,
          const double time,
          const double time_step) const
    {
      BlockVectorType rhs(u.n_blocks());
      for (unsigned int j = 0; j < rhs.n_blocks(); ++j)
        this->matrix.initialize_dof_vector(rhs.block(j));

      BlockVectorType prev_(1);
      prev_.block(0).swap(const_cast<VectorType &>(prev_u));
      this->rhs_matrix.vmult_slice(rhs, prev_);
      this->extrapolate(u, prev_);
      prev_.block(0).swap(const_cast<VectorType &>(prev_u));

      prev_.block(0).swap(const_cast<VectorType &>(prev_v));
      this->rhs_matrix_v.vmult_slice_add(rhs, prev_);
      prev_.block(0).swap(const_cast<VectorType &>(prev_v));

      this->assemble_force(rhs, time, time_step);

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
    System const &rhs_matrix_v;

    FullMatrix<Number> const &Beta;
    FullMatrix<Number> const &Zeta;

    FullMatrix<Number> Alpha_inv;
    FullMatrix<Number> AixB;
    FullMatrix<Number> AixG;
    FullMatrix<Number> AixZ;
  };
} // namespace dealii
