// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by Nils Margenberg and Peter Munch

#pragma once

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>

#include <array>


enum class TimeStepType : unsigned int
{
  CGP = 1,
  DG  = 2,
  GCC = 3
};
static std::unordered_map<std::string, TimeStepType> const str_to_time_type = {
  {"CGP", TimeStepType::CGP},
  {"DG", TimeStepType::DG},
  {"GCC", TimeStepType::GCC}};

enum class TimeMGType : unsigned int
{
  tau  = 1,
  k    = 2,
  none = 3,
};

namespace dealii
{
  template <typename Number>
  std::array<FullMatrix<Number>, 3>
  get_dg_weights(unsigned int const r);

  template <typename Number>
  std::array<FullMatrix<Number>, 2>
  get_cg_weights(unsigned int const r);

  inline std::vector<TimeMGType>
  get_time_mg_sequence(unsigned int const n_sp_lvl,
                       unsigned int const k_max,
                       unsigned int const k_min,
                       unsigned int const n_timesteps_at_once,
                       unsigned int const n_timesteps_at_once_min = 1,
                       TimeMGType const   lower_lvl         = TimeMGType::k,
                       bool               time_before_space = false)
  {
    Assert(n_sp_lvl >= 1, ExcLowerRange(n_sp_lvl, 1));
    unsigned int n_k_lvl = k_max - k_min;
    unsigned int n_t_lvl =
      std::log2(n_timesteps_at_once / n_timesteps_at_once_min);
    auto upper_lvl =
      (lower_lvl == TimeMGType::k) ? TimeMGType::tau : TimeMGType::k;
    unsigned int n_ll = (lower_lvl == TimeMGType::k) ? n_k_lvl : n_t_lvl;
    unsigned int n_ul = (lower_lvl == TimeMGType::k) ? n_t_lvl : n_k_lvl;
    std::vector<TimeMGType> mg_type_level(n_k_lvl + n_t_lvl + n_sp_lvl - 1,
                                          TimeMGType::none);

    unsigned int start_lower_lvl = time_before_space ? n_sp_lvl - 1 : 0;
    unsigned int start_upper_lvl =
      time_before_space ? n_sp_lvl - 1 + n_ll : n_ll;
    std::fill_n(mg_type_level.begin() + start_lower_lvl, n_ll, lower_lvl);
    std::fill_n(mg_type_level.begin() + start_upper_lvl, n_ul, upper_lvl);
    return mg_type_level;
  }

  template <int dim>
  std::vector<std::shared_ptr<const Triangulation<dim>>>
  get_space_time_triangulation(
    std::vector<TimeMGType> const &mg_type_level,
    std::vector<std::shared_ptr<const Triangulation<dim>>> const
      &mg_triangulations)
  {
    AssertDimension(mg_triangulations.size() - 1,
                    std::count(mg_type_level.begin(),
                               mg_type_level.end(),
                               TimeMGType::none));
    std::vector<std::shared_ptr<const Triangulation<dim>>>
         new_mg_triangulations(1 + mg_type_level.size());
    auto mg_tria_it              = mg_triangulations.rbegin();
    new_mg_triangulations.back() = *mg_tria_it;

    unsigned int ii = mg_type_level.size() - 1;
    for (auto mgt = mg_type_level.rbegin(); mgt != mg_type_level.rend();
         ++mgt, --ii)
      {
        if (*mgt == TimeMGType::none)
          ++mg_tria_it;
        new_mg_triangulations.at(ii) = *mg_tria_it;
      }
    return new_mg_triangulations;
  }

  template <typename Number>
  std::array<FullMatrix<Number>, 5>
  get_fe_time_weights_wave(TimeStepType              type,
                           FullMatrix<Number> const &Alpha,
                           FullMatrix<Number> const &Beta,
                           FullMatrix<Number> const &Gamma,
                           FullMatrix<Number> const &Zeta,
                           unsigned int              n_timesteps_at_once = 1)
  {
    FullMatrix<Number> Alpha_inv(Alpha.m(), Alpha.n());
    Alpha_inv.invert(Alpha);

    FullMatrix<Number> BxAixB(Alpha.m(), Alpha.n());
    // Note that the order of the arguments is not the order of the mmults:
    // Alpha_inv, Beta, Beta corresponds to Beta*Alpha_inv*Beta
    // Generally A,B,D -> B*A*D.
    BxAixB.triple_product(Alpha_inv, Beta, Beta);

    // u0 mass
    FullMatrix<Number> BxAixG(Gamma.m(), Gamma.n());
    BxAixG.triple_product(Alpha_inv, Beta, Gamma);
    // u00
    double gxai = Gamma(Gamma.m() - 1, 0) / Alpha(Gamma.m() - 1, Gamma.m() - 1);
    FullMatrix<Number> GxAixG = Gamma;
    GxAixG *= gxai;

    FullMatrix<Number> Beta_row(1, Beta.n());
    for (auto b = Beta.begin(Beta.m() - 1), br = Beta_row.begin(0);
         b != Beta.end(Beta.m() - 1);
         ++br, ++b)
      *br = *b;

    FullMatrix<Number> GxAixB(Gamma.m(), Beta.n());
    Gamma.mmult(GxAixB, Beta_row);
    GxAixB /= Alpha(Gamma.m() - 1, Gamma.m() - 1);



    unsigned int nt_dofs_intvl = Alpha.m();
    unsigned int nt_dofs_tot   = nt_dofs_intvl * n_timesteps_at_once;
    std::array<FullMatrix<Number>, 5> ret{
      {FullMatrix<Number>(nt_dofs_tot, nt_dofs_tot),
       FullMatrix<Number>(nt_dofs_tot, nt_dofs_tot),
       FullMatrix<Number>(nt_dofs_tot, 1),
       FullMatrix<Number>(nt_dofs_tot, 1),
       FullMatrix<Number>(nt_dofs_tot, 1)}};
    if (type == TimeStepType::CGP)
      {
        FullMatrix<Number> BxAixZ(Zeta.m(), Zeta.n());
        BxAixZ.triple_product(Alpha_inv, Beta, Zeta);

        FullMatrix<Number> ZmBxAixG = Zeta;
        ZmBxAixG.add(-1.0, BxAixG);
        FullMatrix<Number> ZmBxAixB(Gamma.m(), Beta.n());
        ZmBxAixG.mmult(ZmBxAixB, Beta_row);
        ZmBxAixB /= Alpha(Gamma.m() - 1, Gamma.m() - 1);
        double zxai = Zeta(Zeta.m() - 1, 0) / Alpha(Zeta.m() - 1, Zeta.m() - 1);

        for (unsigned int it = 0; it < n_timesteps_at_once; ++it)
          for (unsigned int jt = 0; jt <= it; ++jt)
            for (unsigned int i = 0; i < nt_dofs_intvl; ++i)
              {
                // rhs
                if (it == 0 && jt == 0)
                  {
                    ret[2](i, 0) = Gamma(i, 0);
                    ret[3](i, 0) = BxAixZ(i, 0);
                    ret[4](i, 0) = ZmBxAixG(i, 0);
                  }
                else if (jt == 0)
                  {
                    ret[3](i + it * nt_dofs_intvl, 0) =
                      -zxai * pow(gxai, it - 1) * ZmBxAixG(i, 0);
                    ret[4](i + it * nt_dofs_intvl, 0) =
                      pow(gxai, it) * ZmBxAixG(i, 0);
                  }

                if (it == jt + 1) // lower diagonal
                  {
                    ret[0](i + it * nt_dofs_intvl,
                           nt_dofs_intvl - 1 + jt * nt_dofs_intvl) =
                      -Gamma(i, 0);
                    ret[1](i + it * nt_dofs_intvl,
                           nt_dofs_intvl - 1 + jt * nt_dofs_intvl) =
                      -BxAixZ(i, 0);
                  }

                if (it == jt) // Main diagonal
                  for (unsigned int j = 0; j < nt_dofs_intvl; ++j)
                    {
                      ret[0](i + it * nt_dofs_intvl, j + it * nt_dofs_intvl) =
                        Alpha(i, j);
                      ret[1](i + it * nt_dofs_intvl, j + it * nt_dofs_intvl) =
                        BxAixB(i, j);
                    }
                else // lower triangle
                  for (unsigned int j = 0; j < nt_dofs_intvl; ++j)
                    ret[1](i + it * nt_dofs_intvl, j + jt * nt_dofs_intvl) +=
                      -pow(gxai, it - jt - 1) * ZmBxAixB(i, j) +
                      (it > 1 && it - 1 > jt && j == nt_dofs_intvl - 1 ?
                         pow(gxai, it - jt - 2) * zxai * ZmBxAixG(i, 0) :
                         0.0);
              }
      }
    else if (type == TimeStepType::DG)
      {
        for (unsigned int it = 0; it < n_timesteps_at_once; ++it)
          for (unsigned int i = 0; i < nt_dofs_intvl; ++i)
            {
              // 1st block rhs
              if (it == 0)
                {
                  ret[3](i, 0) = BxAixG(i, 0);
                  ret[4](i, 0) = Gamma(i, 0);
                }
              // 2nd block rhs
              if (it == 1)
                ret[3](nt_dofs_intvl + i, 0) = -GxAixG(i, 0);

              // 1st lower block diagonal
              if (it < n_timesteps_at_once - 1)
                for (unsigned int j = 0; j < nt_dofs_intvl; ++j)
                  {
                    ret[1](j + (it + 1) * nt_dofs_intvl,
                           i + it * nt_dofs_intvl) =
                      -GxAixB(j, i) -
                      (i == nt_dofs_intvl - 1 ? BxAixG(j, 0) : 0.0);
                  }

              // 2nd lower diagonal
              if (static_cast<int>(it) <
                    static_cast<int>(n_timesteps_at_once) - 2 &&
                  i == nt_dofs_intvl - 1)
                for (unsigned int j = 0; j < nt_dofs_intvl; ++j)
                  ret[1](j + (it + 2) * nt_dofs_intvl, i + it * nt_dofs_intvl) =
                    GxAixG(j, 0);

              // Main diagonal
              for (unsigned int j = 0; j < nt_dofs_intvl; ++j)
                {
                  ret[0](i + it * nt_dofs_intvl, j + it * nt_dofs_intvl) =
                    Alpha(i, j);
                  ret[1](i + it * nt_dofs_intvl, j + it * nt_dofs_intvl) =
                    BxAixB(i, j);
                }
            }
      }
    return ret;
  }

  template <typename Number>
  FullMatrix<Number>
  get_time_evaluation_matrix(
    std::vector<Polynomials::Polynomial<double>> const &basis,
    unsigned int                                        samples_per_interval)
  {
    double             sample_step = 1.0 / (samples_per_interval - 1);
    FullMatrix<Number> time_evaluator(samples_per_interval, basis.size());
    for (unsigned int s = 0; s < samples_per_interval; ++s)
      {
        double time_ = s * sample_step;
        auto   te    = time_evaluator.begin(s);
        for (auto const &el : basis)
          {
            *te = el.value(time_);
            ++te;
          }
      }
    return time_evaluator;
  }

  /** Generates the time integration weights for time continuous
   * Galerkin-Petrov discretizations or time discontinuous Galerkin
   * discretizations
   *
   * For a given order r the method returns 4 matrices. In the continouus
   * Galerkin-Petrov (CG) case these are:
   * 1. time integration weights, commonly denoted by Alpha (r x r)
   * 2. time integration weights for temporal derivatives ,
   *    commonly denoted by Beta (r x r)
   * 3. time integration weights for terms which can be put on the RHS,
   *    commonly denoded by Gamma (r x 1)
   * 4. time integration weights for temporal derivatives which can be put on
   *    the RHS, commonly denoded by Zeta (r x 1)
   *
   * In the discontinouus Galerkin (DG) case these are:
   * 1. time integration weights, commonly denoted by Alpha (r+1 x r+1)
   * 2. time integration weights for temporal derivatives ,
   *    commonly denoted by Beta (r+1 x r+1)
   * 3. time integration weights for terms which can be put on the RHS,
   *    commonly denoded by Gamma (r+1 x 1)
   * 4. the fourth matrix is zero in the DG case, as we only get jump terms in
   *    the DG case (r+1 x 1)
   */
  template <typename Number>
  std::array<FullMatrix<Number>, 4>
  get_fe_time_weights(TimeStepType       type,
                      unsigned int const r,
                      double             time_step_size,
                      unsigned int       n_timesteps_at_once = 1)
  {
    std::array<FullMatrix<Number>, 4> tmp;
    if (type == TimeStepType::CGP)
      {
        tmp = split_lhs_rhs(get_cg_weights<Number>(r));
        tmp[2] *= time_step_size;
      }
    else if (type == TimeStepType::DG)
      {
        tmp    = split_lhs_rhs(get_dg_weights<Number>(r));
        tmp[3] = tmp[2];
        tmp[2] = 0.0;
      }
    tmp[0] *= time_step_size;

    unsigned int nt_dofs_intvl = tmp[0].m();
    unsigned int nt_dofs_tot   = nt_dofs_intvl * n_timesteps_at_once;
    std::array<FullMatrix<Number>, 4> ret{
      {FullMatrix<Number>(nt_dofs_tot, nt_dofs_tot),
       FullMatrix<Number>(nt_dofs_tot, nt_dofs_tot),
       FullMatrix<Number>(nt_dofs_tot, 1),
       FullMatrix<Number>(nt_dofs_tot, 1)}};

    for (unsigned int it = 0; it < n_timesteps_at_once; ++it)
      for (unsigned int i = 0; i < nt_dofs_intvl; ++i)
        {
          // First lower block diagonal
          if (it < n_timesteps_at_once - 1 && i == nt_dofs_intvl - 1)
            for (unsigned int j = 0; j < nt_dofs_intvl; ++j)
              {
                ret[0](j + (it + 1) * nt_dofs_intvl, i + it * nt_dofs_intvl) =
                  -tmp[2](j, 0);
                ret[1](j + (it + 1) * nt_dofs_intvl, i + it * nt_dofs_intvl) =
                  -tmp[3](j, 0);
              }

          // Main block diagonal
          for (unsigned int j = 0; j < nt_dofs_intvl; ++j)
            {
              ret[0](i + it * nt_dofs_intvl, j + it * nt_dofs_intvl) =
                tmp[0](i, j);
              ret[1](i + it * nt_dofs_intvl, j + it * nt_dofs_intvl) =
                tmp[1](i, j);
            }
        }
    for (unsigned int i = 0; i < nt_dofs_intvl; ++i)
      {
        ret[2](i, 0) = tmp[type == TimeStepType::CGP ? 2 : 3](i, 0);
        ret[3](i, 0) = tmp[type == TimeStepType::CGP ? 3 : 2](i, 0);
      }
    return ret;
  }

  template <typename Number, typename NumberPreconditioner = Number>
  std::vector<std::array<FullMatrix<NumberPreconditioner>, 4>>
  get_fe_time_weights(TimeStepType                   type,
                      unsigned int                   r,
                      double                         time_step_size,
                      unsigned int                   n_timesteps_at_once,
                      std::vector<TimeMGType> const &mg_type_level)
  {
    std::vector<std::array<FullMatrix<NumberPreconditioner>, 4>> time_weights(
      mg_type_level.size() + 1);
    auto tw = time_weights.rbegin();
    *tw     = get_fe_time_weights<NumberPreconditioner>(type,
                                                    r,
                                                    time_step_size,
                                                    n_timesteps_at_once);
    ++tw;
    for (auto mgt = mg_type_level.rbegin(); mgt != mg_type_level.rend();
         ++mgt, ++tw)
      {
        if (*mgt == TimeMGType::k)
          --r;
        else if (*mgt == TimeMGType::tau)
          n_timesteps_at_once /= 2, time_step_size *= 2;
        *tw = get_fe_time_weights<NumberPreconditioner>(type,
                                                        r,
                                                        time_step_size,
                                                        n_timesteps_at_once);
      }
    Assert(tw == time_weights.rend(), ExcInternalError());
    return time_weights;
  }

  template <typename Number, typename NumberPreconditioner = Number>
  std::vector<std::array<FullMatrix<NumberPreconditioner>, 5>>
  get_fe_time_weights_wave(TimeStepType                   type,
                           unsigned int                   r,
                           double                         time_step_size,
                           unsigned int                   n_timesteps_at_once,
                           std::vector<TimeMGType> const &mg_type_level)
  {
    auto time_weights = get_fe_time_weights<Number, NumberPreconditioner>(
      type, r, time_step_size, n_timesteps_at_once, mg_type_level);
    std::vector<std::array<FullMatrix<NumberPreconditioner>, 5>>
         time_weights_wave(mg_type_level.size() + 1);
    auto tw      = time_weights.rbegin();
    auto tw_wave = time_weights_wave.rbegin();
    *tw_wave =
      get_fe_time_weights_wave(type, (*tw)[0], (*tw)[1], (*tw)[2], (*tw)[3]);
    ++tw, ++tw_wave;
    for (auto mgt = mg_type_level.rbegin(); mgt != mg_type_level.rend();
         ++mgt, ++tw, ++tw_wave)
      {
        if (*mgt == TimeMGType::k)
          --r;
        else if (*mgt == TimeMGType::tau)
          n_timesteps_at_once /= 2, time_step_size *= 2;
        *tw_wave = get_fe_time_weights_wave(
          type, (*tw)[0], (*tw)[1], (*tw)[2], (*tw)[3]);
      }
    Assert(tw == time_weights.rend(), ExcInternalError());
    return time_weights_wave;
  }

  std::vector<Polynomials::Polynomial<double>>
  get_time_basis(TimeStepType type, unsigned int const r)
  {
    if (type == TimeStepType::CGP)
      return Polynomials::generate_complete_Lagrange_basis(
        QGaussLobatto<1>(r + 1).get_points());
    else if (type == TimeStepType::DG)
      return Polynomials::generate_complete_Lagrange_basis(
        QGaussRadau<1>(r + 1, QGaussRadau<1>::EndPoint::right).get_points());

    return {{}};
  }


  /** Utility function for splitting the weights into parts belonging to the RHS
   * and LHS.
   */
  template <typename Number>
  std::array<FullMatrix<Number>, 4>
  split_lhs_rhs(std::array<FullMatrix<Number>, 2> const &time_weights)
  {
    // Initialize return array of matrices
    std::array<FullMatrix<Number>, 4> ret{
      {FullMatrix<Number>(time_weights[0].m(), time_weights[0].n() - 1),
       FullMatrix<Number>(time_weights[0].m(), time_weights[0].n() - 1),
       FullMatrix<Number>(time_weights[0].m(), 1),
       FullMatrix<Number>(time_weights[0].m(), 1)}};

    // Generate indices for the last columns (LHS) and the first column (RHS)
    std::vector<unsigned> lhs_indices(time_weights[0].n() - 1);
    std::iota(lhs_indices.begin(), lhs_indices.end(), 1);
    std::vector<unsigned> const rhs_indices{0};

    // Generate row indices
    std::vector<unsigned> row_indices(time_weights[0].m());
    std::iota(row_indices.begin(), row_indices.end(), 0);

    // Scatter matrices
    ret[0].extract_submatrix_from(time_weights[0], row_indices, lhs_indices);
    ret[1].extract_submatrix_from(time_weights[1], row_indices, lhs_indices);
    ret[2].extract_submatrix_from(time_weights[0], row_indices, rhs_indices);
    ret[3].extract_submatrix_from(time_weights[1], row_indices, rhs_indices);

    ret[2] *= -1.0;
    ret[3] *= -1.0;
    return ret;
  }

  template <typename Number>
  std::array<FullMatrix<Number>, 4>
  split_lhs_rhs(std::array<FullMatrix<Number>, 3> time_weights)
  {
    // DG
    return {{time_weights[0],
             time_weights[1],
             time_weights[2],
             FullMatrix<Number>(time_weights[2].m(), 1)}};
  }

  template <typename Number>
  std::array<FullMatrix<Number>, 2>
  get_cg_weights(unsigned int const r)
  {
    auto const trial_points = QGaussLobatto<1>(r + 1).get_points();
    std::vector<Point<1>> const test_points(trial_points.begin() + 1,
                                            trial_points.end());

    auto const poly_lobatto = get_time_basis(TimeStepType::CGP, r);
    auto const poly_test =
      Polynomials::generate_complete_Lagrange_basis(test_points);

    std::vector<Polynomials::Polynomial<double>> poly_lobatto_derivative(
      poly_lobatto.size());

    for (unsigned int i = 0; i < poly_lobatto.size(); ++i)
      poly_lobatto_derivative[i] = poly_lobatto[i].derivative();

    QGauss<1> const    quad(r + 2);
    FullMatrix<Number> matrix(r, r + 1);
    FullMatrix<Number> matrix_der(r, r + 1);

    // Later multiply with tau_n
    for (unsigned int i = 0; i < r; ++i)
      for (unsigned int j = 0; j < r + 1; ++j)
        for (unsigned int q = 0; q < quad.size(); ++q)
          matrix(i, j) += quad.weight(q) *
                          poly_test[i].value(quad.point(q)[0]) *
                          poly_lobatto[j].value(quad.point(q)[0]);


    for (unsigned int i = 0; i < r; ++i)
      for (unsigned int j = 0; j < r + 1; ++j)
        for (unsigned int q = 0; q < quad.size(); ++q)
          matrix_der(i, j) +=
            quad.weight(q) * poly_test[i].value(quad.point(q)[0]) *
            poly_lobatto_derivative[j].value(quad.point(q)[0]);

    return {{matrix, matrix_der}};
  }

  template <typename Number>
  std::array<FullMatrix<Number>, 3>
  get_dg_weights(unsigned int const r)
  {
    // Radau quadrature
    auto const poly_radau = get_time_basis(TimeStepType::DG, r);

    std::vector<Polynomials::Polynomial<double>> poly_radau_derivative(
      poly_radau.size());
    for (unsigned int i = 0; i < poly_radau.size(); ++i)
      poly_radau_derivative[i] = poly_radau[i].derivative();

    QGauss<1> const quad(r + 2);

    FullMatrix<Number> lhs_matrix(r + 1, r + 1); // Later multiply with tau_n
    FullMatrix<Number> jump_matrix(r + 1, 1);
    FullMatrix<Number> lhs_matrix_der(r + 1, r + 1);

    for (unsigned int i = 0; i < r + 1; ++i)
      {
        jump_matrix(i, 0) = poly_radau[i].value(0.0);
        for (unsigned int j = 0; j < r + 1; ++j)
          for (unsigned int q = 0; q < quad.size(); ++q)
            lhs_matrix(i, j) += quad.weight(q) *
                                poly_radau[i].value(quad.point(q)[0]) *
                                poly_radau[j].value(quad.point(q)[0]);
      }

    for (unsigned int i = 0; i < r + 1; ++i)
      for (unsigned int j = 0; j < r + 1; ++j)
        {
          // Jump
          lhs_matrix_der(i, j) +=
            poly_radau[i].value(0) * poly_radau[j].value(0);
          // Integration
          for (unsigned int q = 0; q < quad.size(); ++q)
            lhs_matrix_der(i, j) +=
              quad.weight(q) * poly_radau[i].value(quad.point(q)[0]) *
              poly_radau_derivative[j].value(quad.point(q)[0]);
        }
    return {{lhs_matrix, lhs_matrix_der, jump_matrix}};
  }

  std::vector<size_t>
  get_fe_q_permutation(FE_Q<1> const &fe_time)
  {
    size_t const        n_dofs = fe_time.n_dofs_per_cell();
    std::vector<size_t> permutation(n_dofs, 0);
    std::iota(permutation.begin() + 1, permutation.end() - 1, 2);
    permutation.back() = 1;
    return permutation;
  }

  template <typename Number = double>
  FullMatrix<Number>
  get_time_projection_matrix(TimeStepType       type,
                             unsigned int const r_src,
                             unsigned int const r_dst,
                             unsigned int const n_timesteps_at_once)
  {
    auto n_dofs_intvl_dst = (type == TimeStepType::DG) ? r_dst + 1 : r_dst;
    auto n_dofs_intvl_src = (type == TimeStepType::DG) ? r_src + 1 : r_src;

    auto n_dofs_dst = (type == TimeStepType::DG) ?
                        n_timesteps_at_once * (r_dst + 1) :
                        (n_timesteps_at_once * r_dst + 1);
    auto n_dofs_src = (type == TimeStepType::DG) ?
                        n_timesteps_at_once * (r_src + 1) :
                        (n_timesteps_at_once * r_src + 1);

    FullMatrix<Number> projection(r_dst + 1, r_src + 1),
      projection_n(n_dofs_dst, n_dofs_src);

    if (type == TimeStepType::DG)
      {
        auto quad_time_src =
          QGaussRadau<1>(r_src + 1, QGaussRadau<1>::EndPoint::right);
        FE_DGQArbitraryNodes<1> fe_time_src(quad_time_src.get_points());
        auto                    quad_time_dst =
          QGaussRadau<1>(r_dst + 1, QGaussRadau<1>::EndPoint::right);
        FE_DGQArbitraryNodes<1> fe_time_dst(quad_time_dst.get_points());
        FETools::get_projection_matrix(fe_time_src, fe_time_dst, projection);
      }
    else
      {
        auto    quad_time_src = QGaussLobatto<1>(r_src + 1);
        FE_Q<1> fe_time_src(quad_time_src.get_points());
        auto    quad_time_dst = QGaussLobatto<1>(r_dst + 1);
        FE_Q<1> fe_time_dst(quad_time_dst.get_points());

        FullMatrix<Number> projection_(fe_time_dst.n_dofs_per_cell(),
                                       fe_time_src.n_dofs_per_cell());
        FETools::get_projection_matrix(fe_time_src, fe_time_dst, projection_);
        auto perm_dst = get_fe_q_permutation(fe_time_dst);
        auto perm_src = get_fe_q_permutation(fe_time_src);
        projection.fill_permutation(projection_, perm_dst, perm_src);
      }

    for (unsigned int it = 0; it < n_timesteps_at_once; ++it)
      projection_n.fill(
        projection, it * n_dofs_intvl_dst, it * n_dofs_intvl_src, 0, 0);

    if (type == TimeStepType::CGP)
      {
        FullMatrix<Number> projection_n_(n_dofs_dst - 1, n_dofs_src - 1);
        projection_n_.fill(projection_n, 0, 0, 1, 1);
        return projection_n_;
      }
    return projection_n;
  }

  template <typename Number = double>
  FullMatrix<Number>
  get_time_prolongation_matrix(TimeStepType       time_type,
                               unsigned int const r,
                               unsigned int const n_timesteps_at_once = 2)
  {
    Assert((n_timesteps_at_once > 1 &&
            ((n_timesteps_at_once & (n_timesteps_at_once - 1)) == 0)),
           ExcMessage("Has to be a power of 2 and more than one timestep"));
    FullMatrix<Number> prolongation, prolongation_n;
    Quadrature<1>      quad_time;
    if (time_type == TimeStepType::DG)
      {
        quad_time = QGaussRadau<1>(r + 1, QGaussRadau<1>::EndPoint::right);
        FE_DGQArbitraryNodes<1> fe_time(quad_time.get_points());
        auto left_interval  = fe_time.get_prolongation_matrix(0),
             right_interval = fe_time.get_prolongation_matrix(1);
        prolongation.reinit(2 * (r + 1), r + 1);
        prolongation.fill(left_interval, 0, 0, 0, 0);
        prolongation.fill(right_interval, left_interval.m(), 0, 0, 0);
      }
    else if (time_type == TimeStepType::CGP)
      {
        quad_time = QGaussLobatto<1>(r + 1);
        FE_Q<1> fe_time(quad_time.get_points());
        auto    left_interval_ = fe_time.get_prolongation_matrix(0),
             right_interval_   = fe_time.get_prolongation_matrix(1);
        FullMatrix<Number> right_interval(r + 1, r + 1),
          left_interval(r + 1, r + 1);
        auto perm = get_fe_q_permutation(fe_time);
        right_interval.fill_permutation(right_interval_, perm, perm);
        left_interval.fill_permutation(left_interval_, perm, perm);
        prolongation.reinit(2 * r, r);
        prolongation.fill(left_interval, 0, 0, 1, 1);
        prolongation.fill(right_interval, r, 0, 1, 1);
      }
    auto n_dofs_intvl = (time_type == TimeStepType::DG) ? r + 1 : r;
    prolongation_n.reinit(n_dofs_intvl * n_timesteps_at_once,
                          n_dofs_intvl * n_timesteps_at_once / 2);
    for (unsigned int it = 0; it < n_timesteps_at_once / 2; ++it)
      prolongation_n.fill(
        prolongation, it * 2 * n_dofs_intvl, it * n_dofs_intvl, 0, 0);

    return prolongation_n;
  }

  template <typename Number = double>
  FullMatrix<Number>
  get_time_restriction_matrix(TimeStepType       time_type,
                              unsigned int const r,
                              unsigned int const n_timesteps_at_once = 2)
  {
    Assert((n_timesteps_at_once > 1 &&
            ((n_timesteps_at_once & (n_timesteps_at_once - 1)) == 0)),
           ExcMessage("Has to be a power of 2 and more than one timestep"));
    FullMatrix<Number> restriction, restriction_n;
    Quadrature<1>      quad_time;
    if (time_type == TimeStepType::DG)
      {
        quad_time = QGaussRadau<1>(r + 1, QGaussRadau<1>::EndPoint::right);
        FE_DGQArbitraryNodes<1> fe_time(quad_time.get_points());
        auto left_interval  = fe_time.get_restriction_matrix(0),
             right_interval = fe_time.get_restriction_matrix(1);
        restriction.reinit(r + 1, 2 * (r + 1));
        restriction.fill(left_interval, 0, 0, 0, 0);
        restriction.fill(right_interval, 0, left_interval.n(), 0, 0);
      }
    else if (time_type == TimeStepType::CGP)
      {
        quad_time = QGaussLobatto<1>(r + 1);
        FE_Q<1> fe_time(quad_time.get_points());

        auto left_interval_  = fe_time.get_restriction_matrix(0),
             right_interval_ = fe_time.get_restriction_matrix(1);
        FullMatrix<Number> right_interval(r + 1, r + 1),
          left_interval(r + 1, r + 1);
        auto perm = get_fe_q_permutation(fe_time);
        right_interval.fill_permutation(right_interval_, perm, perm);
        left_interval.fill_permutation(left_interval_, perm, perm);
        restriction.reinit(r, 2 * r);
        restriction.fill(left_interval, 0, 0, 1, 1);
        restriction.fill(right_interval, 0, r, 1, 1);
      }
    auto n_dofs_intvl = (time_type == TimeStepType::DG) ? r + 1 : r;
    restriction_n.reinit(n_dofs_intvl * n_timesteps_at_once / 2,
                         n_dofs_intvl * n_timesteps_at_once);
    for (unsigned int it = 0; it < n_timesteps_at_once / 2; ++it)
      restriction_n.fill(
        restriction, it * n_dofs_intvl, it * 2 * n_dofs_intvl, 0, 0);

    return restriction_n;
  }

} // namespace dealii
