#pragma once

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_q.h>

#include <array>


enum class TimeStepType : unsigned int
{
  CGP = 1,
  DG  = 2,
  GCC = 3
};

namespace dealii
{
  template <typename Number>
  std::array<FullMatrix<Number>, 3>
  get_dg_weights(unsigned int const r);

  template <typename Number>
  std::array<FullMatrix<Number>, 2>
  get_cg_weights(unsigned int const r);

  template <typename Number>
  std::array<FullMatrix<Number>, 4>
  get_fe_time_weights(TimeStepType type, unsigned int const r);

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


  /** Generates the time integration weights for time continuous Galerkin-Petrov
   * discretizations or time discontinuous Galerkin discretizations
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

  template <typename Number>
  std::vector<Polynomials::Polynomial<Number>>
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

    auto const poly_lobatto = get_time_basis<Number>(TimeStepType::CGP, r);
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
    auto const poly_radau = get_time_basis<Number>(TimeStepType::DG, r);

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
} // namespace dealii
