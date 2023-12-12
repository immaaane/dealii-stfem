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
  get_fe_time_weights(TimeStepType type, unsigned int const r)
  {
    if (type == TimeStepType::CGP)
      {
        return split_lhs_rhs(get_cg_weights<Number>(r));
      }
    else if (type == TimeStepType::DG)
      {
        return split_lhs_rhs(get_dg_weights<Number>(r));
      }
    return {{FullMatrix<Number>(),
             FullMatrix<Number>(),
             FullMatrix<Number>(),
             FullMatrix<Number>()}};
  }

  template <typename Number>
  std::vector<Polynomials::Polynomial<Number>>
  get_time_basis(TimeStepType type, unsigned int const r)
  {
    if (type == TimeStepType::CGP)
      {
        return Polynomials::generate_complete_Lagrange_basis(
          QGaussLobatto<1>(r + 1).get_points());
      }
    else if (type == TimeStepType::DG)
      {
        return Polynomials::generate_complete_Lagrange_basis(
          QGaussRadau<1>(r + 1, QGaussRadau<1>::EndPoint::right).get_points());
      }
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
