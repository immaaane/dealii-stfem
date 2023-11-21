#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_q.h>

using namespace dealii;

template <typename T>
void
print_formatted(const FullMatrix<T> &matrix)
{
#if false
  matrix.print_formatted(std::cout, 1, false, 5, "     ");
#else
  for (unsigned int i = 0; i < matrix.m(); ++i)
    {
      for (unsigned int j = 0; j < matrix.n(); ++j)
        if (std::abs(matrix[i][j]) < 0.01)
          printf("       ");
        else
          printf("%7.2f", matrix[i][j]);
      std::cout << std::endl;
    }
  std::cout << std::endl;
#endif
}

void
test(const std::string type, const unsigned int r)
{
  if (type == "CG")
    { // CG
      using Number = double;

      std::cout << "CG(" << r << ")" << std::endl;

      auto const            trial_points = QGaussLobatto<1>(r + 1).get_points();
      std::vector<Point<1>> test_points(trial_points.begin() + 1,
                                        trial_points.end());

      // Lagrange polynomial
      const auto poly_lobatto =
        Polynomials::generate_complete_Lagrange_basis(trial_points);

      // Radau quadrature
      const auto poly_test =
        Polynomials::generate_complete_Lagrange_basis(test_points);

      std::vector<Polynomials::Polynomial<double>> poly_lobatto_derivative(
        poly_lobatto.size());

      for (unsigned int i = 0; i < poly_lobatto.size(); ++i)
        poly_lobatto_derivative[i] = poly_lobatto[i].derivative();

      QGauss<1> quad(r + 2);

      FullMatrix<Number> full_matrix(r, r + 1);

      // Later multiply with tau_n
      for (unsigned int i = 0; i < r; ++i)
        for (unsigned int j = 0; j < r + 1; ++j)
          for (unsigned int q = 0; q < quad.size(); ++q)
            full_matrix(i, j) += quad.weight(q) *
                                 poly_test[i].value(quad.point(q)[0]) *
                                 poly_lobatto[j].value(quad.point(q)[0]);

      FullMatrix<Number> full_matrix_der(r, r + 1);

      for (unsigned int i = 0; i < r; ++i)
        for (unsigned int j = 0; j < r + 1; ++j)
          for (unsigned int q = 0; q < quad.size(); ++q)
            full_matrix_der(i, j) +=
              quad.weight(q) * poly_test[i].value(quad.point(q)[0]) *
              poly_lobatto_derivative[j].value(quad.point(q)[0]);

      print_formatted(full_matrix);
      print_formatted(full_matrix_der);
    }

  if (type == "DG")
    { // DG
      using Number = double;

      std::cout << "DG(" << r << ")" << std::endl;

      // Radau quadrature
      const auto poly_radau = Polynomials::generate_complete_Lagrange_basis(
        QGaussRadau<1>(r + 1, QGaussRadau<1>::EndPoint::right).get_points());

      std::vector<Polynomials::Polynomial<double>> poly_radau_derivative(
        poly_radau.size());

      for (unsigned int i = 0; i < poly_radau.size(); ++i)
        poly_radau_derivative[i] = poly_radau[i].derivative();

      QGauss<1> quad(r + 2);

      FullMatrix<Number> full_matrix(r + 1, r + 1); // Later multiply with tau_n
      FullMatrix<Number> jump_matrix(r + 1, 1);

      for (unsigned int i = 0; i < r + 1; ++i)
        {
          jump_matrix(i, 0) = poly_radau[i].value(0.0);
          for (unsigned int j = 0; j < r + 1; ++j)
            // Integration
            for (unsigned int q = 0; q < quad.size(); ++q)
              full_matrix(i, j) += quad.weight(q) *
                                   poly_radau[i].value(quad.point(q)[0]) *
                                   poly_radau[j].value(quad.point(q)[0]);
        }

      FullMatrix<Number> full_matrix_der(r + 1, r + 1);

      for (unsigned int i = 0; i < r + 1; ++i)
        for (unsigned int j = 0; j < r + 1; ++j)
          {
            // Jump
            full_matrix_der(i, j) +=
              poly_radau[i].value(0) * poly_radau[j].value(0);
            // Integration
            for (unsigned int q = 0; q < quad.size(); ++q)
              full_matrix_der(i, j) +=
                quad.weight(q) * poly_radau[i].value(quad.point(q)[0]) *
                poly_radau_derivative[j].value(quad.point(q)[0]);
          }
      print_formatted(jump_matrix);
      print_formatted(full_matrix);
      print_formatted(full_matrix_der);
    }
}

int
main()
{
  test("CG", 1);
  test("CG", 2);
  test("DG", 1);
  test("DG", 2);
}
