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

int
main()
{
  using Number         = double;
  const unsigned int r = 2;

  // Lagrange polynomial
  const auto poly = Polynomials::generate_complete_Lagrange_basis(
    QGaussLobatto<1>(r + 1).get_points());

  // Radau quadrature
  const auto poly_radau = Polynomials::generate_complete_Lagrange_basis(
    QGaussRadau<1>(r, QGaussRadau<1>::EndPoint::right).get_points());

  std::vector< Polynomials::Polynomial< double > > poly_radau_derivative(poly_radau.size());

  for(unsigned int i = 0; i < poly_radau.size(); ++i)
    poly_radau_derivative[i] = poly_radau[i].derivative ();

  QGauss<1> quad(r + 2);

  FullMatrix<Number> full_matrix(r + 1, r);

  for (unsigned int i = 0; i < r + 1; ++i)
    for (unsigned int j = 0; j < r; ++j)
      for (unsigned int q = 0; q < quad.size(); ++q)
        full_matrix(i, j) += quad.weight(q) * poly[i].value(quad.point(q)[0]) *
                             poly_radau[j].value(quad.point(q)[0]);

  FullMatrix<Number> full_matrix_der(r + 1, r);

  for (unsigned int i = 0; i < r + 1; ++i)
    for (unsigned int j = 0; j < r; ++j)
      for (unsigned int q = 0; q < quad.size(); ++q)
        full_matrix_der(i, j) += quad.weight(q) * poly[i].value(quad.point(q)[0]) *
                             poly_radau_derivative[j].value(quad.point(q)[0]);

  print_formatted(full_matrix);
  print_formatted(full_matrix_der);
}