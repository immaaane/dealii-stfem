#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_q.h>

#include "fe_time.h"
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

      auto const [full_matrix, full_matrix_der] = get_cg_weights<Number>(r);
      print_formatted(full_matrix);
      print_formatted(full_matrix_der);
    }

  if (type == "DG")
    { // DG
      using Number = double;

      std::cout << "DG(" << r << ")" << std::endl;
      auto const [full_matrix, full_matrix_der, jump_matrix] =
        get_dg_weights<Number>(r);
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
