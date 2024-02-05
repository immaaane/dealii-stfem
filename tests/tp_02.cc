#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_q.h>

#include "include/fe_time.h"

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
test(TimeStepType type, const unsigned int r)
{
  if (type == TimeStepType::CGP)
    { // CG
      using Number = double;

      std::cout << "CG(" << r << ")" << std::endl;

      auto const matrix = get_cg_weights<Number>(r);
      print_formatted(matrix[0]);
      print_formatted(matrix[1]);

      auto [Alpha_, Beta_, Gamma_, Zeta_] = split_lhs_rhs(matrix);
      auto [Alpha_lhs, Beta_lhs, rhs_uK_, rhs_uM_, rhs_vM_] =
        get_fe_time_weights_wave(
          TimeStepType::CGP, Alpha_, Beta_, Gamma_, Zeta_);

      print_formatted(Alpha_lhs);
      print_formatted(Beta_lhs);
      print_formatted(rhs_uK_);
      print_formatted(rhs_uM_);
      print_formatted(rhs_vM_);
    }

  if (type == TimeStepType::DG)
    { // DG
      using Number = double;

      std::cout << "DG(" << r << ")" << std::endl;
      auto const [full_matrix, full_matrix_der, jump_matrix] =
        get_dg_weights<Number>(r);
      print_formatted(jump_matrix);
      print_formatted(full_matrix);
      print_formatted(full_matrix_der);
      FullMatrix<Number> nil;
      auto [Alpha_lhs, Beta_lhs, rhs_uK_, rhs_uM_, rhs_vM_] =
        get_fe_time_weights_wave(
          TimeStepType::DG, full_matrix, full_matrix_der, jump_matrix, nil);
      print_formatted(Alpha_lhs);
      print_formatted(Beta_lhs);
      print_formatted(rhs_uK_);
      print_formatted(rhs_uM_);
      print_formatted(rhs_vM_);
    }
}

int
main()
{
  test(TimeStepType::CGP, 1);
  test(TimeStepType::CGP, 2);
  test(TimeStepType::DG, 1);
  test(TimeStepType::DG, 2);
}
