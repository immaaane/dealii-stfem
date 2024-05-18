// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by Nils Margenberg and Peter Munch

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
      auto [Alpha_lhs, Beta_lhs, rhs_uK, rhs_uM, rhs_vM] =
        get_fe_time_weights_wave(
          TimeStepType::CGP, Alpha_, Beta_, Gamma_, Zeta_);
      std::cout << "Waves" << std::endl;
      print_formatted(Alpha_lhs);
      print_formatted(Beta_lhs);
      print_formatted(rhs_uK);
      print_formatted(rhs_uM);
      print_formatted(rhs_vM);
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
      auto [Alpha_lhs, Beta_lhs, rhs_uK, rhs_uM, rhs_vM] =
        get_fe_time_weights_wave(
          TimeStepType::DG, full_matrix, full_matrix_der, jump_matrix, nil);
      std::cout << "Waves" << std::endl;
      print_formatted(Alpha_lhs);
      print_formatted(Beta_lhs);
      print_formatted(rhs_uK);
      print_formatted(rhs_uM);
      print_formatted(rhs_vM);
    }
}


void
test2(TimeStepType type, const unsigned int r, unsigned int n_timesteps_at_once)
{
  std::cout << (type == TimeStepType::CGP ? "CG(" : "DG(") << r << ") - "
            << n_timesteps_at_once << " timesteps in one system" << std::endl;
  auto [Alpha, Beta, Gamma, Zeta] =
    get_fe_time_weights<double>(type, r, 1.0, n_timesteps_at_once);
  print_formatted(Alpha);
  print_formatted(Beta);
  print_formatted(Gamma);
  print_formatted(Zeta);
  auto [Alpha_1, Beta_1, Gamma_1, Zeta_1] =
    get_fe_time_weights<double>(type, r, 1.0, 1);
  auto [lhs_uK, lhs_uM, rhs_uK, rhs_uM, rhs_vM] =
    get_fe_time_weights_wave<double>(
      type, Alpha_1, Beta_1, Gamma_1, Zeta_1, n_timesteps_at_once);
  std::cout << "Waves " << (type == TimeStepType::CGP ? "CG(" : "DG(") << r
            << ") - " << n_timesteps_at_once << " timesteps in one system"
            << std::endl;
  print_formatted(lhs_uK);
  print_formatted(lhs_uM);
  print_formatted(rhs_uK);
  print_formatted(rhs_uM);
  print_formatted(rhs_vM);
}



void
test3(TimeStepType type, const unsigned int r, unsigned int n_timesteps_at_once)
{
  auto [Alpha, Beta, Gamma, Zeta] =
    get_fe_time_weights_stokes<double>(type, r, 1.0, n_timesteps_at_once);
  std::cout << "Stokes " << (type == TimeStepType::CGP ? "CG(" : "DG(") << r
            << ") - " << n_timesteps_at_once << " timesteps in one system"
            << std::endl;
  print_formatted(Alpha);
  print_formatted(Beta);
  print_formatted(Gamma);
  print_formatted(Zeta);
}

int
main()
{
  block_indexing::set_variable_major(true);
  test(TimeStepType::CGP, 1);
  test(TimeStepType::CGP, 2);
  test(TimeStepType::CGP, 3);
  test(TimeStepType::CGP, 4);
  test(TimeStepType::CGP, 5);
  test(TimeStepType::DG, 1);
  test(TimeStepType::DG, 2);
  test(TimeStepType::DG, 3);
  test(TimeStepType::DG, 4);
  test(TimeStepType::DG, 5);

  test2(TimeStepType::CGP, 1, 2);
  test2(TimeStepType::CGP, 2, 2);
  test2(TimeStepType::DG, 1, 2);
  test2(TimeStepType::DG, 2, 2);
  test2(TimeStepType::CGP, 1, 3);
  test2(TimeStepType::CGP, 2, 3);
  test2(TimeStepType::DG, 1, 3);
  test2(TimeStepType::DG, 2, 3);
  test2(TimeStepType::CGP, 1, 4);
  test2(TimeStepType::CGP, 2, 4);
  test2(TimeStepType::DG, 1, 4);
  test2(TimeStepType::DG, 2, 4);

  test3(TimeStepType::CGP, 1, 1);
  test3(TimeStepType::DG, 1, 1);
  test3(TimeStepType::CGP, 2, 1);
  test3(TimeStepType::DG, 2, 1);
  test3(TimeStepType::CGP, 3, 1);
  test3(TimeStepType::DG, 3, 1);
  test3(TimeStepType::CGP, 4, 1);
  test3(TimeStepType::DG, 4, 1);
  test3(TimeStepType::CGP, 1, 1);
  test3(TimeStepType::CGP, 2, 1);
  test3(TimeStepType::DG, 1, 1);
  test3(TimeStepType::DG, 2, 1);
  test3(TimeStepType::CGP, 1, 2);
  test3(TimeStepType::CGP, 2, 2);
  test3(TimeStepType::DG, 1, 2);
  test3(TimeStepType::DG, 2, 2);
  test3(TimeStepType::CGP, 1, 4);
  test3(TimeStepType::CGP, 2, 4);
  test3(TimeStepType::DG, 1, 4);
  test3(TimeStepType::DG, 2, 4);
}
