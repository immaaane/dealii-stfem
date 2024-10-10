// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by Nils Margenberg and Peter Munch

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
test_fe_q_perm(unsigned int r)
{
  auto    quad_time = QGaussLobatto<1>(r + 1);
  FE_Q<1> fe_time(quad_time.get_points());
  auto    perm = get_fe_q_permutation(fe_time);
  for (auto it = perm.begin() + 1; it != perm.end(); ++it)
    if (fe_time.unit_support_point(*std::prev(it))[0] >
        fe_time.unit_support_point(*it)[0])
      std::cout << "FE_Q permutation wrong." << std::endl;
}

void
test_r(TimeStepType       type,
       unsigned int const r_src,
       unsigned int const r_dst,
       unsigned int const n_timesteps_at_once)
{
  using Number = double;
  std::cout << "- Projection\n"
            << (type == TimeStepType::CGP ? "CG " : "DG ") << "From " << r_src
            << " to " << r_dst << "\nTimesteps at once: " << n_timesteps_at_once
            << std::endl;
  auto tpr =
    get_time_projection_matrix(type, r_src, r_dst, n_timesteps_at_once);
  print_formatted(tpr);
  std::cout << std::endl;
}

void
test_t(TimeStepType       type,
       unsigned int const r_src,
       unsigned int const n_timesteps_at_once = 2)
{
  using Number = double;
  std::cout << "- Prolongation\n"
            << (type == TimeStepType::CGP ? "CG(" : "DG(") << r_src << ")"
            << std::endl;
  auto const prol =
    get_time_prolongation_matrix(type, r_src, n_timesteps_at_once);
  print_formatted(prol);
  std::cout << "- Restriction\n"
            << (type == TimeStepType::CGP ? "CG(" : "DG(") << r_src << ")"
            << std::endl;
  auto const rest =
    get_time_restriction_matrix(type, r_src, n_timesteps_at_once);
  print_formatted(rest);
  std::cout << std::endl;
}

void
test_tw(TimeStepType       type,
        unsigned int const r,
        unsigned int const n_timesteps_at_once = 4)
{
  std::cout << "Test MG in time operators\n";
  std::vector<MGType> mg_type_level = get_time_mg_sequence(
    1, r, type == TimeStepType::DG ? 0 : 1, n_timesteps_at_once, 1, MGType::k);
  auto fetw = get_fe_time_weights<double>(
    type, r, 0.25, n_timesteps_at_once, mg_type_level);
  auto fetw_wave = get_fe_time_weights_wave<double>(
    type, r, 0.25, n_timesteps_at_once, mg_type_level);
  for (auto const &el : fetw)
    {
      std::cout << "- Alpha heat\n";
      print_formatted(el[0]);
      std::cout << "- Beta heat\n";
      print_formatted(el[1]);
      std::cout << "- Gamma heat\n";
      print_formatted(el[2]);
      std::cout << "- Zeta heat\n";
      print_formatted(el[3]);
    }
  for (auto const &el : fetw_wave)
    {
      std::cout << "- Alpha wave u\n";
      print_formatted(el[0]);
      std::cout << "- Beta wave u\n";
      print_formatted(el[1]);
      std::cout << "- Gamma wave u\n";
      print_formatted(el[2]);
      std::cout << "- Zeta wave u\n";
      print_formatted(el[3]);
      std::cout << "- Gamma wave v\n";
      print_formatted(el[4]);
    }
}

int
main()
{
  for (int i = 1; i < 6; ++i)
    test_fe_q_perm(i);
  for (int i = 1; i < 6; ++i)
    test_t(TimeStepType::CGP, i);

  for (int i = 0; i < 6; ++i)
    test_t(TimeStepType::DG, i);

  for (int i = 2; i < 6; ++i)
    {
      test_r(TimeStepType::CGP, i - 1, i, 1);
      test_r(TimeStepType::CGP, i, i - 1, 1);
    }

  for (int i = 1; i < 6; ++i)
    {
      test_r(TimeStepType::DG, i - 1, i, 1);
      test_r(TimeStepType::DG, i, i - 1, 1);
    }

  test_r(TimeStepType::CGP, 1, 2, 2);
  test_r(TimeStepType::CGP, 2, 1, 2);
  test_r(TimeStepType::CGP, 2, 3, 2);
  test_r(TimeStepType::CGP, 3, 2, 2);

  test_r(TimeStepType::DG, 0, 1, 2);
  test_r(TimeStepType::DG, 1, 0, 2);
  test_r(TimeStepType::DG, 1, 2, 2);
  test_r(TimeStepType::DG, 2, 1, 2);

  test_t(TimeStepType::CGP, 1, 4);
  test_t(TimeStepType::CGP, 2, 4);
  test_t(TimeStepType::CGP, 3, 4);

  test_t(TimeStepType::DG, 0, 4);
  test_t(TimeStepType::DG, 1, 4);
  test_t(TimeStepType::DG, 2, 4);

  test_tw(TimeStepType::DG, 4, 4);
  test_tw(TimeStepType::CGP, 4, 4);
}
