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
test_t(TimeStepType type, unsigned int const r_src)
{
  using Number = double;
  std::cout << "- Prolongation\n"
            << (type == TimeStepType::CGP ? "CG(" : "DG(") << r_src << ")"
            << std::endl;
  auto const prol = get_time_prolongation_matrix(type, r_src);
  print_formatted(prol);
  std::cout << "- Restriction\n"
            << (type == TimeStepType::CGP ? "CG(" : "DG(") << r_src << ")"
            << std::endl;
  auto const rest = get_time_restriction_matrix(type, r_src);
  print_formatted(rest);
  std::cout << std::endl;
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
}
