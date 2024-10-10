#include <cmath> // for std::log2
#include <iostream>
#include <string>
#include <vector>

#include "fe_time.h"

using namespace dealii;

// Simple test utility
template <typename T>
void
expect_eq(const std::vector<T> &result,
          const std::vector<T> &expected,
          const std::string    &test_name)
{
  if (result.size() != expected.size())
    {
      std::cerr << "[FAIL] " << test_name << ": Size mismatch (expected "
                << expected.size() << ", got " << result.size() << ")"
                << std::endl;
      return;
    }

  for (size_t i = 0; i < result.size(); ++i)
    {
      if (result[i] != expected[i])
        {
          std::cerr << "[FAIL] " << test_name << ": Mismatch at index " << i
                    << " (expected " << static_cast<int>(expected[i])
                    << ", got " << static_cast<int>(result[i]) << ")"
                    << std::endl;
          return;
        }
    }

  std::cout << "[PASS] " << test_name << std::endl;
}

void
run_tests()
{
  // Test 1:
  {
    unsigned int              n_sp_lvl                = 1;
    std::vector<unsigned int> k_seq                   = {1, 2, 4};
    unsigned int              n_timesteps_at_once     = 4;
    unsigned int              n_timesteps_at_once_min = 1;
    MGType                    lower_lvl               = MGType::tau;
    CoarseningType            coarsening_type = CoarseningType::space_and_time;
    bool                      time_before_space = false;

    std::vector<MGType> expected_mg_type_level = {MGType::tau,
                                                  MGType::tau,
                                                  MGType::k,
                                                  MGType::k};

    auto result = get_time_mg_sequence(n_sp_lvl,
                                       k_seq,
                                       n_timesteps_at_once,
                                       n_timesteps_at_once_min,
                                       lower_lvl,
                                       coarsening_type,
                                       time_before_space);
    expect_eq(result,
              expected_mg_type_level,
              "Test 1: No space, lower level tau, then k");
  }
  // Test 2:
  {
    unsigned int              n_sp_lvl                = 1;
    std::vector<unsigned int> k_seq                   = {1, 2};
    unsigned int              n_timesteps_at_once     = 8;
    unsigned int              n_timesteps_at_once_min = 1;
    MGType                    lower_lvl               = MGType::k;
    CoarseningType            coarsening_type = CoarseningType::space_and_time;
    bool                      time_before_space = true;

    std::vector<MGType> expected_mg_type_level = {MGType::k,
                                                  MGType::tau,
                                                  MGType::tau,
                                                  MGType::tau};

    auto result = get_time_mg_sequence(n_sp_lvl,
                                       k_seq,
                                       n_timesteps_at_once,
                                       n_timesteps_at_once_min,
                                       lower_lvl,
                                       coarsening_type,
                                       time_before_space);
    expect_eq(result,
              expected_mg_type_level,
              "Test 2: No space lower level k, then tau");
  }

  // Test 3:
  {
    unsigned int              n_sp_lvl                = 2;
    std::vector<unsigned int> k_seq                   = {1, 2};
    unsigned int              n_timesteps_at_once     = 4;
    unsigned int              n_timesteps_at_once_min = 1;
    MGType                    lower_lvl               = MGType::k;
    CoarseningType            coarsening_type = CoarseningType::space_and_time;
    bool                      time_before_space = false;

    std::vector<MGType> expected_mg_type_level = {MGType::k,
                                                  MGType::tau,
                                                  MGType::tau,
                                                  MGType::h};

    auto result = get_time_mg_sequence(n_sp_lvl,
                                       k_seq,
                                       n_timesteps_at_once,
                                       n_timesteps_at_once_min,
                                       lower_lvl,
                                       coarsening_type,
                                       time_before_space);
    expect_eq(result,
              expected_mg_type_level,
              "Test 3: Interleaving of tau with space, then k");
    auto                p_result   = get_precondition_stmg_types(result,
                                                coarsening_type,
                                                time_before_space,
                                                true);
    std::vector<size_t> expected_p = {1, 1, 1, 0, 1};
    expect_eq(p_result,
              expected_p,
              "Test 3: Interleaving of tau with space, then k");
  }
  // Test 4:
  {
    unsigned int              n_sp_lvl                = 4;
    std::vector<unsigned int> k_seq                   = {1, 2, 3, 4};
    unsigned int              n_timesteps_at_once     = 1;
    unsigned int              n_timesteps_at_once_min = 1;
    MGType                    lower_lvl               = MGType::k;
    CoarseningType            coarsening_type = CoarseningType::space_and_time;
    bool                      time_before_space = false;

    std::vector<MGType> expected_mg_type_level = {
      MGType::k, MGType::h, MGType::k, MGType::h, MGType::k, MGType::h};

    auto result = get_time_mg_sequence(n_sp_lvl,
                                       k_seq,
                                       n_timesteps_at_once,
                                       n_timesteps_at_once_min,
                                       lower_lvl,
                                       coarsening_type,
                                       time_before_space);
    expect_eq(result,
              expected_mg_type_level,
              "Test 4: Equal number of k and space levels");
    auto                p_result   = get_precondition_stmg_types(result,
                                                coarsening_type,
                                                time_before_space,
                                                true);
    std::vector<size_t> expected_p = {1, 0, 1, 0, 1, 0, 1};
    expect_eq(p_result,
              expected_p,
              "Test 4: Equal number of k and space levels");
  }
  // Test 5:
  {
    unsigned int              n_sp_lvl                = 8;
    std::vector<unsigned int> k_seq                   = {1, 2};
    unsigned int              n_timesteps_at_once     = 8;
    unsigned int              n_timesteps_at_once_min = 1;
    MGType                    lower_lvl               = MGType::tau;
    CoarseningType            coarsening_type = CoarseningType::space_and_time;
    bool                      time_before_space = false;

    std::vector<MGType> expected_mg_type_level = {MGType::h,
                                                  MGType::h,
                                                  MGType::h,
                                                  MGType::tau,
                                                  MGType::h,
                                                  MGType::tau,
                                                  MGType::h,
                                                  MGType::tau,
                                                  MGType::h,
                                                  MGType::k,
                                                  MGType::h};

    auto result = get_time_mg_sequence(n_sp_lvl,
                                       k_seq,
                                       n_timesteps_at_once,
                                       n_timesteps_at_once_min,
                                       lower_lvl,
                                       coarsening_type,
                                       time_before_space);
    expect_eq(result,
              expected_mg_type_level,
              "Test 5: Many space levels, some tau and k levels");
    auto                p_result   = get_precondition_stmg_types(result,
                                                coarsening_type,
                                                time_before_space,
                                                true);
    std::vector<size_t> expected_p = {1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1};
    expect_eq(p_result,
              expected_p,
              "Test 5: Many space levels, some tau and k levels");
  }
  // Test 6:
  {
    unsigned int              n_sp_lvl                = 8;
    std::vector<unsigned int> k_seq                   = {1, 2};
    unsigned int              n_timesteps_at_once     = 8;
    unsigned int              n_timesteps_at_once_min = 1;
    MGType                    lower_lvl               = MGType::tau;
    CoarseningType            coarsening_type = CoarseningType::space_and_time;
    bool                      time_before_space = true;

    std::vector<MGType> expected_mg_type_level = {MGType::h,
                                                  MGType::h,
                                                  MGType::h,
                                                  MGType::h,
                                                  MGType::tau,
                                                  MGType::h,
                                                  MGType::tau,
                                                  MGType::h,
                                                  MGType::tau,
                                                  MGType::h,
                                                  MGType::k};

    auto result = get_time_mg_sequence(n_sp_lvl,
                                       k_seq,
                                       n_timesteps_at_once,
                                       n_timesteps_at_once_min,
                                       lower_lvl,
                                       coarsening_type,
                                       time_before_space);
    expect_eq(result,
              expected_mg_type_level,
              "Test 6: Test 5, but time before space");
    auto                p_result   = get_precondition_stmg_types(result,
                                                coarsening_type,
                                                time_before_space,
                                                true);
    std::vector<size_t> expected_p = {1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1};
    expect_eq(p_result, expected_p, "Test 6: Test 5, but time before space");
  }
  // Test 1:
  {
    unsigned int              n_sp_lvl                = 1;
    std::vector<unsigned int> k_seq                   = {1, 2, 4};
    unsigned int              n_timesteps_at_once     = 4;
    unsigned int              n_timesteps_at_once_min = 1;
    MGType                    lower_lvl               = MGType::tau;
    CoarseningType            coarsening_type = CoarseningType::space_and_time;
    bool                      time_before_space = false;

    std::vector<MGType> expected_mg_type_level = {
      MGType::tau, MGType::tau, MGType::k, MGType::p, MGType::k, MGType::p};

    auto result = get_time_mg_sequence(n_sp_lvl,
                                       k_seq,
                                       n_timesteps_at_once,
                                       n_timesteps_at_once_min,
                                       lower_lvl,
                                       coarsening_type,
                                       time_before_space,
                                       true,
                                       true);
    expect_eq(result,
              expected_mg_type_level,
              "Test 1: No space, lower level tau, then k");
    auto                p_result   = get_precondition_stmg_types(result,
                                                coarsening_type,
                                                time_before_space,
                                                true);
    std::vector<size_t> expected_p = {1, 1, 1, 0, 1, 0, 1};
    expect_eq(p_result,
              expected_p,
              "Test 1: No space, lower level tau, then k");
  }
  // Test 2:
  {
    unsigned int              n_sp_lvl                = 1;
    std::vector<unsigned int> k_seq                   = {1, 2};
    unsigned int              n_timesteps_at_once     = 8;
    unsigned int              n_timesteps_at_once_min = 1;
    MGType                    lower_lvl               = MGType::k;
    CoarseningType            coarsening_type = CoarseningType::space_and_time;
    bool                      time_before_space = true;

    std::vector<MGType> expected_mg_type_level = {
      MGType::k, MGType::tau, MGType::tau, MGType::p, MGType::tau};

    auto result = get_time_mg_sequence(n_sp_lvl,
                                       k_seq,
                                       n_timesteps_at_once,
                                       n_timesteps_at_once_min,
                                       lower_lvl,
                                       coarsening_type,
                                       time_before_space,
                                       true,
                                       true);
    expect_eq(result,
              expected_mg_type_level,
              "Test 2: No space lower level k, then tau");
    auto                p_result   = get_precondition_stmg_types(result,
                                                coarsening_type,
                                                time_before_space,
                                                true);
    std::vector<size_t> expected_p = {1, 1, 1, 1, 0, 1};
    expect_eq(p_result, expected_p, "Test 2: No space lower level k, then tau");
  }
  // Test 3:
  {
    unsigned int              n_sp_lvl                = 2;
    std::vector<unsigned int> k_seq                   = {1, 2};
    unsigned int              n_timesteps_at_once     = 4;
    unsigned int              n_timesteps_at_once_min = 1;
    MGType                    lower_lvl               = MGType::k;
    CoarseningType            coarsening_type = CoarseningType::space_and_time;
    bool                      time_before_space = false;

    std::vector<MGType> expected_mg_type_level = {
      MGType::k, MGType::tau, MGType::p, MGType::tau, MGType::h};

    auto result = get_time_mg_sequence(n_sp_lvl,
                                       k_seq,
                                       n_timesteps_at_once,
                                       n_timesteps_at_once_min,
                                       lower_lvl,
                                       coarsening_type,
                                       time_before_space,
                                       true,
                                       true);
    expect_eq(result,
              expected_mg_type_level,
              "Test 3: Interleaving of tau with space, then k");
    auto                p_result   = get_precondition_stmg_types(result,
                                                coarsening_type,
                                                time_before_space,
                                                true);
    std::vector<size_t> expected_p = {1, 1, 0, 1, 0, 1};
    expect_eq(p_result,
              expected_p,
              "Test 3: Interleaving of tau with space, then k");
  }
  // Test 3a:
  {
    unsigned int              n_sp_lvl                = 2;
    std::vector<unsigned int> k_seq                   = {1, 2};
    unsigned int              n_timesteps_at_once     = 4;
    unsigned int              n_timesteps_at_once_min = 1;
    MGType                    lower_lvl               = MGType::tau;
    CoarseningType            coarsening_type = CoarseningType::space_and_time;
    bool                      time_before_space = false;

    std::vector<MGType> expected_mg_type_level = {
      MGType::tau, MGType::tau, MGType::h, MGType::k, MGType::p};

    auto result = get_time_mg_sequence(n_sp_lvl,
                                       k_seq,
                                       n_timesteps_at_once,
                                       n_timesteps_at_once_min,
                                       lower_lvl,
                                       coarsening_type,
                                       time_before_space,
                                       true,
                                       true);
    expect_eq(result,
              expected_mg_type_level,
              "Test 3a: Interleaving of tau with space, then k");
    auto                p_result   = get_precondition_stmg_types(result,
                                                coarsening_type,
                                                time_before_space,
                                                true);
    std::vector<size_t> expected_p = {1, 1, 0, 1, 0, 1};
    expect_eq(p_result,
              expected_p,
              "Test 3a: Interleaving of tau with space, then k");
  }
  // Test 4:
  {
    unsigned int              n_sp_lvl                = 4;
    std::vector<unsigned int> k_seq                   = {1, 2, 3, 4};
    unsigned int              n_timesteps_at_once     = 1;
    unsigned int              n_timesteps_at_once_min = 1;
    MGType                    lower_lvl               = MGType::k;
    CoarseningType            coarsening_type = CoarseningType::space_and_time;
    bool                      time_before_space = false;

    std::vector<MGType> expected_mg_type_level = {MGType::p,
                                                  MGType::p,
                                                  MGType::p,
                                                  MGType::k,
                                                  MGType::h,
                                                  MGType::k,
                                                  MGType::h,
                                                  MGType::k,
                                                  MGType::h};

    auto result = get_time_mg_sequence(n_sp_lvl,
                                       k_seq,
                                       n_timesteps_at_once,
                                       n_timesteps_at_once_min,
                                       lower_lvl,
                                       coarsening_type,
                                       time_before_space,
                                       true,
                                       true);
    expect_eq(result,
              expected_mg_type_level,
              "Test 4: Equal number of k and space levels");
    auto                p_result   = get_precondition_stmg_types(result,
                                                coarsening_type,
                                                time_before_space,
                                                true);
    std::vector<size_t> expected_p = {1, 1, 1, 1, 0, 1, 0, 1, 0, 1};
    expect_eq(p_result,
              expected_p,
              "Test 4: Equal number of k and space levels");
  }
  // Test 4a:
  {
    unsigned int              n_sp_lvl                = 4;
    std::vector<unsigned int> k_seq                   = {1, 2, 3, 4};
    unsigned int              n_timesteps_at_once     = 1;
    unsigned int              n_timesteps_at_once_min = 1;
    MGType                    lower_lvl               = MGType::tau;
    CoarseningType            coarsening_type = CoarseningType::space_and_time;
    bool                      time_before_space = false;

    std::vector<MGType> expected_mg_type_level = {MGType::h,
                                                  MGType::h,
                                                  MGType::h,
                                                  MGType::k,
                                                  MGType::p,
                                                  MGType::k,
                                                  MGType::p,
                                                  MGType::k,
                                                  MGType::p};

    auto result = get_time_mg_sequence(n_sp_lvl,
                                       k_seq,
                                       n_timesteps_at_once,
                                       n_timesteps_at_once_min,
                                       lower_lvl,
                                       coarsening_type,
                                       time_before_space,
                                       true,
                                       true);
    expect_eq(result,
              expected_mg_type_level,
              "Test 4a: Equal number of k and space levels");
    auto                p_result   = get_precondition_stmg_types(result,
                                                coarsening_type,
                                                time_before_space,
                                                true);
    std::vector<size_t> expected_p = {1, 1, 1, 1, 0, 1, 0, 1, 0, 1};
    expect_eq(p_result,
              expected_p,
              "Test 4a: Equal number of k and space levels");
  }
  // Test 5:
  {
    unsigned int              n_sp_lvl                = 8;
    std::vector<unsigned int> k_seq                   = {1, 2};
    unsigned int              n_timesteps_at_once     = 8;
    unsigned int              n_timesteps_at_once_min = 1;
    MGType                    lower_lvl               = MGType::tau;
    CoarseningType            coarsening_type = CoarseningType::space_and_time;
    bool                      time_before_space = false;

    std::vector<MGType> expected_mg_type_level = {MGType::h,
                                                  MGType::h,
                                                  MGType::h,
                                                  MGType::h,
                                                  MGType::tau,
                                                  MGType::h,
                                                  MGType::tau,
                                                  MGType::h,
                                                  MGType::tau,
                                                  MGType::h,
                                                  MGType::k,
                                                  MGType::p};

    auto result = get_time_mg_sequence(n_sp_lvl,
                                       k_seq,
                                       n_timesteps_at_once,
                                       n_timesteps_at_once_min,
                                       lower_lvl,
                                       coarsening_type,
                                       time_before_space,
                                       true,
                                       true);
    expect_eq(result,
              expected_mg_type_level,
              "Test 5: Many space levels, some tau and k levels");
    auto                p_result   = get_precondition_stmg_types(result,
                                                coarsening_type,
                                                time_before_space,
                                                true);
    std::vector<size_t> expected_p = {1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1};
    expect_eq(p_result,
              expected_p,
              "Test 5: Many space levels, some tau and k levels");
  }
  // Test 6:
  {
    unsigned int              n_sp_lvl                = 8;
    std::vector<unsigned int> k_seq                   = {1, 2};
    unsigned int              n_timesteps_at_once     = 8;
    unsigned int              n_timesteps_at_once_min = 1;
    MGType                    lower_lvl               = MGType::tau;
    CoarseningType            coarsening_type = CoarseningType::space_and_time;
    bool                      time_before_space = true;

    std::vector<MGType> expected_mg_type_level = {MGType::h,
                                                  MGType::h,
                                                  MGType::h,
                                                  MGType::h,
                                                  MGType::h,
                                                  MGType::tau,
                                                  MGType::h,
                                                  MGType::tau,
                                                  MGType::h,
                                                  MGType::tau,
                                                  MGType::p,
                                                  MGType::k};

    auto result = get_time_mg_sequence(n_sp_lvl,
                                       k_seq,
                                       n_timesteps_at_once,
                                       n_timesteps_at_once_min,
                                       lower_lvl,
                                       coarsening_type,
                                       time_before_space,
                                       true,
                                       true);
    expect_eq(result,
              expected_mg_type_level,
              "Test 6: Test 5, but time before space");
    auto                p_result   = get_precondition_stmg_types(result,
                                                coarsening_type,
                                                time_before_space,
                                                true);
    std::vector<size_t> expected_p = {1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1};
    expect_eq(p_result, expected_p, "Test 6: Test 5, but time before space");
  }
  // Test 1:
  {
    unsigned int              n_sp_lvl                = 1;
    std::vector<unsigned int> k_seq                   = {1, 2, 4};
    unsigned int              n_timesteps_at_once     = 4;
    unsigned int              n_timesteps_at_once_min = 1;
    MGType                    lower_lvl               = MGType::tau;
    CoarseningType            coarsening_type = CoarseningType::space_and_time;
    bool                      time_before_space = true;

    std::vector<MGType> expected_mg_type_level = {
      MGType::tau, MGType::p, MGType::tau, MGType::p, MGType::k, MGType::k};

    auto result = get_time_mg_sequence(n_sp_lvl,
                                       k_seq,
                                       n_timesteps_at_once,
                                       n_timesteps_at_once_min,
                                       lower_lvl,
                                       coarsening_type,
                                       time_before_space,
                                       true,
                                       false);
    expect_eq(result,
              expected_mg_type_level,
              "Test 1: No space, lower level tau, then k");
    auto                p_result   = get_precondition_stmg_types(result,
                                                coarsening_type,
                                                time_before_space,
                                                false);
    std::vector<size_t> expected_p = {1, 1, 0, 1, 0, 1, 1};
    expect_eq(p_result,
              expected_p,
              "Test 1: No space, lower level tau, then k");
  }
  // Test 2:
  {
    unsigned int              n_sp_lvl                = 1;
    std::vector<unsigned int> k_seq                   = {1, 2};
    unsigned int              n_timesteps_at_once     = 8;
    unsigned int              n_timesteps_at_once_min = 1;
    MGType                    lower_lvl               = MGType::k;
    CoarseningType            coarsening_type = CoarseningType::space_and_time;
    bool                      time_before_space = true;

    std::vector<MGType> expected_mg_type_level = {
      MGType::k, MGType::p, MGType::tau, MGType::tau, MGType::tau};

    auto result = get_time_mg_sequence(n_sp_lvl,
                                       k_seq,
                                       n_timesteps_at_once,
                                       n_timesteps_at_once_min,
                                       lower_lvl,
                                       coarsening_type,
                                       time_before_space,
                                       true,
                                       false);
    expect_eq(result,
              expected_mg_type_level,
              "Test 2: No space lower level k, then tau");
    auto                p_result   = get_precondition_stmg_types(result,
                                                coarsening_type,
                                                time_before_space,
                                                false);
    std::vector<size_t> expected_p = {1, 1, 0, 1, 1, 1};
    expect_eq(p_result, expected_p, "Test 2: No space lower level k, then tau");
  }
  // Test 3:
  {
    unsigned int              n_sp_lvl                = 2;
    std::vector<unsigned int> k_seq                   = {1, 2};
    unsigned int              n_timesteps_at_once     = 4;
    unsigned int              n_timesteps_at_once_min = 1;
    MGType                    lower_lvl               = MGType::k;
    CoarseningType            coarsening_type = CoarseningType::space_and_time;
    bool                      time_before_space = true;

    std::vector<MGType> expected_mg_type_level = {
      MGType::k, MGType::p, MGType::tau, MGType::h, MGType::tau};

    auto result = get_time_mg_sequence(n_sp_lvl,
                                       k_seq,
                                       n_timesteps_at_once,
                                       n_timesteps_at_once_min,
                                       lower_lvl,
                                       coarsening_type,
                                       time_before_space,
                                       true,
                                       false);
    expect_eq(result,
              expected_mg_type_level,
              "Test 3: Interleaving of tau with space, then k");
    auto                p_result   = get_precondition_stmg_types(result,
                                                coarsening_type,
                                                time_before_space,
                                                false);
    std::vector<size_t> expected_p = {1, 1, 0, 1, 0, 1};
    expect_eq(p_result,
              expected_p,
              "Test 3: Interleaving of tau with space, then k");
  }
  // Test 3a:
  {
    unsigned int              n_sp_lvl                = 2;
    std::vector<unsigned int> k_seq                   = {1, 2};
    unsigned int              n_timesteps_at_once     = 4;
    unsigned int              n_timesteps_at_once_min = 1;
    MGType                    lower_lvl               = MGType::tau;
    CoarseningType            coarsening_type = CoarseningType::space_and_time;
    bool                      time_before_space = true;

    std::vector<MGType> expected_mg_type_level = {
      MGType::tau, MGType::h, MGType::tau, MGType::p, MGType::k};

    auto result = get_time_mg_sequence(n_sp_lvl,
                                       k_seq,
                                       n_timesteps_at_once,
                                       n_timesteps_at_once_min,
                                       lower_lvl,
                                       coarsening_type,
                                       time_before_space,
                                       true,
                                       false);
    expect_eq(result,
              expected_mg_type_level,
              "Test 3a: Interleaving of tau with space, then k");
    auto                p_result   = get_precondition_stmg_types(result,
                                                coarsening_type,
                                                time_before_space,
                                                false);
    std::vector<size_t> expected_p = {1, 1, 0, 1, 0, 1};
    expect_eq(p_result,
              expected_p,
              "Test 3a: Interleaving of tau with space, then k");
  }
  // Test 4:
  {
    unsigned int              n_sp_lvl                = 4;
    std::vector<unsigned int> k_seq                   = {1, 2, 3, 4};
    unsigned int              n_timesteps_at_once     = 1;
    unsigned int              n_timesteps_at_once_min = 1;
    MGType                    lower_lvl               = MGType::k;
    CoarseningType            coarsening_type = CoarseningType::space_and_time;
    bool                      time_before_space = true;

    std::vector<MGType> expected_mg_type_level = {MGType::k,
                                                  MGType::p,
                                                  MGType::k,
                                                  MGType::p,
                                                  MGType::k,
                                                  MGType::p,
                                                  MGType::h,
                                                  MGType::h,
                                                  MGType::h};

    auto result = get_time_mg_sequence(n_sp_lvl,
                                       k_seq,
                                       n_timesteps_at_once,
                                       n_timesteps_at_once_min,
                                       lower_lvl,
                                       coarsening_type,
                                       time_before_space,
                                       true,
                                       false);
    expect_eq(result,
              expected_mg_type_level,
              "Test 4: Equal number of k and space levels");
    auto                p_result   = get_precondition_stmg_types(result,
                                                coarsening_type,
                                                time_before_space,
                                                false);
    std::vector<size_t> expected_p = {1, 1, 0, 1, 0, 1, 1, 1, 1, 1};
    expect_eq(p_result,
              expected_p,
              "Test 4: Equal number of k and space levels");
  }
  // Test 4a:
  {
    unsigned int              n_sp_lvl                = 4;
    std::vector<unsigned int> k_seq                   = {1, 2, 3, 4};
    unsigned int              n_timesteps_at_once     = 1;
    unsigned int              n_timesteps_at_once_min = 1;
    MGType                    lower_lvl               = MGType::tau;
    CoarseningType            coarsening_type = CoarseningType::space_and_time;
    bool                      time_before_space = true;

    std::vector<MGType> expected_mg_type_level = {MGType::k,
                                                  MGType::h,
                                                  MGType::k,
                                                  MGType::h,
                                                  MGType::k,
                                                  MGType::h,
                                                  MGType::p,
                                                  MGType::p,
                                                  MGType::p};

    auto result = get_time_mg_sequence(n_sp_lvl,
                                       k_seq,
                                       n_timesteps_at_once,
                                       n_timesteps_at_once_min,
                                       lower_lvl,
                                       coarsening_type,
                                       time_before_space,
                                       true,
                                       false);
    expect_eq(result,
              expected_mg_type_level,
              "Test 4a: Equal number of k and space levels");
    auto                p_result   = get_precondition_stmg_types(result,
                                                coarsening_type,
                                                time_before_space,
                                                false);
    std::vector<size_t> expected_p = {1, 1, 0, 1, 0, 1, 1, 1, 1, 1};
    expect_eq(p_result,
              expected_p,
              "Test 4a: Equal number of k and space levels");
  }
  // Test 5:
  {
    unsigned int              n_sp_lvl                = 8;
    std::vector<unsigned int> k_seq                   = {1, 2};
    unsigned int              n_timesteps_at_once     = 8;
    unsigned int              n_timesteps_at_once_min = 1;
    MGType                    lower_lvl               = MGType::tau;
    CoarseningType            coarsening_type = CoarseningType::space_and_time;
    bool                      time_before_space = true;

    std::vector<MGType> expected_mg_type_level = {MGType::tau,
                                                  MGType::h,
                                                  MGType::tau,
                                                  MGType::h,
                                                  MGType::tau,
                                                  MGType::h,
                                                  MGType::k,
                                                  MGType::h,
                                                  MGType::h,
                                                  MGType::h,
                                                  MGType::h,
                                                  MGType::p};

    auto result = get_time_mg_sequence(n_sp_lvl,
                                       k_seq,
                                       n_timesteps_at_once,
                                       n_timesteps_at_once_min,
                                       lower_lvl,
                                       coarsening_type,
                                       time_before_space,
                                       true,
                                       false);
    expect_eq(result,
              expected_mg_type_level,
              "Test 5: Many space levels, some tau and k levels");
    auto                p_result   = get_precondition_stmg_types(result,
                                                coarsening_type,
                                                time_before_space,
                                                false);
    std::vector<size_t> expected_p = {1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1};
    expect_eq(p_result,
              expected_p,
              "Test 5: Many space levels, some tau and k levels");
  }
  // Test 6:
  {
    unsigned int              n_sp_lvl                = 8;
    std::vector<unsigned int> k_seq                   = {1, 2};
    unsigned int              n_timesteps_at_once     = 8;
    unsigned int              n_timesteps_at_once_min = 1;
    MGType                    lower_lvl               = MGType::tau;
    CoarseningType            coarsening_type = CoarseningType::space_and_time;
    bool                      time_before_space = true;

    std::vector<MGType> expected_mg_type_level = {MGType::tau,
                                                  MGType::h,
                                                  MGType::tau,
                                                  MGType::h,
                                                  MGType::tau,
                                                  MGType::h,
                                                  MGType::k,
                                                  MGType::h,
                                                  MGType::h,
                                                  MGType::h,
                                                  MGType::h,
                                                  MGType::p};

    auto result = get_time_mg_sequence(n_sp_lvl,
                                       k_seq,
                                       n_timesteps_at_once,
                                       n_timesteps_at_once_min,
                                       lower_lvl,
                                       coarsening_type,
                                       time_before_space,
                                       true,
                                       false);
    expect_eq(result,
              expected_mg_type_level,
              "Test 6: Test 5, but time before space");
    auto                p_result   = get_precondition_stmg_types(result,
                                                coarsening_type,
                                                time_before_space,
                                                false);
    std::vector<size_t> expected_p = {1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1};
    expect_eq(p_result, expected_p, "Test 6: Test 5, but time before space");
  }
}


void
run_idx_tests(bool         is_variable_major,
              unsigned int n_timesteps_at_once,
              unsigned int n_variables,
              unsigned int n_timedofs)
{
  block_indexing::set_variable_major(is_variable_major);
  BlockSlice indexer(n_timesteps_at_once, n_variables, n_timedofs);

  std::cout << "Testing "
            << (is_variable_major ? "variable-major" : "timedof-major")
            << " layout\n";
  for (unsigned int timestep = 0; timestep < n_timesteps_at_once; ++timestep)
    for (unsigned int variable = 0; variable < n_variables; ++variable)
      for (unsigned int timedof = 0; timedof < n_timedofs; ++timedof)
        {
          unsigned int index = indexer.index(timestep, variable, timedof);
          auto [dec_timestep, dec_variable, dec_timedof] =
            indexer.decompose(index);
          std::cout << "Computed Index: " << index << " Decomposed: "
                    << "Timestep: " << dec_timestep
                    << ", variable: " << dec_variable
                    << ", timedof: " << dec_timedof;
          std::cout << (dec_timestep == timestep && dec_variable == variable &&
                            dec_timedof == timedof ?
                          " [PASS]" :
                          " [FAIL]")
                    << std::endl;
        }
  for (unsigned int timestep = 0; timestep < n_timesteps_at_once; ++timestep)
    for (unsigned int timedof = 0; timedof < n_timedofs; ++timedof)
      {
        std::vector<unsigned int> l(n_variables);
        unsigned int              n =
          -n_timedofs + timedof + timestep * n_timedofs * n_variables;
        std::generate(l.begin(), l.end(), [&] { return n += n_timedofs; });
        auto const vv = indexer.get_variable(timestep, timedof);
        std::cout << "get_variable: " << (l == vv ? " [PASS]" : " [FAIL]")
                  << std::endl;
      }
}

int
main()
{
  run_idx_tests(true, 2, 3, 4);
  run_idx_tests(true, 1, 1, 4);
  run_idx_tests(true, 2, 1, 2);
  run_idx_tests(true, 1, 1, 1);
  run_idx_tests(true, 1, 1, 2);
  run_idx_tests(true, 2, 2, 2);
  // Has no effect, but we should figure out a way to test both
  run_idx_tests(false, 2, 3, 4);

  run_tests();

  return 0;
}
