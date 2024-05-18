#include <iostream>

#include "include/fe_time.h"

using namespace dealii;
void
run_idx_tests(bool         is_variable_major,
              unsigned int n_timesteps_at_once,
              unsigned int n_variables,
              unsigned int n_timedofs)
{
  block_indexing::set_variable_major(is_variable_major);
  block_indexing indexer(n_timesteps_at_once, n_variables, n_timedofs);

  std::cout << "Testing "
            << (is_variable_major ? "variable-major" : "timedof-major")
            << " layout\n";
  for (unsigned int timestep = 0; timestep < n_timesteps_at_once; ++timestep)
    for (unsigned int variable = 0; variable < n_variables; ++variable)
      for (unsigned int timedof = 0; timedof < n_timedofs; ++timedof)
        {
          unsigned int index = indexer(timestep, variable, timedof);
          auto [dec_timestep, dec_variable, dec_timedof] = indexer(index);
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
  return 0;
}
