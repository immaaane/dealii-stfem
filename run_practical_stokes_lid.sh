#!/bin/bash
source ./generate_parameters.sh

# Run the function and store the output filenames
generate_practical_stokes_parameters "tf05stokes.json" "DG" "0.0" "1 2 4" "4"

source ./submit_job_postprocess.sh

for i in {64,128,192,256,320,384}; do
    for file in "${filenames_p[@]}"; do
        if [ $i -gt 6 ]; then
            submit_job "$file" "$i" "medium-s" "24:00:00" "--precon_float" "./tests/tp_03stokes.release/tp_03stokes.release"
        elif [ $i -gt 32  ]; then
            submit_job "$file" "$i" "medium-m" "24:00:00" "--precon_float" "./tests/tp_03stokes.release/tp_03stokes.release"
        elif [ $i -gt 64 ]; then
            submit_job "$file" "$i" "medium-l" "24:00:00" "--precon_float" "./tests/tp_03stokes.release/tp_03stokes.release"
        elif [ $i -gt 256 ]; then
            submit_job "$file" "$i" "large" "24:00:00" "--precon_float" "./tests/tp_03stokes.release/tp_03stokes.release"
        else
            submit_job "$file" "$i" "medium" "24:00:00" "--precon_float" "./tests/tp_03stokes.release/tp_03stokes.release"
        fi
    done
done
