#!/bin/bash
source ./generate_parameters.sh

types="DG CGP"
distortCoeffs="0.0 0.6"

# Run the function and store the output filenames
generate_practical_parameters "wave" "$types" "$distortCoeffs"

source ./submit_job_postprocess.sh

for i in {3..9}; do
    nodes=$((2**i))
    time_est=$((24/8*2**i))
    for file in "${filenames_p[@]}"; do
        if [ $i -eq 9 ]; then
            submit_job "$file" "$nodes" "large" "$time_est:00:00" "--precon_float"
        else
            submit_job "$file" "$nodes" "medium" "$time_est:00:00" "--precon_float"

        fi
    done
done
