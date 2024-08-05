#!/bin/bash
source ./generate_parameters.sh

types="DG CGP"
distortCoeffs="0.0 0.6"

# Run the function and store the output filenames
generate_practical_parameters "wave" "$types" "$distortCoeffs" "1 2 4"

source ./submit_job_postprocess.sh

for i in {64,96,128,160,192,224,256}; do
    for file in "${filenames_p[@]}"; do
        if [ $i -gt 256 ]; then
            submit_job "$file" "$i" "large" "12:00:00" "--precon_float"
        else
            submit_job "$file" "$i" "medium" "12:00:00" "--precon_float"
        fi
    done
done
