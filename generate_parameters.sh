#!/bin/bash
declare -a filenames_c=()
declare -a filenames_p=()
# Define a function to generate parameter files and collect filenames
generate_convergence_parameters() {
    local problems="$1"
    local types="$2"
    local distortGrids="$3"
    local dim=3
    local nDegCycles=4
    local nRefCycles=5
    local frequency=2.0
    local extrapolate="--extrapolate"
    local spaceTimeConvergenceTest="--spaceTimeConvergenceTest"
    local printTiming="--printTiming"
    local spaceTimeMg="--spaceTimeMg"
    local rIsTransposeP="--restrictIsTransposeProlongate"
    local estRelax="--estimateRelaxation"
    # Define problems and their respective time types
    for problem in $problems; do
        for timeType in $types; do
            for distortGrid in $distortGrids; do
                testNameSuffix=""
                if [[ $distortGrid != "0.0" ]]; then
                    testNameSuffix="_distort"
                fi
                testName="tests/json/convergence${testNameSuffix}_${problem}_${timeType}"
                filename=$(python tests/json/generate.py --testName $testName --dim $dim $printTiming $spaceTimeMg $rIsTransposeP $estRelax --timeType $timeType --problemType $problem --nDegCycles $nDegCycles --nRefCycles $nRefCycles --frequency $frequency $spaceTimeConvergenceTest $extrapolate --distortGrid $distortGrid --feDegree 2 --variable)
                filenames_c+=($filename)
            done
        done
    done
}

generate_practical_parameters() {
    local problems="$1"
    local types="$2"
    local distortCoeffs="$3"
    local dim=3
    local nDegCycles=5

    local nRefCycles=5
    local doOutput="--doOutput"
    local printTiming="--printTiming"
    local spaceTimeMg="--spaceTimeMg"
    local rIsTransposeP="--restrictIsTransposeProlongate"
    local estRelax="--estimateRelaxation"
    # Define problems and their respective time types
    for problem in $problems; do
        for timeType in $types; do
            for distortC in $distortCoeffs; do
                testNameSuffix=""
                if [[ $distortC != "0.0" ]]; then
                    testNameSuffix="_rough"
                fi
                testName="tests/json/practical${testNameSuffix}_${problem}_${timeType}"
                filename=$(python tests/json/generate.py --testName $testName --dim $dim $printTiming $spaceTimeMg $rIsTransposeP $estRelax --timeType $timeType --problemType $problem --nDegCycles $nDegCycles --nRefCycles $nRefCycles --distortCoeff $distortC --feDegree 3 --variable)
                filenames_p+=($filename)
            done
        done
    done
}
