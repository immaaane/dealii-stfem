#!/bin/bash
declare -a filenames_c=()
declare -a filenames_p=()
# Define a function to generate parameter files and collect filenames
generate_convergence_parameters() {
    local problems="$1"
    local types="$2"
    local distortGrids="$3"
    local sms="$4"
    local dim=3
    local nDegCycles=4
    local nRefCycles=5
    local frequency=2.0
    local extrapolate="--extrapolate"
    local spaceTimeConvergenceTest="--spaceTimeConvergenceTest"
    local printTiming="--printTiming"
    local spaceTimeMg="--spaceTimeMg"
    local rIsTransposeP="--restrictIsTransposeProlongate"
    local estRelax="--relaxation 0.0"
    # Define problems and their respective time types
    for s in $sms; do
        for problem in $problems; do
            for timeType in $types; do
                for distortGrid in $distortGrids; do
                    testNameSuffix=""
                    if [[ $distortGrid != "0.0" ]]; then
                        testNameSuffix="_distort"
                    fi
                    testName="tests/json/convergence${testNameSuffix}_${problem}_${timeType}"
                    filename=$(python tests/json/generate.py --testName $testName --dim $dim $printTiming $spaceTimeMg $rIsTransposeP $estRelax --timeType $timeType --problemType $problem --nDegCycles $nDegCycles --nRefCycles $nRefCycles --frequency $frequency $spaceTimeConvergenceTest $extrapolate --distortGrid $distortGrid --feDegree 2 --smoothingSteps $s)
                    filenames_c+=($filename)
                done
            done
        done
    done
}

generate_practical_parameters() {
    local problems="$1"
    local types="$2"
    local distortCoeffs="$3"
    local sms="$4"
    local dim=3
    local nDegCycles=3
    local T=2
    local nRefCycles=2
    local doOutput="--doOutput"
    local printTiming="--printTiming"
    local spaceTimeMg="--spaceTimeMg"
    local rIsTransposeP="--restrictIsTransposeProlongate"
    local estRelax="--relaxation 0.0"
    # Define problems and their respective time types
    for s in $sms; do
        for problem in $problems; do
            for timeType in $types; do
                for distortC in $distortCoeffs; do
                    testNameSuffix=""
                    if [[ $distortC != "0.0" ]]; then
                        testNameSuffix="_rough"
                    fi
                    testName="tests/json/practical${testNameSuffix}_${problem}_${timeType}"
		                filename=$(python tests/json/generate.py --testName $testName --endTime $T --dim $dim $printTiming $spaceTimeMg $rIsTransposeP $estRelax --timeType $timeType --problemType $problem --nDegCycles $nDegCycles --nRefCycles $nRefCycles --distortCoeff $distortC --feDegree 2  --smoothingSteps $s --refinement 5)
		    filenames_p+=($filename)
                done
            done
        done
    done
}


generate_practical_stokes_parameters() {
    local base_files="$1"
    local types="$2"
    local distortCoeffs="$3"
    local sms="$4"
    local nRef="$5"
    local dim=3
    local nDegCycles=2
    local T=8
    local nRefCycles=2
    local doOutput="--doOutput"
    local extrapolate="--extrapolate"
    local printTiming="--printTiming"
    local spaceTimeMg="--spaceTimeMg"
    local rIsTransposeP="--restrictIsTransposeProlongate"
    local estRelax="--relaxation 0.0"
    local colorizeBd="--colorizeBoundary"
    local stokes="--problemType stokes"
    local degMin="--feDegreeMin 1"
    local ntspMin="--nTimestepsAtOnceMin 1"
    # Define problems and their respective time types
    for s in $sms; do
        for problem in $base_files; do
            for timeType in $types; do
                for distortC in $distortCoeffs; do
                    testNameSuffix=""
                    if [[ $distortC != "0.0" ]]; then
                        testNameSuffix="_rough"
                    fi
                    testName="tests/json/practical${testNameSuffix}_${problem}_${timeType}"
                    filename=$(python tests/json/generate.py --testName $testName --endTime $T --dim $dim $ntspMin $degMin $stokes $extrapolate $colorizeBd $printTiming $spaceTimeMg $rIsTransposeP $estRelax --timeType $timeType --baseFile $problem --nDegCycles $nDegCycles --nRefCycles $nRefCycles --distortCoeff $distortC --feDegree 2  --smoothingSteps $s --refinement $nRef)
                    filenames_p+=($filename)
                done
            done
        done
    done
}

generate_practical_cdr_parameters() {
    local base_files="$1"
    local types="$2"
    local distortCoeffs="$3"
    local sms="$4"
    local nRef="$5"
    local dim=3
    local nDegCycles=2
    local nRefCycles=1
    local doOutput="--doOutput"
    local extrapolate="--extrapolate"
    local printTiming="--printTiming"
    local spaceTimeMg="--spaceTimeMg"
    local rIsTransposeP="--restrictIsTransposeProlongate"
    local stLevelFirst="$6"
    local estRelax="--relaxation 0.0"
    local colorizeBd="--colorizeBoundary"
    local cdr="--problemType cdr"
    local degMin="--feDegreeMin 1"
    local ntspMin="--nTimestepsAtOnceMin 1"
    # Define problems and their respective time types
    for s in $sms; do
        for problem in $base_files; do
            for timeType in $types; do
                for distortC in $distortCoeffs; do
                    testNameSuffix=""
                    if [[ $distortC != "0.0" ]]; then
                        testNameSuffix="_rough"
                    fi
                    testName="tests/json/practical${testNameSuffix}_${problem}_${timeType}"
                    filename=$(python tests/json/generate.py --testName $testName --dim $dim $ntspMin $degMin $cdr $extrapolate $colorizeBd $printTiming $spaceTimeMg $rIsTransposeP $estRelax --timeType $timeType --baseFile $problem --nDegCycles $nDegCycles --nRefCycles $nRefCycles --distortCoeff $distortC --feDegree 2  --smoothingSteps $s --refinement $nRef $stLevelFirst)
                    filenames_p+=($filename)
                done
            done
        done
    done
}
