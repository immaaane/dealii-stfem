import json
import os
from argparse import ArgumentParser

def run_instance(counter, options, subdivisions, source_point, lower_left, upper_right):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/practical01.json", 'r') as f:
       datastore = json.load(f)

    datastore["doOutput"] = options.doOutput
    datastore["printTiming"] = options.printTiming
    datastore["spaceTimeMg"] = options.spaceTimeMg
    datastore["mgTimeBeforeSpace"] = options.mgTimeBeforeSpace
    datastore["timeType"] = options.timeType
    datastore["problemType"] = options.problemType
    datastore["nTimestepsAtOnce"] = options.nTimestepsAtOnce
    datastore["nTimestepsAtOnceMin"] = options.nTimestepsAtOnceMin
    datastore["feDegree"] = options.feDegree
    datastore["feDegreeMin"] = options.feDegreeMin
    datastore["nDegCycles"] = options.nDegCycles
    datastore["nRefCycles"] = options.nRefCycles
    datastore["frequency"] = options.frequency
    datastore["refinement"] = options.refinement
    datastore["spaceTimeConvergenceTest"]= options.spaceTimeConvergenceTest
    datastore["extrapolate"] = options.extrapolate
    datastore["functionalFile"] = options.functionalFile
    datastore["distortGrid"] = options.distortGrid
    datastore["distortCoeff"] = options.distortCoeff
    datastore["endTime"] = options.endTime
    datastore["subdivisions"] = subdivisions
    datastore["sourcePoint"] = source_point
    datastore["hyperRectLowerLeft"] = lower_left
    datastore["hyperRectUpperRight"] = upper_right
    datastore["smoothingDegree"] = options.smoothingDegree
    datastore["estimateRelaxation"] = options.estimateRelaxation
    datastore["coarseGridSmootherType"] = options.coarseGridSmootherType
    datastore["coarseGridMaxiter"] = options.coarseGridMaxiter
    datastore["coarseGridAbstol"] = options.coarseGridAbstol
    datastore["coarseGridReltol"] = options.coarseGridReltol
    datastore["restrictIsTransposeProlongate"] = options.restrictIsTransposeProlongate
    datastore["variable"] = options.variable

    # write data to output file
    with open(f"./{options.testName}_{str(counter).zfill(4)}.json", 'w') as f:
        json.dump(datastore, f, indent=4, separators=(',', ': '))

def parseArguments():
    parser = ArgumentParser(description="Submit a simulation as a batch job")
    parser.add_argument("--testName", default="input");
    parser.add_argument("--dim", type=int, default=3);
    parser.add_argument("--doOutput", action="store_false");
    parser.add_argument("--printTiming", action="store_false");
    parser.add_argument("--spaceTimeMg", action="store_true");
    parser.add_argument("--mgTimeBeforeSpace", action="store_false");
    parser.add_argument("--timeType", default="DG");
    parser.add_argument("--problemType", default="wave");
    parser.add_argument("--nTimestepsAtOnce", type=int, default=1);
    parser.add_argument("--nTimestepsAtOnceMin", type=int, default=-1);
    parser.add_argument("--feDegree", type=int, default=1);
    parser.add_argument("--feDegreeMin", type=int, default=-1);
    parser.add_argument("--nDegCycles", type=int, default=1);
    parser.add_argument("--nRefCycles", type=int, default=1);
    parser.add_argument("--frequency", type=float, default=1.0);
    parser.add_argument("--refinement", type=int, default=2);
    parser.add_argument("--spaceTimeConvergenceTest", action="store_true");
    parser.add_argument("--extrapolate", action="store_true");
    parser.add_argument("--functionalFile", default="functionals.txt");
    parser.add_argument("--distortGrid", type=float, default=0.0);
    parser.add_argument("--distortCoeff", type=float, default=0.0);
    parser.add_argument("--endTime", type=float, default=1.0);
    parser.add_argument("--smoothingDegree", type=int, default=5);
    parser.add_argument("--estimateRelaxation", action="store_true");
    parser.add_argument("--coarseGridSmootherType", default="Smoother");
    parser.add_argument("--coarseGridMaxiter", type=int, default=10);
    parser.add_argument("--coarseGridAbstol", type=float, default=1.e-20);
    parser.add_argument("--coarseGridReltol", type=float, default=1.e-4);
    parser.add_argument("--restrictIsTransposeProlongate", action="store_true");
    parser.add_argument("--variable", action="store_true");

    arguments = parser.parse_args()
    return arguments

def main():
    options = parseArguments()

    if options.dim==3:
        if not options.spaceTimeConvergenceTest:
            subdivisions="5,5,5"
            source_point="0.0,0.0,0.0"
            lower_left="-1.0,-1.0,-1.0"
            upper_right="1.0,1.0,1.0"
        else:
            subdivisions="1,1,1"
            source_point="0.0,0.0,0.0"
            lower_left="0.0,0.0,0.0"
            upper_right="1.0,1.0,1.0"
    else:
        if not options.spaceTimeConvergenceTest:
            subdivisions="5,5"
            source_point="0.0,0.0"
            lower_left="-1.0,-1.0"
            upper_right="1.0,1.0"
        else:
            subdivisions="1,1"
            source_point="0.0,0.0"
            lower_left="0.0,0.0"
            upper_right="1.0,1.0"

    counter=0
    initial_refinement=options.refinement

    for n_refinements in range(2,8):
        options.refinement=initial_refinement + n_refinements
        run_instance(counter, options, subdivisions, source_point, lower_left, upper_right)
        counter = counter + 1


if __name__== "__main__":
  main()
