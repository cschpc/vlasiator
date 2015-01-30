## Define test and runs
#vlasiator binary

bin=vlasiator
if [ ! -f $bin ]
then
   echo Executable $bin does not exist
   exit
fi

# where the tests are run
run_dir="run"

# where the directories for different tests, including cfg and other needed data files are located 
test_dir="tests"

# choose tests to run
run_tests=( 1 )

# test 1
test_name[1]="acctest_1_maxw_500k_30kms_1deg"
comparison_vlsv[1]="fullf.0000001.vlsv"
#only one process does anything -> in _1 phiprof here
comparison_phiprof[1]="phiprof_full_1.txt"


# define here the variables you want to be tested
variables_name=( "rho" "rho_v" "rho_v" "rho_v" "B" "B" "B" "E" "E" "E" "avgs" )
# and the corresponding components to variables, 
variables_components=( 0 0 1 2 0 1 2 0 1 2 0)
#arrays variables_name and variables_components should have same number of elements, e.g., 4th variable is variables_name[4] variables_components[4]=  

