repeats=2
cores=2

# subfolder="plot_many_functions"
subfolder_add="TEST_SERVER_PLOT"
subfolder="sim-$(date '+%Y%m%d-%H%M%S')_parallel_$subfolder_add"
# date="date +%s"
# subfolder="$date$subfolder"

command="python3 train.py -b 1 -g 80 -t 2 -rec_c 70 -ref 40 -c_props 50 10 -2 2 100 40 -c 20 -subfolder ${subfolder} -n Run_{1}"



parallel --bar --eta -j${cores} ${command} ::: $(seq ${repeats})

# seq 5 | parallel -n0 ${command} 