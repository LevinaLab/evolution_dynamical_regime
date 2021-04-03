repeats=1
cores=1

# subfolder="plot_many_functions"
subfolder_add="b1_dynamic_range_c_20_g4000_t2000_1_SIM_replace_errorprone_Run4"
subfolder="sim-$(date '+%Y%m%d-%H%M%S')_parallel_$subfolder_add"
# date="date +%s"
# subfolder="$date$subfolder"

command="python3 train.py -b 1 -g 4001 -t 2000 -compress -noplt -rec_c 20 -c 10 -subfolder ${subfolder} -n Run_{1}"



parallel --bar --eta -j${cores} ${command} ::: $(seq ${repeats})

# seq 5 | parallel -n0 ${command} 