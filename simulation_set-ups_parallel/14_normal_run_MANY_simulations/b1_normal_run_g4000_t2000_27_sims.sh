repeats=27
cores=27

# subfolder="plot_many_functions"
subfolder_add="b1_normal_run_g4000_t2000_27_sims"
subfolder="sim-$(date '+%Y%m%d-%H%M%S')_parallel_$subfolder_add"
# date="date +%s"
# subfolder="$date$subfolder"

command="python3 train.py -b 1 -g 4001 -t 2000 -compress -noplt -rec_c 2000 -c_props 10 10 -2 2 100 40 -c 1 -subfolder ${subfolder} -n Run_{1}"



parallel --bar --eta -j${cores} ${command} ::: $(seq ${repeats})

# seq 5 | parallel -n0 ${command} 