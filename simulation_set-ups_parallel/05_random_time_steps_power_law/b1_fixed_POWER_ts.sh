repeats=10
cores=10

# subfolder="plot_many_functions"
subfolder_add="b1_fixed_POWER_ts"
subfolder="sim-$(date '+%Y%m%d-%H%M%S')_parallel_$subfolder_add"
# date="date +%s"
# subfolder="$date$subfolder"

command="python3 train.py -b 1 -g 4000 -rand_ts_power -rand_ts_power_lim 100 1000000 700 -rec_c 2000 -c_props 10 10 -2 2 100 40 -c 1 -subfolder ${subfolder} -n Run_{1}"



parallel --bar --eta -j${cores} ${command} ::: $(seq ${repeats})

# seq 5 | parallel -n0 ${command} 