repeats=10
cores=10

# subfolder="plot_many_functions"
subfolder_add="b1_rand_seas_g4000_t2000_lim_1_499"
subfolder="sim-$(date '+%Y%m%d-%H%M%S')_parallel_$subfolder_add"
# date="date +%s"
# subfolder="$date$subfolder"

command="python3 train.py -b 1 -g 4000 -t 2000 -rand_seas -rand_seas_lim 1 499 -subfolder ${subfolder} -n Run_{1}"



parallel --bar --eta -j${cores} ${command} ::: $(seq ${repeats})

# seq 5 | parallel -n0 ${command} 