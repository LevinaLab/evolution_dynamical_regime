repeats=10
cores=10

# subfolder="plot_many_functions"
subfolder_add="b1_rand_ts_AND_rand_food_2000ts_100foods"
subfolder="sim-$(date '+%Y%m%d-%H%M%S')_parallel_$subfolder_add"
# date="date +%s"
# subfolder="$date$subfolder"

command="python3 train.py -b 1 -g 2000 -rand_ts -rand_ts_lim 100 3900 -rand_seas -rand_seas_lim 1 199 -energies 0 100 500 1000 2000 3000 3999 -rec_c 999 -c_props 10 10 -2 2 100 40 -c 1 -subfolder ${subfolder} -n Run_{1}"



parallel --bar --eta -j${cores} ${command} ::: $(seq ${repeats})

# seq 5 | parallel -n0 ${command} 