repeats=10
cores=10

# subfolder="plot_many_functions"
subfolder_add="old_b10_fixed_ts_ver_10_reps_4050ts"
subfolder="sim-$(date '+%Y%m%d-%H%M%S')_parallel_$subfolder_add"
# date="date +%s"
# subfolder="$date$subfolder"

command="python3 train.py -b 10 -g 2000 -t 4050 -subfolder ${subfolder} -n Run_{1}"



parallel --bar --eta -j${cores} ${command} ::: $(seq ${repeats})

# seq 5 | parallel -n0 ${command} 