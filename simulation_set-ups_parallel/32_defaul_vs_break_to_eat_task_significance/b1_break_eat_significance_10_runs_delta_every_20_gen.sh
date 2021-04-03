repeats=20
cores=20

# subfolder="plot_many_functions"
subfolder_add="b1_break_eat_significance_10_runs_delta_every_20_gen"
subfolder="sim-$(date '+%Y%m%d-%H%M%S')_parallel_$subfolder_add"
# date="date +%s"
# subfolder="$date$subfolder"

command="python3 train.py -b 1 -g 4001 -t 2000 -compress -v_eat_max 0.005 -noplt -rec_c 20 -c 3 -subfolder ${subfolder} -n Run_{1}"



parallel --bar --eta -j${cores} ${command} ::: $(seq ${repeats})

# seq 5 | parallel -n0 ${command} 