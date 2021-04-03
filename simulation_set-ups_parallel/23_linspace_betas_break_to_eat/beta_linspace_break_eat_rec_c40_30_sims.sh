repeats=30
cores=30

# subfolder="plot_many_functions"
subfolder_add="beta_linspace_break_eat_rec_c40_30_sims"
subfolder="sim-$(date '+%Y%m%d-%H%M%S')_parallel_$subfolder_add"
# date="date +%s"
# subfolder="$date$subfolder"

command="python3 train.py -b_linspace {1} -1 1.5 30 -g 4001 -t 2000 -v_eat_max 0.005 -compress -noplt -rec_c 40 -c 6 -subfolder ${subfolder} -n Run_{1}"



parallel --bar --eta -j${cores} ${command} ::: $(seq ${repeats})

# seq 5 | parallel -n0 ${command} 