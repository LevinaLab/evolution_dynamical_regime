repeats=20
cores=20

# subfolder="plot_many_functions"
subfolder_add="b_linspace_within_sim_dynamic_range_20_sims_30_gens"
subfolder="sim-$(date '+%Y%m%d-%H%M%S')_parallel_$subfolder_add"
# date="date +%s"
# subfolder="$date$subfolder"

command="python3 train.py -b_linspace_within_sim -g 30 -t 2000 -compress -noplt -rec_c 10 -c 1 -subfolder ${subfolder} -n Run_{1}"



parallel --bar --eta -j${cores} ${command} ::: $(seq ${repeats})

# seq 5 | parallel -n0 ${command} 