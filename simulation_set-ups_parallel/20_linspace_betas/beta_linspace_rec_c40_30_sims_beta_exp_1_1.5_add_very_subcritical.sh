repeats=8
cores=8

# subfolder="plot_many_functions"
subfolder_add="beta_linspace_rec_c40_30_sims_beta_exp_1_1.5_add_very_subcritical"
subfolder="sim-$(date '+%Y%m%d-%H%M%S')_parallel_$subfolder_add"
# date="date +%s"
# subfolder="$date$subfolder"

command="python3 train.py -b_linspace {1} 1 1.5 8 -g 4001 -t 2000 -compress -noplt -rec_c 80 -c 11 -subfolder ${subfolder} -n Run_{1}"



parallel --bar --eta -j${cores} ${command} ::: $(seq ${repeats})

# seq 5 | parallel -n0 ${command} 