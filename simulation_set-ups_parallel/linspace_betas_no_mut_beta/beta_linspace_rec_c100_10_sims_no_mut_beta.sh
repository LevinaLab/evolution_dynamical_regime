repeats=10
cores=10

# subfolder="plot_many_functions"
subfolder_add="beta_linspace_rec_c100_10_sims_no_mut_beta"
subfolder="sim-$(date '+%Y%m%d-%H%M%S')_parallel_$subfolder_add"
# date="date +%s"
# subfolder="$date$subfolder"

command="python3 train.py -nmb -b_linspace {1} -1 1 10 -g 2001 -t 2000 -compress -noplt -rec_c 100 -c 6 -subfolder ${subfolder} -n Run_{1}"



parallel --bar --eta -j${cores} ${command} ::: $(seq ${repeats})

# seq 5 | parallel -n0 ${command} 