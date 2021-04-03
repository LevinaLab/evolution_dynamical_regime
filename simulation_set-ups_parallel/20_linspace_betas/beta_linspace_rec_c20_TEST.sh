repeats=3
cores=3

# subfolder="plot_many_functions"
subfolder_add="beta_linspace_rec_c20_TEST"
subfolder="sim-$(date '+%Y%m%d-%H%M%S')_parallel_$subfolder_add"
# date="date +%s"
# subfolder="$date$subfolder"

command="python3 train.py -b_linspace {1} -1 1 3 -g 20 -t 2 -compress -c_props 2 2 -2 2 100 40 -noplt -rec_c 10 -c 3 -subfolder ${subfolder} -n Run_{1}"



parallel --bar --eta -j${cores} ${command} ::: $(seq ${repeats})

# seq 5 | parallel -n0 ${command} 