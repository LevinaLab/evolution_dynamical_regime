repeats=10
cores=10

# subfolder="plot_many_functions"
subfolder_add="b1_num_neurons28_g4000_t2000"
subfolder="sim-$(date '+%Y%m%d-%H%M%S')_parallel_$subfolder_add"
# date="date +%s"
# subfolder="$date$subfolder"

command="python3 train.py -b 1 -rec_c 2000 -num_neurons 28 -g 4000 -t 2000 -noplt -subfolder ${subfolder} -n Run_{1}"



parallel --bar --eta -j${cores} ${command} ::: $(seq ${repeats})

# seq 5 | parallel -n0 ${command} 