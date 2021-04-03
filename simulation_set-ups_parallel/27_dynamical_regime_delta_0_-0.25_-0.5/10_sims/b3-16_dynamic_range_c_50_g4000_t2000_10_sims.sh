repeats=10
cores=10

# subfolder="plot_many_functions"
subfolder_add="b3-16_dynamic_range_c_50_g4000_t2000_10_sims"
subfolder="sim-$(date '+%Y%m%d-%H%M%S')_parallel_$subfolder_add"
# date="date +%s"
# subfolder="$date$subfolder"

command="python3 train.py -b 3.16 -g 2001 -t 2000 -compress -noplt -rec_c 50 -c 3 -subfolder ${subfolder} -n Run_{1}"



parallel --bar --eta -j${cores} ${command} ::: $(seq ${repeats})

# seq 5 | parallel -n0 ${command} 