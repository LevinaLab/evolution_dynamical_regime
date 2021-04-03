repeats=4
cores=4

# subfolder="plot_many_functions"
subfolder_add="b10_t10000_g4000_ts_dynamic_range"
subfolder="sim-$(date '+%Y%m%d-%H%M%S')_parallel_$subfolder_add"
# date="date +%s"
# subfolder="$date$subfolder"

command="python3 train.py -b 10 -t 10000 -g 4000 -t 2000 -noplt -subfolder ${subfolder} -n Run_{1}"



parallel --bar --eta -j${cores} ${command} ::: $(seq ${repeats})

# seq 5 | parallel -n0 ${command} 