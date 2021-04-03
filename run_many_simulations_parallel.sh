repeats=3
cores=3

# subfolder="plot_many_functions_compressed"
subfolder_add="Test_dynamic_range_param_many_heat_caps_compressed"
subfolder="sim-$(date '+%Y%m%d-%H%M%S')_parallel_$subfolder_add"
# date="date +%s"
# subfolder="$date$subfolder"

command="python3 train.py -b 10 -g 9 -t 10 -c 10 -rec_c 3 -compress -c_props 3 5 -2 2 100 40 -c 8 -noplt -subfolder ${subfolder} -n Run_{1}"



parallel --bar --eta -j${cores} ${command} ::: $(seq ${repeats})

# seq 5 | parallel -n0 ${command} 