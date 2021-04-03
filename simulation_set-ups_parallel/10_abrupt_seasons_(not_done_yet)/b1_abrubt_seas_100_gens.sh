repeats=10
cores=10

# subfolder="plot_many_functions"
subfolder_add="b1_abrubt_seas_100_gens"
subfolder="sim-$(date '+%Y%m%d-%H%M%S')_parallel_$subfolder_add"
# date="date +%s"
# subfolder="$date$subfolder"

command="python3 train.py -b 1 -g 2000 -t 2000 -aseas_len 100 -energies 0 100 500 1000 1999 -subfolder ${subfolder} -n Run_{1}"



parallel --bar --eta -j${cores} ${command} ::: $(seq ${repeats})

# seq 5 | parallel -n0 ${command} 