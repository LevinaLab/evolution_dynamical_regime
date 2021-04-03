repeats=10
cores=10

# subfolder="plot_many_functions"
subfolder_add="normal_seas_heat_cap_b_10"
subfolder="sim-$(date '+%Y%m%d-%H%M%S')_parallel_$subfolder_add"
# date="date +%s"
# subfolder="$date$subfolder"

command="python3 train.py -g 2 -t 2000 -no_commands -compress -li 3999 -l sim-20201022-190615_parallel_b10_normal_seas_g4000_t2000/sim-20201022-190618-b_10_-g_4000_-t_2000_-noplt_-subfolder_sim-20201022-190615_parallel_b10_normal_seas_g4000_t2000_-n_Run_{1} -rec_c 1 -c 5 -c_props 500 10 -2 2 300 40 -c 2 -noplt -subfolder ${subfolder} -n Run_{1}"



parallel --bar --eta -j${cores} ${command} ::: $(seq ${repeats})

# seq 5 | parallel -n0 ${command} 