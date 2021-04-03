repeats=27
cores=27

# subfolder="plot_many_functions"
subfolder_add="respone_plot_gen_100_heat_cap"
subfolder="sim-$(date '+%Y%m%d-%H%M%S')_parallel_$subfolder_add"
# date="date +%s"
# subfolder="$date$subfolder"

command="python3 train.py -g 2 -t 2000 -no_commands -compress -li 100 -l sim-20201119-190135_parallel_b1_normal_run_g4000_t2000_27_sims/sim-20201119-190137-b_1_-g_4001_-t_2000_-compress_-noplt_-rec_c_2000_-c_props_10_10_-2_2_100_40_-c_1_-subfolder_sim-20201119-190135_parallel_b1_normal_run_g4000_t2000_27_sims_-n_Run_{1} -rec_c 1 -c 3 -noplt -subfolder ${subfolder} -n Run_{1}"



parallel --bar --eta -j${cores} ${command} ::: $(seq ${repeats})

# seq 5 | parallel -n0 ${command} 