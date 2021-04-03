repeats=10
cores=10

# subfolder="plot_many_functions"
subfolder_add="b1_push_evolved_simple_into_complex_foraging_task"
subfolder="sim-$(date '+%Y%m%d-%H%M%S')_parallel_$subfolder_add"
# date="date +%s"
# subfolder="$date$subfolder"

command="python3 train.py -g 2001 -no_commands -compress -v_eat_max 0.005 -li 4000 -l sim-20201210-200605_parallel_b1_dynamic_range_c_20_g4000_t2000_10_sims/sim-20201210-200606-b_1_-g_4001_-t_2000_-compress_-noplt_-rec_c_20_-c_4_-subfolder_sim-20201210-200605_parallel_b1_dynamic_range_c_20_g4000_t2000_10_sims_-n_Run_{1} -rec_c 20 -c 8 -subfolder ${subfolder} -n Run_{1}"



parallel --bar --eta -j${cores} ${command} ::: $(seq ${repeats})

# seq 5 | parallel -n0 ${command} 