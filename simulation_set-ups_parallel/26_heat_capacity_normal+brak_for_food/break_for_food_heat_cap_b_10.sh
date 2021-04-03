repeats=10
cores=10

# subfolder="plot_many_functions"
subfolder_add="break_for_food_heat_cap_b_10"
subfolder="sim-$(date '+%Y%m%d-%H%M%S')_parallel_$subfolder_add"
# date="date +%s"
# subfolder="$date$subfolder"

command="python3 train.py -g 2 -t 2000 -no_commands -compress -li 3999 -l sim-20201226-111308_parallel_b10_break_eat_v_eat_max_0_005_g4000_t2000_10_sims/sim-20201226-111312-b_10_-g_4001_-t_2000_-compress_-v_eat_max_0.005_-noplt_-rec_c_2000_-c_props_10_10_-2_2_100_40_-c_1_-subfolder_sim-20201226-111308_parallel_b10_break_eat_v_eat_max_0_005_g4000_t2000_10_sims_-n_Run_{1} -rec_c 1 -c 5 -c_props 500 10 -2 2 300 40 -c 2 -noplt -subfolder ${subfolder} -n Run_{1}"



parallel --bar --eta -j${cores} ${command} ::: $(seq ${repeats})

# seq 5 | parallel -n0 ${command} 