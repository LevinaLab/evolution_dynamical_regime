repeats=10
cores=10

# subfolder="plot_many_functions"
subfolder_add="extend_b1_hard_task_by_4000_generations"
subfolder="sim-$(date '+%Y%m%d-%H%M%S')_parallel_$subfolder_add"
# date="date +%s"
# subfolder="$date$subfolder"

command="python3 train.py -g 4001 -t 2000 -no_commands -compress -v_eat_max 0.005 -li 4000 -l sim-20210226-023745_parallel_b1_break_eat_significance_10_runs_delta_every_20_gen/sim-20210226-023747-b_1_-g_4001_-t_2000_-compress_-v_eat_max_0.005_-noplt_-rec_c_20_-c_3_-subfolder_sim-20210226-023745_parallel_b1_break_eat_significance_10_runs_delta_every_20_gen_-n_Run_{1} -rec_c 100 -c 8 -subfolder ${subfolder} -n Run_{1}"



parallel --bar --eta -j${cores} ${command} ::: $(seq ${repeats})

# seq 5 | parallel -n0 ${command} 
