#!/bin/bash

#EXAMPLE INPUT:
#bash bash-heat-capacity-generational-automatic.sh "sim-20200327-215417-g_8000_-b_1_-ref_2000_-a_500_1000_2000_4000_6000_8000_-n_4_sensors" "0 3000" 5


#echo "Choose a simulation: "
#read sim
#echo "Starting Heat Capacity Calculations"


# COMMAND LINE INPUTS:
sim=$1
generations=$2
cores=$3


args=("$@")

gens=(save/$sim/isings/*) 

# find all generations in simulations folder
# for ((i=0; i<${#gens[@]}; i++)); do
	# find the generation number
    # gens[i]=$(echo ${gens[$i]} | cut -d "[" -f2 | cut -d "]" -f1)	
# done

# parallel --bar --eta -j14 "python3 compute-heat-capacity-generational-2 $sim {1} {2}" ::: \
# $(seq 101) ::: \
# ${gens[@]}
#j14 <-- Number of procesors
#0 600 780 1600 1999 

parallel --bar --eta -j${cores} "python3 compute-heat-capacity-generational-2 ${sim} {1} {2}" ::: $(seq 101) ::: $2

#$generations
#0 4000
#1 500 1000 2000
#1 2 3 10 20 30 40 300 600 900 1000 1300 1600 1900 2300 2500 2800 3100 3400 3700 3990

#0 600 780 1600 1999

# parallel --bar 'python3 compute-heat-capacity {1}' ::: $(seq 101)

# NUMC=$numCores
# numCores=12
# i=0
# while [[ $i -lt 101 ]]
# do
# if [[ $NUMC > 0 ]]
# then
	# for ((j=0; j<NUMC; j++)); 
	# do
		# echo "b_bin =" $i
		# python3 compute-heat-capacity $i &
		# ((i++))
	# done
	# NUMC=$(($NUMC-$j))
	# echo "NUMC:" $NUMC

	# for job in `jobs -p`
	# do
		# wait $job && ((NUMC++))
	# done
# fi
# sleep 2
# done	


# for ((i=17; i<=32; i++))
# do
	# python3 compute-heat-capacity $i &
# done
# wait

# for ((i=33; i<=48; i++))
# do
	# python3 compute-heat-capacity $i &
# done
# wait

# for ((i=49; i<=64; i++))
# do
	# python3 compute-heat-capacity $i &
# done
# wait

# for ((i=65; i<=80; i++))
# do
	# python3 compute-heat-capacity $i &
# done
# wait

# for ((i=81; i<=96; i++))
# do
	# python3 compute-heat-capacity $i &
# done
# wait

# for ((i=97; i<=100; i++))
# do
	# python3 compute-heat-capacity $i &
# done
