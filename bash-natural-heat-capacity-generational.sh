#!/bin/bash

echo "Choose a simulation: "
read sim
echo "Starting Heat Capacity Calculations"

gens=(save/$sim/isings/*) 

# parallel --bar --eta -j14 "python3 compute-heat-capacity-generational-2 $sim {1} {2}" ::: \
# $(seq 101) ::: \
# ${gens[@]}

parallel --bar --eta -j10 "python3 compute-natural-heat-capacity-generational $sim {1} {2}" ::: \
$(seq 101) ::: \
0  4000
#0 1000 1999 3000 3999

# parallel --bar 'python3 compute-heat-capacity {1}' ::: $(seq 101)