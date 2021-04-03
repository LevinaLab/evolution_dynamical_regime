

In this simulation, organisms are evolved (by an EA) to perform well at a foraging game of adjustable difficulty in a 2D environment. The organisms are controlled by a neural network based on the Ising model (INN). These neural network allows the organsisms to easily evolve into different dynamical regimes (by changing their inverse temperature Beta), which can be measured (with the measure Delta). Further this module provides different experiment set-ups to test the generalizability, ability to solve complex task as well as the evolvability of the populations of organisms.

The research results created with this repository can be accessed under: https://arxiv.org/abs/2103.12184





## Running a Simulation
To run your own simulation, simply run:
```
python3 train.py
```
For the numerous options of the model can be displayed using:
```
python3 train.py -h
```
For a quick demonstration, which creates a playable movie of the last generation run:

```
python3 train.py -g 51 -t 100 -a 50
```

Moreover for the statistical validity of the results, simulations can be run in parallel.
In the folder "simulation_set-ups_parallel" many suggestions on different parallel simulation set-ups can be accessed.
Note, that the number of simulations you parallely run (controlled by the first two lines in the bash script) should not exceed the number of available cores on your CPU. For parallel runs we recommend, to always use the arguments "-compress", which reduces the file size and "-noplt" which skips the result plotting for each run of the simulation.

For all scripts, the working directory should be located in the same directory, that "train.py" as well as the directory "save" are located in.

Bear in mind that the settings in the `train` python script will need to be edited to your own preferences and directories.

##Experiments
The experiments of the paper can be reproduced by running the scripts in the folders "pipelines_for_paper_experiments" as well as "make_paper_figures". Again make sure, that the working directory is set to where the dir "save" is located

Most Figures of the paper can be reproduced immediately just by running the scripts (with the suffix "with_data") in the folder "make_paper_figures" as we included their undelying data in the repo.

If you want to run an experiment from scratch follow the instructions below:

Generally an experiment consists of either 2 or 3 steps:
1. Run a set of simulations with a bash script, which executes "train.py", similar to those in the directory "simulation_set-ups_parallel". (Usually ~12-48h)
2. In cases of the experiment for Figure 6 and 7 an experiment pipeline has to be executed with the previously generated simulations (in folder "pipelines_for_paper_experiments"). Give your pipeline run a unique name (settings, add_save_file_name) (Usually ~12-48h)
3. Now you can run the plotting script (in folder "make_paper_figures"). When executing the plotting script the first time switch the "only_plot" toggle to False to process the simulation data
 and subsequently to True to skip the processing and speed up the process by loading a previously generated preprocessed file. Make sure, that in the settings the folder names, and in case step 2 was required, the include name in the settings corresponds the name of the pipeline run, you previously used.


## Ising Class
This project is based upon the following repository
https://github.com/heysoos/CriticalForagingOrgs

The Ising class is defined in the `embodied_ising.py` file. This file was originally forked from the project:

https://github.com/MiguelAguilera/Criticality-as-It-Could-Be

and its associated arXiv link:

https://arxiv.org/abs/1704.05255

This file has been heavily modified, retro-fitted, and mutated to generalize the simulations done in the "Criticality as it Could Be" project. Originally this project was looking at learning criticality in a single agent playing a simple game. Instead we are evolving agents and subsequently measure how close they are to the critical regime.


## Evolutionary Algorithm (EA)
The genotypes of an individual ising-embodied organism is defined by the connectivity matrix of its neural network and its local Beta (inverse temperature) (which is the control parameter pressing the network towards a certain dynamical regime). Starting with randomly generated neural networks for each organism, the community is allowed to evolve for a given number of discrete generations. A combination of **elitism** methods to duplicate (with mutations) the top organisms that have consumed the most food as well as **crossover** mating interactions.


## Research Goals
This project is motivated by the criticality hypothesis. The latter states that evolution presses biological systems into a dynamical regime close to the critical point between order and disorder as this provides to be beneficial to those systems. While some research concludes, that biological systems are poised exactly at the critical point, other studies find, that being rather in the sub-critical regime is beneficial for biological systems. It is believed, that the dynamical regime plays a big role in the context of a system's generalizability, ability to solve complex tasks as well as its evolvability, however the underlying dynamics are not yet fully understood. We wanted to find out, how and why these properties differ depending on the dynamical regime, that a population evolved into.