#!/bin/bash
array=(random_moving bump_and_rotate curiosity_planner_fixed exploration_planner_fixed rotate_in_place space_filling_curves )
for i in "${array[@]}"
do
	python train.py --train_path "/home/rene/thesis/imgs/experiment_$i"
	mv checkpoints checkpoint_$i
done
