#!/bin/bash

# Script to reproduce results

for ((i=0;i<6;i+=1))
do 
	python main.py \
	--policy "TD3" \
	--env "HalfCheetah-v3" \
	--seed $i \
	--save_model
done
