#!/bin/bash

for file in run_commands/4_clip_coef/*;
  do
      source $file;
done
for file in run_commands/5_learning_rate/*;
  do
      source $file;
done

for file in run_commands/6_max_grad_norm/*;
  do
      source $file;
done
