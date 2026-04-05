#!/bin/bash

# training
python train.py --exp_name SelectiveCL --divide Seen
python train.py --exp_name SelectiveCL --divide Unseen
