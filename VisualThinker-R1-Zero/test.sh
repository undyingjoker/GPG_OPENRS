export MASTER_ADDR=127.0.0.1
export MASTER_PORT=21232
export WORLD_SIZE=1
export RANK=0
export GPUS=2

timestamp=$(date "+%Y%m%d%H%M%S")

# OMP_NUM_THREADS=4 bash ./src/scripts/2B_base65cate_6k.sh ${timestamp}

# OMP_NUM_THREADS=4 bash ./src/scripts/2B_aircraft_4_shot.sh ${timestamp}

# OMP_NUM_THREADS=4 bash ./src/scripts/2B_lisa_grounding.sh ${timestamp}
OMP_NUM_THREADS=4 bash ./src/open-r1-multimodal/run_grpo_SAT.sh ${timestamp}
