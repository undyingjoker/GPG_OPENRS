export MASTER_ADDR=127.0.0.1
export MASTER_PORT=21231
export WORLD_SIZE=1
export RANK=0
export GPUS=2

timestamp=$(date "+%Y%m%d%H%M%S")

OMP_NUM_THREADS=4 bash ./scripts/run_grpo_clevr.sh ${timestamp}
