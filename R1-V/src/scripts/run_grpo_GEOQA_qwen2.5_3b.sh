timestamp=$1
echo "timestamp: ${timestamp}"
pg_name="gpg"
adjust_gd="true"
min_inverse_alpha="0.4"

# Wandb
export WANDB_PROJECT="R1-V"

DATA_PATH=leonardPKU/GEOQA_R1V_Train_8K
CKPT_PATH=Qwen2.5-VL-3B-Instruct

RUN_NAME=${DATA_PATH##*/}_${CKPT_PATH##*/}_${timestamp}
SAVE_PATH="./output/${pg_name}/${RUN_NAME}"

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./${SAVE_PATH}/debug_log.txt"

mkdir -p ${SAVE_PATH}

torchrun --master_addr ${MASTER_ADDR} --master-port ${MASTER_PORT} \
    --nnodes ${WORLD_SIZE} --node_rank ${RANK} --nproc-per-node=${GPUS} \
    src/r1-v/src/open_r1/grpo.py \
    --output_dir ${SAVE_PATH} \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name ${DATA_PATH} \
    --deepspeed src/r1-v/local_scripts/zero3.json \
    --pg_name ${pg_name} \
    --adjust_gd ${adjust_gd} \
    --min_inverse_alpha ${min_inverse_alpha} \
    --max_prompt_length 1024 \
    --max_completion_length 256 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name "${RUN_NAME}" \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 8 \
    --learning_rate 1e-6 \
    2>&1 | tee -a "./${SAVE_PATH}/training_log.log"
