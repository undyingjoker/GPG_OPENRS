timestamp=$1
echo "timestamp: ${timestamp}"
pg_name="grpo"
# pg_name="gpg"

DATA_PATH=laolao77/ViRFT_CLS_flower_4_shot
CKPT_PATH=Qwen2-VL-2B-Instruct

RUN_NAME=${DATA_PATH##*/}_${CKPT_PATH##*/}_${timestamp}
SAVE_PATH="./output/${pg_name}/${RUN_NAME}"

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./${SAVE_PATH}/debug_log.txt"

mkdir -p ${SAVE_PATH}

torchrun --master_addr ${MASTER_ADDR} --master-port ${MASTER_PORT} \
    --nnodes ${WORLD_SIZE} --node_rank ${RANK} --nproc-per-node=${GPUS} \
    src/virft/src/open_r1/grpo_classification.py \
    --output_dir ${SAVE_PATH}  \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name ${DATA_PATH} \
    --deepspeed src/virft/local_scripts/zero3.json \
    --pg_name ${pg_name} \
    --temperature 0.9 \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to tensorboard \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 8 \
    --run_name "${RUN_NAME}" \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 8 \
    2>&1 | tee -a "./${SAVE_PATH}/training_log.log"
