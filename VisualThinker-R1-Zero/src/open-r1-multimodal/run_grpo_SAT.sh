timestamp=$1
echo "timestamp: ${timestamp}"
pg_name="gpg"
DATA_PATH=SAT
CKPT_PATH=Qwen2-VL-2B

RUN_NAME=${DATA_PATH##*/}_${CKPT_PATH##*/}_${timestamp}
SAVE_PATH="./output/${pg_name}/${RUN_NAME}"
mkdir -p ${SAVE_PATH}
export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./${SAVE_PATH}/debug_log.txt"
# export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS-1)) | sed 's/,$//g')

accelerate launch --config_file=src/open-r1-multimodal/configs/zero2.yaml \
    --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT} \
    --num_machines ${WORLD_SIZE} --machine_rank ${RANK} --num_processes ${GPUS} \
    src/open-r1-multimodal/src/open_r1/grpo.py \
    --pg_name ${pg_name} \
    --output_dir ${SAVE_PATH} \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name ${DATA_PATH} \
    --max_prompt_length 1024 \
    --max_completion_length 700 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --gradient_checkpointing 1 \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name ${RUN_NAME} \
    --save_steps 100 \
    --save_only_model true \
    --report_to tensorboard \
    2>&1 | tee -a "./${SAVE_PATH}/training_log.log"
