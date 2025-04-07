ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/zero2.yaml \
  --num_processes=16 \
  src/open_r1/gpg.py \
  --config recipes/gpg.yaml & >> open-rs1-gpg.log