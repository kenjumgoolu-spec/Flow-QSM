export CUDA_VISIBLE_DEVICES=0,2,3
MODEL_CHECKPOINT="checkpoints/mp_rank_00_model_states.pt"
MODEL_CONFIG="./config/model.json"
SCHEDULER_CONFIG="./config/flow_scheduler.json"


NUM_INFERENCE_STEPS=8
NOISE_STEP=50
WINDOW_SIZE="64 64 64"
STRIDE="32 32 32"
B0_DIR="0.0 0.0 1.0"
PIX_DIM="1.0 1.0 1.0"
DEVICE="cuda"
TYPE="flow"

echo "Starting QSM inference processing (True 4-GPU Task Parallel)..."
echo "Using model checkpoint: $MODEL_CHECKPOINT"
echo "Processing ${#INPUT_MASK_PAIRS[@]} input files"

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPUs"

if [ $NUM_GPUS -ge 4 ]; then
    echo "Using true 4-GPU parallel inference"
    python inference.py \
        --config-file ./config/test.json \
        --model_checkpoint "$MODEL_CHECKPOINT" \
        --model_config "$MODEL_CONFIG" \
        --scheduler_config "$SCHEDULER_CONFIG" \
        --num_inference_steps $NUM_INFERENCE_STEPS \
        --noise_step $NOISE_STEP \
        --window_size $WINDOW_SIZE \
        --stride $STRIDE \
        --device $DEVICE \
        --distributed
    
    exit_code=$?
    rm -f "$CONFIG_FILE"
    echo "All files processed in parallel!"
    
else
    echo "Warning: Only $NUM_GPUS GPUs available, using sequential single GPU mode"
    for input_file in "${!INPUT_MASK_PAIRS[@]}"; do
        mask_file="${INPUT_MASK_PAIRS[$input_file]}"
        echo "Processing file: $input_file with mask: $mask_file"
        
        python inference.py \
            --input "$input_file" \
            --mask "$mask_file" \
            --model_checkpoint "$MODEL_CHECKPOINT" \
            --model_config "$MODEL_CONFIG" \
            --scheduler_config "$SCHEDULER_CONFIG" \
            --num_inference_steps $NUM_INFERENCE_STEPS \
            --noise_step $NOISE_STEP \
            --window_size $WINDOW_SIZE \
            --stride $STRIDE \
            --B0_dir $B0_DIR \
            --pix_dim $PIX_DIM \
            --device $DEVICE \
            --type $TYPE
        
        exit_code=$?
        if [ $exit_code -ne 0 ]; then
            echo "Error processing file: $input_file (exit code: $exit_code)"
            exit 1
        fi
        
        echo "Successfully processed: $input_file"
    done
    
    echo "All files processed sequentially!"
fi

echo "Results saved in: ./inference_results/$(basename $(dirname $MODEL_CHECKPOINT))/"