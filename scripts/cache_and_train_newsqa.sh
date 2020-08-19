#!/bin/bash
echo "Caching examples in $CACHE_LOCATION"
python main.py cache-examples data/datasets/newsqa/train.json \
  --model-path "$MODEL" --num-workers 32 --debug-features \
  --out-folder "$CACHE_LOCATION" --dataset-only
echo "Running distributed training...."
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
  --debug train "$CACHE_LOCATION/train-$MODEL.bin" \
  --model-path "$MODEL" --model-type "$MODEL_TYPE" --eval-file data/datasets/newsqa/dev.json \
  --save-model-folder "$SAVE_TO" --do-eval-after-training \
  --per-gpu-train-batch-size "$BATCH_SIZE" --max-answer-length 30 --gradient-accumulation-steps "$ACC_STEPS" \
  --fp16 --debug-features --save-steps 0 --num-train-epochs 3 --num-workers 8
#echo "Removing $CACHE_LOCATION"
#rm -fr "$CACHE_LOCATION"
echo "Done!"