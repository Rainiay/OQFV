CUDA_VISIBLE_DEVICES=0 python run_vit.py  \
  --model_name_or_path model/vit_base_patch16_224_in21k \
  --seed 0 \
  --dataset caltech101 \
  --train_batch_size 64 \
  --eval_batch_size 256 \
  --sample_batch_size 128 \
  --sample_method distribution \
  --learning_rate 1e-3 \
  --weight_decay 1e-4 \
  --num_train_epochs 20 \
  --reduced_rank 32 \
  --int_bit 2 \
  --oqfv \
  --quant_method uniform \

