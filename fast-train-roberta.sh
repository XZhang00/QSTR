export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=4,5,6,7
NUM_GPUS=8
lr=3e-5
select_num=10

output_path=../outputs-QQPPos-fast-AESOP-1W-roberta/sort_batch+mse-lr_$lr
if [[ ! -e ${output_path} ]]; then
    mkdir -p ${output_path}
fi


nohup python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --nnodes=1 --node_rank=0 --master_port 66$(($RANDOM%90+10))  \
    fast-train-roberta.py \
    --n_gpu=$NUM_GPUS \
    --model_name=../roberta-base \
    --bert_path=../roberta-base \
    --skip_epochs=0 \
    --src_path=../data-QQPPos/train/src.txt \
    --src_random_parse_path=../data-QQPPos/train/save4train-1W/src_random.tree \
    --scores_path=../data-QQPPos/train/save4train-1W/src_random.score-ref015 \
    --test_path=../data-QQPPos/test/src.txt \
    --test_parse_path=../data-QQPPos/test/src.tree_h5 \
    --test_random_trees_path=../data-QQPPos/test/save4test-1W/src_random.tree \
    --test_random_scores_path=../data-QQPPos/test/save4test-1W/src_score.ref015 \
    --val_path=../data-QQPPos/val/src.txt \
    --val_parse_path=../data-QQPPos/val/src.tree_h5 \
    --val_random_trees_path=../data-QQPPos/val/save4val-1W/src_random.tree \
    --val_random_scores_path=../data-QQPPos/val/save4val-1W/src_score.ref015 \
    --syn_vocab_path=../data-QQPPos/syn2id-roberta.json \
    --select_num=$select_num \
    --saved_model_path=$output_path \
    --warm_up_steps=2000 \
    --lr=$lr \
    --weight_decay=0.01 \
    --gradient_accumulation_steps=1 \
    --max_sen_len=64 \
    --max_syn_len=192 \
    --n_epochs=20 \
    --batch_size=32 \
    --eval_batch_size=200 \
    --log_interval=100 \
    --save_interval=2000 >> ${output_path}/log-lr_$lr.txt 2>&1 &



