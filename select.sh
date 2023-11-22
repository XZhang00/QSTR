# GPU卡的数量
num_array=(0 1 2 3 4 5 6 7)

# QQPPos-select：
for num in ${num_array[@]};
do
    nohup python -u top_select.py \
        --model_name=../outputs-QQPPos-fast-siscp-roberta/sort_batch+mse-lr_3e-5/epoch19 \
        --output_path=../retrieve-results-siscp-qqppos \
        --syn_vocab_path=../data-QQPPos/syn2id-roberta.json \
        --batch_size=3000 \
        --device_id=$num  >> log/select.log$num 2>&1 &
done


# ParaNMT-select:
# for num in ${num_array[@]};
# do
#     nohup python -u top_select.py \
#         --model_name=../outputs-ParaNMT-fast-ref015/ref015-SimFT_ffn-sort_batch+mse-lr_2e-5/epoch9 \
#         --output_path=../retrieve-results-fast-ref015-ParaNMT \
#         --per_device_num=160 \
#         --src_path=../data-ParaNMT/test/src.txt \
#         --input_aesop_src=../data-ParaNMT/test/aesop-level5-src.source \
#         --input_aesop_ref=../data-ParaNMT/test/aesop-level5-ref.source \
#         --all_parse_path=../data-ParaNMT/candidate_trees_h5_5W.json \
#         --syn_vocab_path=../data-ParaNMT/syns2id.json \
#         --batch_size=3000 \
#         --device_id=$num  >> log/select.log$num 2>&1 &
# done



# Few-shot QQP
# for num in ${num_array[@]};
# do
#     nohup python -u top_select.py \
#         --model_name=../outputs-QQPPos-fast-ref015-roberta/sort_batch+mse-lr_3e-5/epoch19 \
#         --output_path=../Few_shot-per5-shot/QQP-parses \
#         --per_device_num=50 \
#         --src_path=../Few_shot-per5-shot/QQP.txt \
#         --input_aesop_src=../Few_shot-per5-shot/QQP-aesop.source \
#         --input_aesop_ref=../Few_shot-per5-shot/QQP-aesop.source \
#         --all_parse_path=../data-QQPPos/candidate_trees_h5.json \
#         --syn_vocab_path=../data-QQPPos/syn2id-roberta.json \
#         --batch_size=3000 \
#         --device_id=$num  >> log/select.log$num 2>&1 &
# done


# Few-shot MRC
# for num in ${num_array[@]};
# do
#     nohup python -u top_select.py \
#         --model_name=../outputs-ParaNMT-fast-ref015/ref015-SimFT_ffn-sort_batch+mse-lr_2e-5/epoch10 \
#         --output_path=../Few_shot-per5-shot/MRC-parses \
#         --per_device_num=50 \
#         --src_path=../Few_shot-per5-shot/MRC.txt \
#         --input_aesop_src=../Few_shot-per5-shot/MRC-aesop.source \
#         --input_aesop_ref=../Few_shot-per5-shot/MRC-aesop.source \
#         --all_parse_path=../data-ParaNMT/candidate_trees_h5_5W.json \
#         --syn_vocab_path=../data-ParaNMT/syns2id.json \
#         --batch_size=3000 \
#         --device_id=$num  >> log/select.log$num 2>&1 &
# done
