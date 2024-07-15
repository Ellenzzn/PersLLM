'''
We refer to the code of UltraChat project:
@article{ding2023enhancing,
  title={Enhancing Chat Language Models by Scaling High-quality Instructional Conversations},
  author={Ding, Ning and Chen, Yulin and Xu, Bokai and Qin, Yujia and Zheng, Zhi and Hu, Shengding and Liu, Zhiyuan and Sun, Maosong and Zhou, Bowen},
  journal={arXiv preprint arXiv:2305.14233},
  year={2023}
}
'''


# GPUS_PER_NODE=4
# pip install torch==1.11.0
#pip install transformers==4.31.0
#pip install bmtrain==0.2.2
#pip install jieba
#pip install datasets
# pip install model-center
# cd ../lm-evaluation-harness
# pip install -e .
#pip install protobuf==3.20.0
#pip install sentencepiece
#pip install einops
#pip install accelerate==0.20.3
#pip list

echo "current path"
pwd
MASTER_ADDR=localhost
MASTER_PORT=12349
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=4
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

OPTS=""
OPTS+=" --gradient_accumulation_steps 4"
OPTS+=" --logging_step 10" 
OPTS+=" --batch_size_per_device 8"
OPTS+=" --save_step 500"
OPTS+=" --epochs 1"
OPTS+=" --lr 5e-5"
OPTS+=" --max_seq_length 3000"
OPTS+=" --weight-decay 0.1"
OPTS+=" --train-iters -1"
OPTS+=" --warmup_iters 100"
OPTS+=" --start-step 0"
OPTS+=" --model_name_or_path checkpoint/miniCPM-mc-format"
OPTS+=" --model persLLM_harry_mix"
OPTS+=" --save_dir save_checkpoint"
OPTS+=" --data_dir Harry_Potter_data"
OPTS+=" --ultra_split processed_harry_mix.jsonl"
OPTS+=" --save_limit 3"
OPTS+=" --loss-scale 32768000"
echo "export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512"
CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} src/train_bm_modelcenter.py ${OPTS}"

echo "-------final CMD is------"
echo "${CMD}"
echo "-------final CMD end------"
$CMD

