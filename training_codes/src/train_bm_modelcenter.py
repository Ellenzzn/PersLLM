
import argparse
import torch
from ultrachat_dataset import load_raw_data, PromptIterableDataset, collator
from transformers.optimization import get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.utils.data import DataLoader
import bmtrain as bmt
from functools import partial
import time
import os
from accelerate import load_checkpoint_and_dispatch
from accelerate import init_empty_weights
import sys
sys.path.append("./ModelCenter")
from model_center.model import Llama
from model_center.tokenizer import LlamaTokenizer
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import logging
import shutil
import numpy as np
import math
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_model_tokenizer(args):
    logger.info("loading model...")
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    special_dict = {'additional_special_tokens': ['<TIME-I>', '<TIME-II>', '<TIME-III>']}
    tokenizer.add_special_tokens(special_dict)

    model = Llama.from_pretrained(args.model_name_or_path)
    logger.info("loaded")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.add_special_tokens({'pad_token': "<pad>"})
    #model.resize_token_embeddings(len(tokenizer))
    if args.load_ckpt is not None:
        logger.info(f"loading model from {args.load_ckpt}")
        bmt.load(model, os.path.join(args.load_ckpt, "pytorch_model.pt"))
    return model, tokenizer

def get_optimizer(args, model):
    optimizer = bmt.optim.AdamOptimizer(
        model.parameters(), weight_decay=args.weight_decay, eps=1e-5, betas=(0.9, 0.95)
    )
    if args.load_ckpt is not None:
        file_name = os.path.join(args.load_ckpt, "optim.rank-{}.opt".format(bmt.rank()))
        logger.info(file_name)
        if os.path.exists(file_name):
            logger.info("start to load grad ckpt {}".format(file_name))
            states = torch.load(file_name)
            optimizer.load_state_dict(states)
    return optimizer


def get_learning_rate_scheduler(args, optimizer):
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters
    if args.lr_decay_style == "linear":
        lr_scheduler = bmt.lr_scheduler.Linear(
            optimizer,
            start_lr=args.lr,
            warmup_iter=int(args.warmup_iters),
            end_iter=args.lr_decay_iters,
            num_iter=args.start_step,
        )
    elif args.lr_decay_style == "cosine":
        bmt.print_rank("use cosine")
        class Cosine(bmt.lr_scheduler.WarmupLRScheduler):
            def get_lr_warmup(self, num_iter) -> float:
                return self.start_lr * num_iter / self.warmup_iter
                
            def get_lr_decay(self, num_iter) -> float:
                progress = (num_iter - self.warmup_iter) / max(1, (self.end_iter - self.warmup_iter))
                return max(self.start_lr * 0.1, self.start_lr * (0.1 + 0.45 * (1.0 + math.cos(progress * math.pi))))

        lr_scheduler = Cosine(
            optimizer,
            start_lr=args.lr,
            warmup_iter=int(args.warmup_iters),
            end_iter=args.lr_decay_iters,
            num_iter=args.start_step,
        )

    elif args.lr_decay_style == "noam":
        logger.info("use noam")
        lr_scheduler = bmt.lr_scheduler.Noam(
            optimizer,
            start_lr=args.lr,
            warmup_iter=int(args.warmup_iters),
            end_iter=args.lr_decay_iters,
            num_iter=args.start_step,
        )
    else:
        raise NotImplementedError
    return lr_scheduler


def setup_model_and_optimizer(args):
    model, tokenizer = get_model_tokenizer(args)
    bmt.synchronize()
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    bmt.synchronize()
    return tokenizer, model, optimizer, lr_scheduler

def save(model, model_fname):
    if os.path.exists(model_fname): os.remove(model_fname)
    
    temp_fname = os.path.join("/local", model_fname.strip("/"))
    if os.path.exists(temp_fname): os.remove(temp_fname)
    temp_dir = os.path.dirname(temp_fname)
    os.makedirs(temp_dir, exist_ok=True)
    logger.info(f"temp_fname: {temp_fname}")

    time1 = time.time()
    bmt.save(model, temp_fname)
    time2 = time.time()
    logger.info(f"bmt.save used {(time2 - time1)} sec")

    if bmt.rank() == 0:
        logger.info(f"src:{temp_fname} -> dst:{model_fname}")
        shutil.move(temp_fname, model_fname)
        time3 = time.time()
        logger.info(f"mv used {(time3 - time2)} sec")
        
        dir_list = os.listdir(os.path.dirname(model_fname))
        logger.info(f"{os.path.dirname(model_fname)}: {','.join(dir_list)}")
        
def train(args):

    bmt.init_distributed(
        seed=args.seed,
        zero_level=3,
    )

    if args.wandb and bmt.rank() == 0:
        wandb.init()
    
    if args.tensorboard is not None and bmt.rank() == 0:
        from torch.utils.tensorboard import SummaryWriter
        import distutils.version  # noqa: F401

        if not os.path.exists(args.tensorboard):
            os.makedirs(args.tensorboard)
        writer = SummaryWriter(log_dir=args.tensorboard)

    
    
    original_dataset = []
    logger.info(f"load ultrachat data, split={args.ultra_split}")
    original_dataset += load_raw_data(args.data_dir, max_sample=args.max_sample, random_state=0, split=args.ultra_split)
    args.train_iters = args.epochs * (len(original_dataset) // (bmt.world_size() * args.batch_size_per_device) + 1 )

    tokenizer, model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    optim_manager = bmt.optim.OptimManager(loss_scale=args.loss_scale)
    optim_manager.add_optimizer(optimizer, lr_scheduler)
    bmt.synchronize()
    

    bmt.print_rank("Model memory")
    bmt.print_rank(torch.cuda.memory_summary())

    avg_time_recorder = bmt.utils.AverageRecorder()
    avg_loss_recorder = bmt.utils.AverageRecorder()
    train_start_time = time.time()
    global_step = 0

    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)

    logger.info("entering trainig loop...")
    for epoch in range(args.epochs):
        logger.info("permuting data...")
        indices = torch.randperm(len(original_dataset))
        dataset = [original_dataset[i] for i in indices]

        logger.info("split data for each process")
        data_per_gpu = len(dataset) // bmt.world_size()
        dataset = dataset[bmt.rank() * data_per_gpu : (bmt.rank() + 1) * data_per_gpu]

        logger.info("wrapping up data")
        dataset = PromptIterableDataset(dataset, tokenizer = tokenizer, max_seq_length = args.max_seq_length, teacher_forcing=True, truncate_method="tail")
        dataloader = DataLoader(dataset, batch_size=args.batch_size_per_device, collate_fn=partial(collator, tokenizer))

        if global_step >= args.train_iters:
            break
        progress_bar = tqdm(range(len(dataloader)), disable=not bmt.rank()==0, desc=f"epoch {epoch}")
        logger.info(f"*******start {epoch} epoch training********")
        for step, inputs in enumerate(dataloader):
            if global_step < args.start_step:
                global_step += 1
                progress_bar.update(1)
                continue
            st = time.time()

            with bmt.inspect.inspect_tensor() as inspector:
                for k in inputs:
                    inputs[k] = inputs[k].cuda()
                labels = inputs.pop("labels")
                logits = model(**inputs).logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                shift_logits = shift_logits.view(-1, len(tokenizer))
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                # print("logits:", shift_logits[:5, :10])
                # print("labels:", shift_labels[:10])
                loss = loss_func(shift_logits, shift_labels)
                global_loss = bmt.sum_loss(loss).item()

                optim_manager.backward(loss)


                if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(dataloader) - 1:
                    optim_manager.clip_grad_norm(optimizer.param_groups, max_norm=args.clip_grad)

                    optim_manager.step()
                    optim_manager.zero_grad()

            
            global_step += 1
            progress_bar.update(1)

            # record time and loss
            iteration_time = time.time() - st

            avg_time_recorder.record(iteration_time)
            if not np.isnan(global_loss):
                avg_loss_recorder.record(global_loss)

            # print time and loss
            if global_step % args.logging_step == 0:
                bmt.print_rank(
                    "| Iter: {:6d} | loss: {:.4f} average_loss: {:.4f} | lr: {:.4e} | time: {:.4f} seconds | total_time_passed: {:.4f} minutes".format(
                        global_step,
                        global_loss,
                        avg_loss_recorder.value,
                        lr_scheduler.current_lr,
                        avg_time_recorder.value,
                        (time.time() - train_start_time) / 60
                    )
                )
                if args.wandb and bmt.rank() == 0:
                    wandb.log({
                        "loss": global_loss,
                        "average_loss": avg_loss_recorder.value,
                        "lr": lr_scheduler.current_lr,
                    }, step=global_step)
                if args.tensorboard and bmt.rank() == 0:
                    writer.add_scalar("Loss/train", global_loss, global_step)
                    writer.add_scalar("average_Loss/train", avg_loss_recorder.value, global_step)
                    writer.add_scalar("lr/train", lr_scheduler.current_lr, global_step)


            # save model
            if global_step % args.save_step == 0:
                try_time = 0
                while try_time < 10:
                    try:
                        save_dir = os.path.join(args.save_dir, f"{args.model}/step_{global_step}")
                        os.makedirs(save_dir, exist_ok=True)

                        bmt.save(model, os.path.join(save_dir, "pytorch_model.pt"))
                        # save(model, os.path.join(save_dir, "pytorch_model.pt"))
                        print("saving optimizer state", str(os.path.join(save_dir, "optim.rank-%d.opt" % bmt.rank())))
                        torch.save(optimizer.state_dict(),
                                os.path.join(save_dir, "optim.rank-%d.opt" % bmt.rank()))

                        if bmt.rank() == 0:
                            # torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
                            

                            torch.save(lr_scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))
                            tokenizer.save_pretrained(save_dir)
                        bmt.print_rank(f"model saved at {save_dir}")

                        
                    except:
                        try_time += 1
                        continue
                    else:
                        break
                        
                if bmt.rank() == 0:
                    if args.save_limit is not None:
                        output_dir = os.path.join(args.save_dir, args.model)
                        files = os.listdir(output_dir)
                        ckpt_id = list(sorted([int(f[5:]) for f in files if f.startswith("step_") and "_hf" not in f], reverse=True))
                        for i in ckpt_id[args.save_limit:]:
                            path = os.path.join(output_dir, f"step_{i}")
                            if not os.path.exists(os.path.join(output_dir, f"step_{i}_hf")):
                                shutil.rmtree(path)
            
            if global_step == args.train_iters:
                break
    
    # bmt.save(model, os.path.join(args.save_dir, f"ultrachat_{args.model}/final.pt"))

  

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--model", type=str, default='llama-13b')
    parser.add_argument("--model_name_or_path", default='/mnt/data/user/tc_agi/user/chenyulin/llama/llama-7b')
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--max_seq_length", default=2048, type=int)
    parser.add_argument("--batch_size_per_device", default=2, type=int)
    parser.add_argument("--logging_step", default=100, type=int)
    parser.add_argument("--save_step", default=50000, type=int)
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--ultra_split", default=None, type=str)


    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--with_eval", action="store_true")

    parser.add_argument("--clip-grad", type=float, default=1.0, help="gradient clipping")
    # Learning rate.
    parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay rate")
    parser.add_argument("--loss-scale", type=float, default=6553600, help="loss scale")

    parser.add_argument("--train-iters", type=int, default=2000000)

    parser.add_argument("--save_dir", type=str, default="/data/models/chenyulin/ultrachat-llama")

    parser.add_argument("--max_sample", type=int, default=None, help="max training sample num for ultrachat")
    parser.add_argument("--save_limit", type=int, default=None, help="ckpt saved limit number")

    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--lr-decay-style",
        type=str,
        default="cosine",
        choices=["constant", "linear", "cosine", "exponential", "noam"],
        help="learning rate decay function",
    )
    parser.add_argument("--lr-decay-iters", type=int, default=None, help="lr decay steps")
    parser.add_argument(
        "--start-step", type=int, default=0, help="step to start or continue training"
    )
    parser.add_argument("--tensorboard", type=str, default=None, help="lr decay steps")
    parser.add_argument("--load_ckpt", type=str, default=None, help="resumed ckpt")
    parser.add_argument("--local_rank", type=str,default=None)

    args = parser.parse_args()

    train(args)
    args.model = args.model_name_or_path.split("/")[-1]
    logger.info(args.model)
