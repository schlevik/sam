import glob
import os
import sys

import click
import torch
from torch.utils.data import RandomSampler, DistributedSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, WEIGHTS_NAME, AdamW, \
    get_linear_schedule_with_warmup

from scripts.predict_transformers import evaluate
from scripts.utils import write_json
from scripts.utils_transformers import set_seed, get_tokenizer, get_model, load_examples, Args, load_or_convert
from loguru import logger


@click.command()
@click.argument("train-file", type=str)
@click.option("--model-path", type=str)
@click.option("--model-type", type=str)
@click.option("--save-model-folder", type=str)
@click.option("--eval-file", type=str, default=None)
@click.option("--predictions-folder", type=str, default=None)
@click.option("--no-cuda", is_flag=True, type=bool, default=None)
@click.option('--do-not-lower-case', is_flag=True, default=False)
@click.option("--per-gpu-eval-batch-size", type=int, default=8)
@click.option("--lang-id", type=int, default=0)
@click.option("--v2", type=bool, is_flag=True, default=False)
@click.option("--overwrite-output-dir", is_flag=True, type=bool, default=False)
@click.option("--save-steps", type=int, default=500)
@click.option("--evaluate-during-training", is_flag=True, type=bool, default=False)
@click.option("--logging-steps", type=int, default=500)
@click.option("--max-grad-norm", type=float, default=1.0)
@click.option("--n-best-size", type=int, default=5)
@click.option("--max-answer-length", type=int, default=10)
@click.option("--verbose-logging", is_flag=True, type=bool, default=False)
@click.option("--null-score-diff-threshold", type=float, default=0.0)
@click.option("--seed", type=int, default=42)
@click.option("--fp16-opt-level", type=str, default="O1")
@click.option("--weight-decay", type=float, default=0.0)
@click.option("--fp16", type=bool, is_flag=True, default=False)
@click.option("--warmup-steps", type=int, default=0)
@click.option("--do-eval-after-training", is_flag=True, type=bool, default=False)
@click.option("--eval-all-checkpoints", is_flag=True, type=bool, default=False)
@click.option("--adam-epsilon", type=float, default=1e-8)
@click.option("--learning-rate", type=float, default=5e-5)
@click.option("--gradient-accumulation-steps", type=int, default=1)
@click.option("--max-steps", type=int, default=-1)
@click.option("--per-gpu-train-batch-size", type=int, default=8)
@click.option("--num-train-epochs", type=int, default=3)
@click.option('--max-seq-length', type=int, default=384)
@click.option('--doc-stride', type=int, default=128)
@click.option('--max-query-length', type=int, default=64)
@click.option('--num-workers', type=int, default=1)
@click.option('--debug-features', type=bool, is_flag=True, default=False)
# @click.option("--local-rank", type=int, default=-1)
# @click.option("--device", default='cpu')
def train(**kwargs):
    # doc_stride = kwargs.pop("doc_stride")
    # max_query_length = kwargs.pop('max_query_length')
    # max_seq_length = kwargs.pop("max_seq_length")
    # num_workers = kwargs.pop('num_workers')
    # debug_features = kwargs.pop('debug_features')
    # do_lower_case = not kwargs.pop('do_not_lower_case')
    # kwargs['logging_steps'] = [int(i) for i in kwargs['logging_steps'].split(',')] if kwargs['logging_steps'] else []
    args = Args(**kwargs)
    logger.debug(args)
    if (
            os.path.exists(args.save_model_folder)
            and os.listdir(args.save_model_folder)
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.save_model_folder
            )
        )
    # os.makedirs(args.predictions_folder, exist_ok=True)
    os.makedirs(args.save_model_folder, exist_ok=True)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    if not args.local_rank in [-1, 0]:
        logger.remove()
        logger.add(sys.stdout, level="WARNING")
    logger.warning(
        f"Process rank: {args.local_rank}, device: {device}, n_gpu: "
        f"{args.n_gpu}, distributed training: "
        f"{bool(args.local_rank != -1)}, 16-bits training: {args.fp16}",

    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()

    tokenizer = get_tokenizer(args.model_path, args.do_lower_case)
    model = get_model(args.model_path)

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    # logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum
    # if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations.
    # Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    train_dataset, *_ = load_or_convert(args, tokenizer)
    # train_dataset, e, f = load_examples(args.train_file)
    logger.info("loaded dataset")
    global_step, tr_loss = do_train(args, train_dataset, model, tokenizer)
    logger.info(f"global_step = {global_step}, average loss = {tr_loss}")

    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        logger.info(f"Saving model checkpoint to {args.save_model_folder}")
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.save_model_folder)
        tokenizer.save_pretrained(args.save_model_folder)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.save_model_folder, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = AutoModelForQuestionAnswering.from_pretrained(args.save_model_folder)  # , force_download=True)
        tokenizer = AutoTokenizer.from_pretrained(args.save_model_folder, do_lower_case=args.do_lower_case)
        model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval_after_training and args.local_rank in [-1, 0]:
        logger.info("Loading checkpoints saved during training for evaluation")
        checkpoints = [args.save_model_folder]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c)
                for c in sorted(glob.glob(args.save_model_folder + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            # logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs

        logger.info(f"Evaluate the following checkpoints: {checkpoints}")
        dataset, examples, features = load_or_convert(args, tokenizer, evaluate=True)
        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)  # , force_download=True)
            model.to(args.device)

            # Evaluate

            result = evaluate(args, model, tokenizer, dataset, examples, features, suffix=global_step)

            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)

    logger.info("Results: {}".format(results))
    write_json(results, os.path.join(args.save_model_folder, 'dev-results.json'))
    return results


def do_train(args: Args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_path, "scheduler.pt")))

    if args.fp16:
        try:
            # noinspection PyPackageRequirements
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        # noinspection PyArgumentList
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"Num examples = {len(train_dataset)}")
    logger.info(f"Num Epochs = {args.num_train_epochs}")
    logger.info(f"Instantaneous batch size per GPU = {args.per_gpu_train_batch_size}")
    # noinspection PyUnresolvedReferences
    logger.info(
        "Total train batch size (w. parallel, distributed & accumulation) = {}".format(
            args.train_batch_size * args.gradient_accumulation_steps * (
                torch.distributed.get_world_size() if args.local_rank != -1 else 1)),
    )
    logger.info(f"Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"Total optimization steps = {t_total}")

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_path):
        try:
            # set global_step to global_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info(f"Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"Continuing training from epoch {epochs_trained}")
            logger.info(f"Continuing training from global step {global_step}")
            logger.info(f"Will skip the first {steps_trained_in_current_epoch} steps in the first epoch")
        except ValueError:
            logger.info("Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(args)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                del inputs["token_type_ids"]

            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
                if args.v2:
                    inputs.update({"is_impossible": batch[7]})
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                    )

            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps and global_step % args.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    # TODO: fix evaluate_during_training with on-the-fly created features
                    if args.local_rank == -1 and args.evaluate_during_training and False:
                        dataset, examples, features = load_examples(args.eval_file)
                        results = evaluate(args, model, tokenizer, dataset, examples, features, suffix=str(global_step))
                        for key, value in results.items():
                            # noinspection PyUnboundLocalVariable
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                # Save model checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.save_model_folder, "checkpoint-{}".format(global_step))
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info(f"Saving model checkpoint to  {output_dir}")

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info(f"Saving optimizer and scheduler states to {output_dir}")

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step
