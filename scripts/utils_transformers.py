import os
import random
from dataclasses import dataclass

import click
import numpy as np
import torch
from loguru import logger
from transformers import AutoTokenizer, AutoConfig, AutoModelForQuestionAnswering, SquadV2Processor, SquadV1Processor, \
    squad_convert_examples_to_features, MODEL_FOR_QUESTION_ANSWERING_MAPPING


def set_seed(args: 'Args'):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def _is_gpu_available():
    import torch
    try:
        gpu = torch.cuda.is_available()
        if gpu:
            str(torch.rand(1).to(torch.device("cuda")))
    except:
        gpu = False
    return gpu


def get_tokenizer(name_or_model_name_or_path, do_lower_case):
    return AutoTokenizer.from_pretrained(
        # args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        name_or_model_name_or_path,
        # do_lower_case=args.do_lower_case,
        do_lower_case=do_lower_case,
        # cache_dir=args.cache_dir if args.cache_dir else None,
        cache_dir=None
    )


def get_model(model_name_or_path):
    config = AutoConfig.from_pretrained(
        # args.config_name if args.config_name else args.model_name_or_path,
        model_name_or_path,
        # cache_dir=args.cache_dir if args.cache_dir else None,
        cache_dir=None,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        # args.model_name_or_path,
        model_name_or_path,
        # from_tf=bool(".ckpt" in args.model_name_or_path),
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
        # cache_dir=args.cache_dir if args.cache_dir else None,
        cache_dir=None,
    )
    return model


@click.command()
@click.argument("in-files", nargs=-1)
@click.option("--out-folder")
@click.option('--model-path', type=str)
@click.option('--do-not-lower-case', is_flag=True, default=False)
@click.option('--evaluate', is_flag=True, default=False)
@click.option('--v2', is_flag=True, default=False)
@click.option('--max-seq-length', type=int, default=384)
@click.option('--doc-stride', type=int, default=128)
@click.option('--max-query-length', type=int, default=64)
@click.option('--num-workers', type=int, default=1)
@click.option('--debug-features', type=bool, is_flag=True, default=False)
@click.option('--dataset-only', type=bool, is_flag=True, default=False)
def cache_examples(in_files, out_folder, model_path, do_not_lower_case, evaluate, v2, max_seq_length, doc_stride,
                   max_query_length, num_workers, debug_features, dataset_only):
    do_lower_case = not do_not_lower_case
    tokenizer = get_tokenizer(model_path, do_lower_case)

    for in_file in in_files:
        file_name = os.path.basename(in_file)
        dataset, examples, features = convert_to_features(in_file, evaluate, doc_stride,
                                                          max_query_length,
                                                          max_seq_length, num_workers,
                                                          tokenizer, v2, debug_features)
        out_file = os.path.join(out_folder, f"{os.path.splitext(file_name)[0]}-"
                                            f"{os.path.basename(os.path.normpath(model_path))}.bin")
        click.echo(f"Saving features into cached file {click.style(out_file, fg='blue')}")
        if os.path.dirname(out_file).replace(".", ""):
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
        if not dataset_only:
            torch.save({"features": features, "dataset": dataset, "examples": examples}, out_file)
        else:
            torch.save({"dataset": dataset}, out_file)

def convert_to_features(in_file, evaluate, doc_stride, max_query_length, max_seq_length,
                        num_workers, tokenizer, debug_features=False, v2=False):
    processor = SquadV2Processor() if v2 else SquadV1Processor()
    data_dir = os.path.dirname(in_file)
    file_name = os.path.basename(in_file)
    if doc_stride >= max_seq_length - max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )
    if evaluate:
        examples = processor.get_dev_examples(data_dir, filename=file_name)
    else:
        examples = processor.get_train_examples(data_dir, filename=file_name)
    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=not evaluate,
        return_dataset="pt",
        threads=num_workers,
    )
    if debug_features:
        debug_features_examples_dataset(dataset, examples, features, tokenizer)
    return dataset, examples, features


def debug_features_examples_dataset(dataset, examples, features, tokenizer):
    n = random.randint(0, len(features))
    logger.info(f"passage {n} as decoded by tokenizer: ")
    logger.info(" ".join(f"[{tokenizer.decode(e.item())}]" for e in dataset[n][0] if e))
    if len(dataset.tensors) == 6:
        logger.info("This is a dataset for evaluation!")
    elif len(dataset.tensors) == 8:
        logger.info("This is a dataset for training!")
        max_answer_length = max(dataset[i][4].item() - dataset[i][3].item() for i in range(len(dataset)))
        start, end = dataset[n][3].item(), dataset[n][4].item() + 1
        logger.info(" ".join(f"[{tokenizer.decode(e.item())}]" for e in dataset[n][0][start:end] if e))
        logger.info(f"Max answer length: {max_answer_length}")


def load_examples(location, dataset_only=False):
    features_and_dataset = torch.load(location)
    if not dataset_only:
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
        return dataset, examples, features
    else:
        return features_and_dataset['dataset']


MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class Args:
    model_path: str
    model_type: str
    eval_file: str = ""
    predictions_folder: str = None
    no_cuda: bool = None

    per_gpu_eval_batch_size: int = 8
    lang_id: int = 0
    v2: bool = False
    n_best_size: int = 5
    max_answer_length: int = 30
    verbose_logging: bool = False
    null_score_diff_threshold: float = 0.0
    train_file: str = None
    overwrite_output_dir: bool = False
    save_steps: int = 500
    evaluate_during_training: bool = False
    logging_steps: int = 0
    max_grad_norm: float = 1.0
    save_model_folder: str = ''
    seed: int = None
    fp16_opt_level: str = "O1"
    weight_decay: float = 0.0
    fp16: bool = False
    warmup_steps: int = 0
    do_eval_after_training: bool = False
    eval_all_checkpoints: bool = False
    adam_epsilon: float = 1e-8
    learning_rate: float = 5e-5
    gradient_accumulation_steps: int = 1
    max_steps: int = -1
    per_gpu_train_batch_size: int = 8
    num_train_epochs: int = 3
    n_gpu: int = None
    local_rank: int = -1
    device: str = 'cpu'
    doc_stride: int = 128
    max_query_length: int = 64
    max_seq_length: int = 384
    num_workers: int = 8
    debug_features: bool = False
    do_not_lower_case: bool = False

    @property
    def do_lower_case(self):
        return not self.do_not_lower_case


def load_or_convert(args: Args, tokenizer, evaluate=False, dataset_only=False):
    attr = "eval_file" if evaluate else "train_file"
    file = getattr(args, attr)
    if dataset_only and evaluate:
        raise ValueError("You don't want that!")
    if file.endswith('bin'):
        click.echo("Loading train features from cache...")
        if dataset_only:
            return load_examples(file, dataset_only=True)
        else:
            train_dataset, e, f = load_examples(file)
    elif file.endswith('json'):
        click.echo("Converting features on the fly...")
        train_dataset, e, f = convert_to_features(file, evaluate=evaluate, tokenizer=tokenizer, v2=args.v2,
                                                  doc_stride=args.doc_stride, max_query_length=args.max_query_length,
                                                  max_seq_length=args.max_seq_length,
                                                  num_workers=args.num_workers, debug_features=args.debug_features)
        if dataset_only:
            return train_dataset
    else:
        raise NotImplementedError(f"Unknown file extension for evaluation file: {file}")
    return train_dataset, e, f
