import os

import click

from scripts.utils import get_output_predictions_file_name


def get_predictions_folder(sam_path, dataset_name):
    return f"data/football/{sam_path}/predictions/{dataset_name}"


@click.command()
@click.argument('command', type=str)
@click.option('--sam', default='RB')
@click.option('--dataset-name', default='squad1')
#@click.option('--model-name', default='bert-base-uncased')
#@click.option('--model-type', default='bert')
@click.option("--train-file", default="train-v1.1.json")
@click.option("--eval-file", default="dev-v1.1.json")
@click.option("--batch-size", default=None)
@click.option("--extra-args", default="--fp16 --save-steps 0 --debug-features")
@click.option("--notify", type=str, default='viktor.schlegel@manchester.ac.uk')
@click.option("model_names", '--model-name', type=str, multiple=True)
@click.option("model_types", '--model-type', type=str, multiple=True)
def generate_dvc(command, sam, dataset_name,
                 #model_name, model_type,
                 train_file, eval_file, batch_size, extra_args,
                 notify, model_names, model_types):
    model_name = model_names[0]
    model_type = model_types[0]
    sam_path = sam.lower()
    sam_dataset_path = f"data/football/{sam_path}"
    baseline_file = f"{sam_dataset_path}/baseline-{sam_path}.json"
    intervention_file = f"{sam_dataset_path}/intervention-{sam_path}.json"
    control_file = f"{sam_dataset_path}/control-{sam_path}.json"
    predictions_folder = get_predictions_folder(sam_path, dataset_name)
    model_folder = f"{model_name}-{dataset_name}"
    model_path = f"models/{model_folder}"
    train_path = f"data/datasets/{dataset_name}/{train_file}"
    eval_path = f"data/datasets/{dataset_name}/{eval_file}"
    baseline_predictions = get_output_predictions_file_name(baseline_file, predictions_folder, model_folder)
    intervention_predictions = get_output_predictions_file_name(intervention_file, predictions_folder, model_folder)
    control_predictions = get_output_predictions_file_name(control_file, predictions_folder, model_folder)
    if not batch_size:
        batch_size = 8 if "large" in model_name else 24

    if command == 'train-transformers':
        cmd = (
            f"python main.py --debug --notify {notify} train {train_path} --model-path {model_name} "
            f"--model-type {model_type} --eval-file {eval_path} --save-model-folder {model_path} "
            f"--do-eval-after-training --num-workers 8 --per-gpu-train-batch-size {batch_size} "
            f"--max-answer-length 30 {extra_args}"
        )
        stage_name = f"train-{model_name}-on-{dataset_name}"
        dvc_cmd = (
            f"dvc run -n {stage_name} -d {train_path} -d {eval_path} -o {model_path} "
            f"{cmd}"
        )
    elif command == 'predict-transformers':
        models_str = " ".join(f"--model-path models/{mp}-{dataset_name} --model-type {mt}" for mp, mt in
                              zip(model_names, model_types))
        cmd = (f"python main.py --debug --notify {notify} "
               f"predictions {baseline_file} {intervention_file} {control_file} "
               f" {models_str} "
               f"--out-folder {predictions_folder} --max-answer-length 10 {extra_args}")

        stage_name = f"predict-transformers-{dataset_name}-on-{sam_path}"
        deps_str = " ".join(f"-d models/{mp}-{dataset_name}" for mp in model_names)
        outs_str = ' '.join(
            f"-o {get_output_predictions_file_name(baseline_file, predictions_folder, f'{mp}-{dataset_name}')} "
            f"-o {get_output_predictions_file_name(intervention_file, predictions_folder, f'{mp}-{dataset_name}')} "
            f"-o {get_output_predictions_file_name(control_file, predictions_folder, f'{mp}-{dataset_name}')}"
            for mp in model_names
        )
        dvc_cmd = (f"dvc run -n {stage_name} {deps_str} -d {model_path} -d {baseline_file} "
                   f"-d {intervention_file} -d {control_file}  {outs_str} {cmd}")
    # elif command == 'predict-transformers-filter':
    #     predictions_folder = get_predictions_folder(sam_path, 'filter')
    #     assert len(model_names) == len(model_types)
    #     models_str = " ".join(f"--model-path models/{mp}-{dataset_name} --model-type {mt}" for mp, mt in
    #                           zip(model_names, model_types))
    #     cmd = (f"python main.py --debug --notify {notify} "
    #            f"predictions {filter_file} "
    #            f"{models_str} "
    #            f"--out-folder {predictions_folder} --max-answer-length 10 {extra_args}")

        # stage_name = f"filter-transformers-{dataset_name}"
        # deps_str = " ".join(f"-d models/{mp}-{dataset_name}" for mp in model_names)
        # outs_str = ' '.join(
        #     f"-o {get_output_predictions_file_name(filter_file, predictions_folder, mp)}" for mp in model_names)
        # dvc_cmd = f"dvc run -n {stage_name} {deps_str} -d {filter_file} {outs_str} {cmd}"
    elif command == 'evaluate':
        eoi_metric_file = f"{dataset_name}-{sam_path}.json"
        eoi_metric_path = f"metrics/{eoi_metric_file}"
        cmd = (f"python main.py evaluate-intervention --baseline-file {baseline_file} "
               f"--predictions-folder {predictions_folder} --control --output {eoi_metric_path}")

        stage_name = f"evaluate-intervention-{dataset_name}-{sam_path}"

        dvc_cmd = (f"dvc run -n {stage_name} -d {baseline_file} -d {intervention_file} -d {control_file} "
                   f"-d scripts/evaluate_intervention.py -d {predictions_folder} -M {eoi_metric_path} {cmd}")
    elif command == "generate":
        #conf_name = f'conf/{dataset_name}.json'
        cmd = (f"python main.py generate-balanced --config conf/evaluate.json --seed 56 "
               f"--num-workers 8 --do-save --out-path {sam_dataset_path} --modifier-type {sam}")
        stage_name = f"generate-{sam_path}-for-{dataset_name}"
        dvc_cmd = (f"dvc run -n {stage_name} -d scripts/generate_balanced.py "
                   f"-d conf/evaluate.json -o {baseline_file} -o {intervention_file} -o {control_file} {cmd}")
    # elif command == 'filter':
    #     predictions_folder = get_predictions_folder(sam_path, 'filter')
    #     stage_name = f'filter-{dataset_name}'
    #     metric_file = f'metrics/{dataset_name}-filter.json'
    #     cmd = (f"python main.py evaluate {filter_file} --prediction-folder {predictions_folder} "
    #            f"--output -M {metric_file} --metric EM,F1 --split-reasoning")
    #     dvc_cmd = (f"dvc run -n {stage_name} -d {filter_file} -d {predictions_folder}"
    #                f"-M {metric_file} {cmd}")
    elif command == 'predict-allennlp':
        stage_name = f"predict-{model_name}-on-{dataset_name}"
        model_path = os.path.join(model_path, "model.tar.gz")
        cmd = (
            f"mkdir -p {predictions_folder} &&"
            f"allennlp predict {model_path} {baseline_file} --output-file {baseline_predictions} "
            f"--use-dataset-reader --silent && python main.py convert-allennlp {baseline_file} {baseline_predictions} "
            f"&& allennlp predict {model_path} {intervention_file} --output-file {intervention_predictions} "
            f"--use-dataset-reader --silent &&"
            f"python main.py convert-allennlp {intervention_file} {intervention_predictions} &&"
            f"allennlp predict {model_path} {control_file} --output-file {control_predictions} "
            f"--use-dataset-reader --silent && python main.py convert-allennlp {control_file} {control_predictions}"
        )

        dvc_cmd = (f"dvc run -n {stage_name} -d {model_path} -d {baseline_file} "
                   f"-d {intervention_file} -d {control_file} -o {baseline_predictions} -o {control_predictions} "
                   f"-o {intervention_predictions}  '{cmd}'")
    elif command == 'predict-baselines':
        stage_name = f"predict-baselines-on-{sam_path}"
        cmd = (
            f"python main.py predict {baseline_file} {control_file} {intervention_file} "
            f"--output-folder {predictions_folder} "
            f"--model 'random-baseline' --cls RandomBaseline "
            f"--model 'educated-baseline' --cls EducatedBaseline "
            f"--model 'informed-baseline' --cls InformedBaseline"
        )
        dvc_cmd = (
            f"dvc run -n {stage_name} -d {baseline_file} "
            f"-d {intervention_file} -d {control_file} -o {get_output_predictions_file_name(baseline_file, predictions_folder, 'random')}"
            f"-o {get_output_predictions_file_name(baseline_file, predictions_folder, 'random')}"
            f"-o {get_output_predictions_file_name(baseline_file, predictions_folder, 'random')}"
        )
    elif command == 'train-allennlp':
        cmd = f"TRAIN_SET={train_path} EVAL_SET={eval_path} CUDA=0 " \
              f"allennlp train conf/{model_name}.jsonnet -s {model_folder}"
        stage_name = f"train-{model_name}-on-{dataset_name}"
        dvc_cmd = f"dvc run -n {stage_name} -d {train_path} -d {eval_path} -o {model_path} {cmd}"
    elif command == 'train-bert-mixed':
        raise NotImplementedError()
    elif command == 'train-partial-baselines':
        raise NotImplementedError()
    elif command == 'finetune':
        raise NotImplementedError()
    elif command == 'train-t5':

        cmd = (f"PYTHONPATH='.' python scripts/t5.py --model_name_or_path {model_name} --output_dir {model_path} "
               f"--train_file {train_path} --eval_file {eval_path} "
               f"--do_train --do_eval --num_workers 8 --per_device_train_batch_size {batch_size} "
               f"{extra_args}")
        stage_name = f"train-{model_name}-on-{dataset_name}"
        dvc_cmd = (
            f"dvc run -n {stage_name} -d {train_path} -d {eval_path} -o {model_path} "
            f"{cmd}"
        )
    else:
        raise NotImplementedError()
    click.secho("Python command:", fg='green', bold=True)
    click.echo(cmd)
    click.secho("DVC command:", fg='green', bold=True)
    click.echo(dvc_cmd)
