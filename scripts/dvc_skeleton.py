import os

import click

@click.command()
@click.argument('command', type=str)
@click.option('--dataset-name', '--dataset', default='squad1')
@click.option("--train-file", default="train-v1.1.json")
@click.option("--eval-file", '--dev-file', default="dev-v1.1.json")
@click.option("--batch-size", default=None)
@click.option("--extra-args", default="--fp16 --save-steps 0 --debug-features")
@click.option("--notify", type=str, default='viktor.schlegel@manchester.ac.uk')
@click.option("model_names", '--model-name', type=str, multiple=True, default='bert-base-uncased')
@click.option("model_types", '--model-type', type=str, multiple=True, default='bert')
@click.option('--gradient-accumulation-steps', type=int, default=2)
def generate_dvc(command, dataset_name,
                 train_file, eval_file, batch_size, extra_args,
                 notify, model_names, model_types, gradient_accumulation_steps):
    model_name = model_names[0]
    model_type = model_types[0]
    root_data_path = 'data/football'
    baseline_file = f"{root_data_path}/full/baseline.json"
    intervention_file = f"{root_data_path}/full/intervention.json"
    control_file = f"{root_data_path}/full/control.json"
    predictions_folder = f"{root_data_path}/full/predictions/{dataset_name}"
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
    elif command == 'train-transformers-hotpotqa':
        train_path = f"data/datasets/hotpotqa/train.json"
        eval_path = f"data/datasets/hotpotqa/dev.json"
        model_folder = f"{model_name}-hotpotqa"
        model_path = f"models/{model_folder}"
        cmd = (
            f"MODEL={model_name} CACHE_LOCATION=~/localscratch/hotpotqa/ SAVE_TO={model_path} "
            f"BATCH_SIZE={batch_size} ACC_STEPS={gradient_accumulation_steps} MODEL_TYPE={model_type} "
            f"bash scripts/cache_and_train_hotpotqa.sh"
        )
        stage_name = f"train-{model_name}-on-hotpotqa"
        dvc_cmd = (
            f"dvc run -n {stage_name} -d {train_path} -d {eval_path} -o {model_path} "
            f"{cmd}"
        )
    elif command == 'train-transformers-wikihop':
        train_path = f"data/datasets/wikihop/train.json"
        eval_path = f"data/datasets/wikihop/dev.json"
        model_folder = f"{model_name}-wikihop"
        model_path = f"models/{model_folder}"
        cmd = (
            f"MODEL={model_name} CACHE_LOCATION=~/localscratch/wikihop/ SAVE_TO={model_path} "
            f"BATCH_SIZE={batch_size} ACC_STEPS={gradient_accumulation_steps} MODEL_TYPE={model_type} "
            f"bash scripts/cache_and_train_wikihop.sh"
        )
        stage_name = f"train-{model_name}-on-wikihop"
        dvc_cmd = (
            f"dvc run -n {stage_name} -d {train_path} -d {eval_path} -o {model_path} "
            f"{cmd}"
        )
    elif command == 'predict-transformers':
        model_names = ("albert-base-v2", "albert-large-v2", "albert-xlarge-v2", "albert-xxlarge-v2",
                       'bert-base-uncased', 'bert-large-uncased', 'roberta-base', 'roberta-large')
        model_types = ["albert"] * 4 + ["bert"] * 2 + ["roberta"] * 2
        models_str = " ".join(f"--model-path models/{mp}-{dataset_name} --model-type {mt}" for mp, mt in
                              zip(model_names, model_types))
        cmd = (f"python main.py --debug --notify {notify} "
               f"predictions {baseline_file} {intervention_file} {control_file} "
               f" {models_str} "
               f"--out-folder {predictions_folder} --max-answer-length 10 {extra_args}")

        stage_name = f"predict-transformers-{dataset_name}"
        deps_str = " ".join(f"-d models/{mp}-{dataset_name}" for mp in model_names)
        outs_str = ' '.join(
            f"-o {get_output_predictions_file_name(baseline_file, predictions_folder, f'{mp}-{dataset_name}')} "
            f"-o {get_output_predictions_file_name(intervention_file, predictions_folder, f'{mp}-{dataset_name}')} "
            f"-o {get_output_predictions_file_name(control_file, predictions_folder, f'{mp}-{dataset_name}')}"
            for mp in model_names
        )
        dvc_cmd = (f"dvc run -n {stage_name} {deps_str} -d {baseline_file} "
                   f"-d {intervention_file} -d {control_file}  {outs_str} {cmd}")

    elif command == 'evaluate':
        eoi_metric_file = f"eoi.json"
        eoi_metric_path = f"metrics/football/{dataset_name}/{eoi_metric_file}"
        cmd = (f"python main.py evaluate-intervention --baseline-file {baseline_file} "
               f"--predictions-folder {predictions_folder} --control --output {eoi_metric_path} "
               f"--split-reasoning --split-num-modifier --split-sam --do-save")

        stage_name = f"evaluate-intervention-{dataset_name}"

        dvc_cmd = (f"dvc run -n {stage_name} -d {baseline_file} -d {intervention_file} -d {control_file} "
                   f"-d scripts/evaluate_intervention.py -d {predictions_folder} -M {eoi_metric_path} {cmd}")
    elif command == "generate":
        # conf_name = f'conf/{dataset_name}.json'
        cmd = (f"python main.py generate-balanced --config conf/evaluate.json --seed 56 "
               f"--num-workers 8 --do-save --out-path {root_data_path}/full --multiplier 35")
        stage_name = f"generate-sam"
        dvc_cmd = (f"dvc run -n {stage_name} -d scripts/generate_balanced.py "
                   f"-d conf/evaluate.json -o {baseline_file} -o {intervention_file} -o {control_file} {cmd}")

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

    elif command == 'train-allennlp':
        cmd = f"TRAIN_SET={train_path} EVAL_SET={eval_path} CUDA=0 " \
              f"allennlp train conf/{model_name}.jsonnet -s {model_folder}"
        stage_name = f"train-{model_name}-on-{dataset_name}"
        dvc_cmd = f"dvc run -n {stage_name} -d {train_path} -d {eval_path} -o {model_path} {cmd}"
    elif command == 'train-baselines':
        cmds = []
        dvc_cmds = []
        for masking in [None, 'q', 'p']:
            if masking:
                mask = f"-mask-{masking}"
            else:
                mask = ''
            train_file = f"{root_data_path}/split{mask}/train/combined-train.json"
            model_folder = f"models/bert{mask}-baseline"
            cmd = ("python main.py --debug --notify 'viktor.schlegel@manchester.ac.uk' train "
                   f" {train_file} "
                   "--model-path bert-base-uncased --model-type bert "
                   f"--save-model-folder {model_folder} "
                   "--num-workers 8 --per-gpu-train-batch-size 24 --max-answer-length 10  --debug-features "
                   "--num-train-epochs 15 --overwrite-output-dir "
                   "--save-steps 0 --per-gpu-eval-batch-size 64 --gradient-accumulation-steps 3 --learning-rate 5e-5")
            cmds.append(cmd)
            stage_name = f"train-bert{mask}-baseline"
            dvc_cmd = (
                f"dvc run -n {stage_name} -d {train_file} -o {model_folder} "
                f"{cmd}"
            )
            dvc_cmds.append(dvc_cmd)
        cmd = "\n".join(cmds)
        dvc_cmd = "\n".join(dvc_cmds)
    elif command == 'predict-baselines':
        cmds = []
        dvc_cmds = []

        for masking in [None, 'q', 'p']:
            if masking:
                mask = f"-mask-{masking}"
            else:
                mask = ''
            out_path = f"{root_data_path}/split{mask}/test/"
            predictions_folder = f'{out_path}predictions/'
            baseline_file = f"{out_path}baseline-test.json"
            intervention_file = f"{out_path}intervention-test.json"
            control_file = f"{out_path}control-test.json"
            if masking:
                mask = f"-mask-{masking}"
            else:
                mask = ''
                stage_name = f"predict-random-baselines"
                cmd = (
                    f"python main.py predict {baseline_file} {control_file} {intervention_file} "
                    f"--output-folder {predictions_folder} "
                    f"--model 'random-baseline' --cls RandomBaseline "
                    f"--model 'educated-baseline' --cls EducatedBaseline "
                    f"--model 'informed-baseline' --cls InformedBaseline"
                )
                outs = " ".join(
                    f"-o {get_output_predictions_file_name(f, predictions_folder, n)}"
                    for f in [baseline_file, intervention_file, control_file] for n in
                    ['random-baseline', 'educated-baseline', 'informed-baseline']
                )
                # f"-o {get_output_predictions_file_name(intervention_file, predictions_folder, 'random')}"
                # f"-o {get_output_predictions_file_name(control_file, predictions_folder, 'random')}"]
                cmds.append(cmd)
                dvc_cmd = (
                    f"dvc run -n {stage_name} -d {baseline_file} -d {intervention_file} -d {control_file} {outs} {cmd}"
                )
                dvc_cmds.append(dvc_cmd)
            model_name = f"bert{mask}-baseline"
            model_folder = f"models/bert{mask}-baseline"
            cmd = (f"python main.py --debug --notify {notify} "
                   f"predictions {baseline_file} {intervention_file} {control_file} "
                   f"--model-path {model_folder} --model-type bert "
                   f"--out-folder {predictions_folder} --max-answer-length 10 --per-gpu-eval-batch-size 64")
            cmds.append(cmd)
            stage_name = f"predict-bert{mask}-baseline"
            outs = " ".join(
                f"-o {get_output_predictions_file_name(f, predictions_folder, model_name)}"
                for f in [baseline_file, intervention_file, control_file]
            )
            dvc_cmd = (
                f"dvc run -n {stage_name} -d {model_folder} -d {baseline_file} -d {intervention_file} "
                f"-d {control_file} {outs} "
                f"{cmd}"
            )
            dvc_cmds.append(dvc_cmd)
        cmd = "\n".join(cmds)
        dvc_cmd = "\n".join(dvc_cmds)
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
    elif command == "generate-baselines":
        cmds = []
        dvc_cmds = []
        for masking in [None, 'q', 'p']:
            # conf_name = f'conf/{dataset_name}.json'
            if masking:
                mask = f"-mask-{masking}"
                mask_opt = f'--mask-{masking}' + (' --keep-answer-candidates' if masking == 'p' else '')
            else:
                mask = ''
                mask_opt = ''
            for split, multiplier, seed in [("train", 100, 56), ("test", 20, 38676)]:
                out_path = f"{root_data_path}/split{mask}/{split}/"
                baseline_file = f"{out_path}baseline-{split}.json"
                intervention_file = f"{out_path}intervention-{split}.json"
                control_file = f"{out_path}control-{split}.json"
                combined_file = f"{out_path}combined-{split}.json"
                if split == 'train':
                    combine = '--combine'
                    outs = f"-o {combined_file}"
                else:
                    combine = ''
                    outs = f' -o {baseline_file} -o {intervention_file} -o {control_file} '
                cmd = (f"python main.py generate-balanced --config conf/finetune.json "
                       f"--seed {seed} --num-workers 8 --do-save --out-path {out_path} "
                       f"--multiplier {multiplier} --split {split} "
                       f"{combine} {mask_opt}")
                cmds.append(cmd)

                stage_name = f"generate-{split}{mask}"
                dvc_cmd = (f"dvc run -n {stage_name} -d scripts/generate_balanced.py "
                           f"-d conf/finetune.json {outs} "
                           f"{cmd}")
                dvc_cmds.append(dvc_cmd)
        cmd = "\n".join(cmds)
        dvc_cmd = "\n".join(dvc_cmds)
    elif command == 'evaluate-baselines':
        cmds = []
        dvc_cmds = []
        for masking in [None, 'q', 'p']:
            split = 'test'
            mask = f"-mask-{masking}" if masking else ''
            out_path = f"{root_data_path}/split{mask}/{split}/"
            baseline_file = f"{out_path}baseline-{split}.json"
            predictions_folder = f'{out_path}predictions/'
            intervention_file = f"{out_path}intervention-{split}.json"''
            control_file = f"{out_path}control-{split}.json"
            metric = f"metrics/football/baselines/baselines{mask}.json"
            cmd = (f"python main.py evaluate {baseline_file} "
                   f"{intervention_file} {control_file} --prediction-folder {predictions_folder} "
                   f"--output {metric} --metric EMRelaxed")
            stage_name = f"evaluate-baselines{mask}"
            dvc_cmd = (
                (f"dvc run -n {stage_name} -d {baseline_file} -d {intervention_file} -d {control_file} "
                 f"-d {predictions_folder} -M {metric} "
                 f"'{cmd}'")
            )
            cmds.append(cmd)
            dvc_cmds.append(dvc_cmd)
        cmd = "\n\n".join(cmds)
        dvc_cmd = "\n\n".join(dvc_cmds)
    else:
        raise NotImplementedError()

    click.secho("Python command:", fg='green', bold=True)
    click.echo(cmd)
    click.secho("DVC command:", fg='green', bold=True)
    click.echo(dvc_cmd)


from scripts.utils import get_output_predictions_file_name
