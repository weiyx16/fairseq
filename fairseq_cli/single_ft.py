import os
import json
import argparse

task2metric = {'CoLA': ['Accuracy', 'MCC'], 'MRPC': ['Accuracy', 'F1'], 'STS-B': ['Pearson', 'Spearman']}

def get_parser():
    parser = argparse.ArgumentParser(description="single task fine-tuning script")
    parser.add_argument("--config-dir", type=str, default="../examples/roberta/config/finetuning", help="to hydra")
    parser.add_argument("--config-name", type=str, required=True, help="to hydra")
    parser.add_argument("--task-name", type=str, required=True, help="to hydra")
    parser.add_argument("--data-path", type=str, required=True, help="to hydra")
    parser.add_argument("--ckpt", type=str, required=True, help="to hydra")
    parser.add_argument("--device-id", type=int, default=0, help="to hydra; unfortunately, we only support single gpu test now")

    parser.add_argument("--run-time", type=int, default=1, help="the repeat times you want to run the fine-tune") 
    # we don't use it now; default we pick by eyes
    parser.add_argument("--stat", type=str, default="median", choices=["max", "mean", "median", "std"], help="how to report the final performance")    
    # we don't use it now; default is best
    # We finetune for 10 epochs and perform ***early stopping*** based on each task’s eval- uation metric on the dev set.
    parser.add_argument("--is-final", action="store_true", default=False, help="we use the best result as the current one") 
    # we don't use it now; default is True    
    parser.add_argument("--stat-all", action="store_true", default=False, help="output everything in stat")
    
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    # TODO: log to wandb
    PWD = os.getcwd()
    # e.g. cola: python single_ft.py --config-name cola --task-name CoLA --data-path /msrhyper-weka/public/hanhu/atrebor/glue --ckpt /exp/pretrain_models/roberta.base/model.pt --device-id 0 --run-time 5
    # e.g. RTE: python single_ft.py --config-name rte --task-name RTE --data-path /msrhyper-weka/public/hanhu/atrebor/glue --ckpt /exp/pretrain_models/roberta.base/model.pt --device-id 1 --run-time 5
    train_cmd = (
        f"rm -r {PWD}/{args.ckpt.split('/')[-2]}/{args.task_name}; rm {PWD}/{args.task_name}.txt; CUDA_VISIBLE_DEVICES={args.device_id} fairseq-hydra-train "
        f"--config-dir {args.config_dir} --config-name {args.config_name} "
        f"task.data={args.data_path}/data-bin/{args.task_name}-bin checkpoint.restore_file={args.ckpt} checkpoint.save_dir={PWD}/{args.ckpt.split('/')[-2]}/{args.task_name} common.log_file={PWD}/{args.task_name}.txt common.seed=0; "
    )
    if args.task_name in ['CoLA', 'STS-B', 'MRPC']:
        task2metricfile = {'CoLA': 'CoLA_MCC', 'MRPC': 'MRPC_F1', 'STS-B': 'STSB_SP'}
        train_cmd += f'rm {args.task_name}.log; python {task2metricfile[f"{args.task_name}"]}.py --data_root {args.data_path} --model_path {PWD}/{args.ckpt.split("/")[-2]}/{args.task_name} 2>&1 | tee {args.task_name}.log'

    metric_type = task2metric.get(args.task_name, ['Accuracy'])
    metric_results = {metric: [] for metric in metric_type}
    for i in range(args.run_time):
        print(f" >>>> Task: {args.task_name}; Round: {i}")
        print(train_cmd.replace('common.seed=0', f'common.seed={i}'))
        os.system(train_cmd.replace('common.seed=0', f'common.seed={i}'))
        # fetch results
        if args.task_name in ['CoLA', 'STS-B', 'MRPC']:
            with open(f'{args.task_name}.log') as f:
                lines = f.read().splitlines()[-len(metric_type):]
                for l in lines:
                    name, number = l.split('|')[-1].split(':')
                    metric_results[name.strip()].append(float(number.strip()))
        else:
            with open(f'{args.task_name}.txt') as f:
                lines = f.read().splitlines()
                lines = [json.loads(l) for l in lines if 'valid_best_accuracy' in l]
                best_acc = max([l['valid_best_accuracy'] for l in lines])
                metric_results['Accuracy'].append(best_acc)
            
    print(f'For CopyPaste - Task: {args.task_name}\n', metric_results)
        

if __name__ == "__main__":
    main()

