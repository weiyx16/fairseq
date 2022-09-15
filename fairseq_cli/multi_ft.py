import os
import torch
import multiprocessing as mp

data_path = os.environ.get('data_path')
pre_model = os.environ.get('pre_model')
def train(is_end, gpu_id):
    if gpu_id == 0:
        os.system(f'MASTER_PORT=12345 python single_ft.py --config-name mnli --task-name MNLI --data-path {data_path} --ckpt {pre_model} --device-id 0 --run-time 5')
        with is_end.get_lock():
            is_end.value += 1
    elif gpu_id == 1:
        os.system(f'MASTER_PORT=12346 python single_ft.py --config-name qqp --task-name QQP --data-path {data_path} --ckpt {pre_model} --device-id 1 --run-time 5')
        with is_end.get_lock():
            is_end.value += 1
    elif gpu_id == 2:
        os.system(f'MASTER_PORT=12347 python single_ft.py --config-name qnli --task-name QNLI --data-path {data_path} --ckpt {pre_model} --device-id 2 --run-time 5')
        with is_end.get_lock():
            is_end.value += 1
    elif gpu_id == 3:
        os.system(f'MASTER_PORT=12348 python single_ft.py --config-name sst_2 --task-name SST-2 --data-path {data_path} --ckpt {pre_model} --device-id 3 --run-time 5;MASTER_PORT=12348 python single_ft.py --config-name cola --task-name CoLA --data-path {data_path} --ckpt {pre_model} --device-id 3 --run-time 5;MASTER_PORT=12348 python single_ft.py --config-name mrpc --task-name MRPC --data-path {data_path} --ckpt {pre_model} --device-id 3 --run-time 5;MASTER_PORT=12348 python single_ft.py --config-name sts_b --task-name STS-B --data-path {data_path} --ckpt {pre_model} --device-id 3 --run-time 5;MASTER_PORT=12348 python single_ft.py --config-name rte --task-name RTE --data-path {data_path} --ckpt {pre_model} --device-id 3 --run-time 5;')
        with is_end.get_lock():
            is_end.value += 1


def main(os_context):
    is_end = os_context.Value("i", 0)
    group0 = os_context.Process(target=train, args=(is_end, 0))
    group1 = os_context.Process(target=train, args=(is_end, 1))
    group2 = os_context.Process(target=train, args=(is_end, 2))
    group3 = os_context.Process(target=train, args=(is_end, 3))

    group0.start()
    group1.start()
    group2.start()
    group3.start()

    group0.join()
    group1.join()
    group2.join()
    group3.join()

    print('finish group: ', is_end.value)


if __name__ == '__main__':
    print('GPU Count: ', torch.cuda.device_count())
    context = mp.get_context('spawn')
    main(context)