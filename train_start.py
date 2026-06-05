#!/usr/bin/env python3
"""一键启动 NTC 训练 / 评估。

用法:
    -- 训练全精度
    python train_start.py --mode train-uc
    -- 训练压缩；可恢复
    python train_start.py --mode train-bc --ckpt checkpoints/xxx
    -- 评估
    python train_start.py --mode eval --ckpt checkpoints/xxx
    # 后台运行
    python train_start.py --mode train-uc --daemon
"""
import argparse
import os
import subprocess
import sys
import time


def detect_workers():
    try:
        import torch
        n = torch.cuda.device_count()
        # 单卡显存很小，默认每卡 2 个 worker；如果检测不到 GPU 就用 1
        return max(1, n * 2)
    except Exception:
        return 1


def main():
    parser = argparse.ArgumentParser(description='一键启动 NTC 训练/评估')
    parser.add_argument('--mode', choices=['train-uc', 'train-bc', 'eval'], default='train-uc',
                        help='运行模式')
    parser.add_argument('--config', default='configs/bc1_bcf05k.yaml',
                        help='YAML 配置文件路径')
    parser.add_argument('--ckpt', default=None,
                        help='eval 模式必需的 checkpoint 目录')
    parser.add_argument('--workers', type=int, default=None,
                        help=f'并行进程数（默认自动检测，当前建议 {detect_workers()}）')
    parser.add_argument('--gpus', default=None,
                        help='指定 GPU，如 "0" 或 "0,1"')
    parser.add_argument('--materials', default=None,
                        help='逗号分隔的材质名，默认全部')
    parser.add_argument('--vis-dir', default=None,
                        help='可视化输出目录')
    parser.add_argument('--log-dir', default='logs',
                        help='日志存放目录')
    parser.add_argument('--daemon', action='store_true',
                        help='后台运行（ detached ）')
    args = parser.parse_args()

    if args.mode == 'eval' and not args.ckpt:
        print('ERROR: eval 模式需要 --ckpt <checkpoint_dir>')
        sys.exit(1)

    workers = args.workers if args.workers is not None else detect_workers()

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(args.log_dir, f'{args.mode}_{timestamp}.log')
    os.makedirs(args.log_dir, exist_ok=True)

    cmd = [sys.executable, 'Tool.py',
           '--config', args.config,
           '--num-workers', str(workers)]

    if args.mode == 'train-uc':
        cmd.append('--train-uc')
    elif args.mode == 'train-bc':
        cmd.append('--train-bc')
        if args.ckpt:
            cmd.extend(['--ckpt', args.ckpt])
    else:
        cmd.extend(['--ckpt', args.ckpt])

    if args.gpus:
        cmd.extend(['--gpus', args.gpus])
    if args.materials:
        cmd.extend(['--materials', args.materials])
    if args.vis_dir:
        cmd.extend(['--vis-dir', args.vis_dir])

    pid_file = os.path.join(args.log_dir, 'train.pid')

    print(f'[train_start] mode={args.mode}, workers={workers}')
    print(f'[train_start] log -> {log_file}')
    print(f'[train_start] cmd: {" ".join(cmd)}')

    if args.daemon:
        # Windows / bash 通用后台启动
        if sys.platform == 'win32':
            # Windows: 用 CREATE_NEW_PROCESS_GROUP 避免 Ctrl+C 传过去
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
            proc = subprocess.Popen(
                cmd,
                stdout=open(log_file, 'w'),
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
        else:
            proc = subprocess.Popen(
                cmd,
                stdout=open(log_file, 'w'),
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        with open(pid_file, 'w') as f:
            f.write(str(proc.pid))

        print(f'[train_start] daemon started, pid={proc.pid}')
        print(f'[train_start] 停止: python train_stop.py')
    else:
        # 前台运行，直接阻塞
        with open(log_file, 'w') as f:
            try:
                proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
                with open(pid_file, 'w') as pf:
                    pf.write(str(proc.pid))
                proc.wait()
            except KeyboardInterrupt:
                print('\n[train_start] Interrupted, terminating...')
                proc.terminate()
                proc.wait()
                if os.path.exists(pid_file):
                    os.remove(pid_file)
                sys.exit(130)

        if os.path.exists(pid_file):
            os.remove(pid_file)


if __name__ == '__main__':
    main()
