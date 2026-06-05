#!/usr/bin/env python3
"""一键停止 NTC 训练 / 评估进程。

用法:
    python train_stop.py           # 读取 logs/train.pid 停止主进程
    python train_stop.py --all     # 强制杀掉所有 Tool.py 相关进程（兜底）
"""
import argparse
import os
import signal
import sys
import subprocess
import time

PID_FILE = 'logs/train.pid'


def kill_pid(pid, graceful=True):
    """尝试优雅终止进程，超时后强制杀死。"""
    try:
        if sys.platform == 'win32':
            # Windows: 先尝试 taskkill /T（连同子进程一起终止）
            cmd = ['taskkill', '/PID', str(pid), '/T']
            if not graceful:
                cmd.append('/F')
            subprocess.run(cmd, capture_output=True)
        else:
            sig = signal.SIGTERM if graceful else signal.SIGKILL
            os.kill(pid, sig)
    except ProcessLookupError:
        return False
    except Exception as e:
        print(f'[train_stop] kill error: {e}')
        return False
    return True


def wait_pid_gone(pid, timeout=10):
    """等待进程消失，返回是否成功。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.5)
    return False


def find_evaluate_processes():
    """查找所有名为 Tool.py 的 Python 进程（兜底用）。"""
    pids = []
    try:
        import psutil
        for p in psutil.process_iter(['pid', 'name', 'cmdline']):
            cmdline = p.info.get('cmdline') or []
            if any('Tool.py' in s for s in cmdline):
                pids.append(p.info['pid'])
    except ImportError:
        # 没有 psutil，用 tasklist / pgrep 兜底
        if sys.platform == 'win32':
            result = subprocess.run(
                ['wmic', 'process', 'where', 'name="python.exe"', 'get', 'ProcessId,CommandLine'],
                capture_output=True, text=True
            )
            for line in result.stdout.splitlines():
                if 'Tool.py' in line:
                    parts = line.strip().split()
                    if parts:
                        try:
                            pids.append(int(parts[-1]))
                        except ValueError:
                            pass
        else:
            result = subprocess.run(
                ['pgrep', '-f', 'Tool.py'],
                capture_output=True, text=True
            )
            for line in result.stdout.splitlines():
                line = line.strip()
                if line.isdigit():
                    pids.append(int(line))
    return pids


def main():
    parser = argparse.ArgumentParser(description='一键停止 NTC 训练/评估')
    parser.add_argument('--all', action='store_true',
                        help='强制杀掉所有 Tool.py 进程（兜底）')
    parser.add_argument('--force', action='store_true',
                        help='直接强制杀死，不等待优雅退出')
    args = parser.parse_args()

    stopped = []

    # 1. 优先通过 pid 文件停止主进程
    if os.path.exists(PID_FILE):
        with open(PID_FILE) as f:
            try:
                pid = int(f.read().strip())
            except ValueError:
                pid = None

        if pid:
            print(f'[train_stop] Stopping pid={pid} from {PID_FILE} ...')
            if kill_pid(pid, graceful=not args.force):
                if not args.force and wait_pid_gone(pid, timeout=10):
                    print(f'[train_stop] pid={pid} exited gracefully.')
                elif args.force:
                    print(f'[train_stop] pid={pid} force killed.')
                else:
                    print(f'[train_stop] pid={pid} did not exit in time, force killing...')
                    kill_pid(pid, graceful=False)
                stopped.append(pid)
            else:
                print(f'[train_stop] pid={pid} already gone.')
        os.remove(PID_FILE)

    # 2. 兜底：查找并停止所有 Tool.py 进程
    if args.all or not stopped:
        pids = find_evaluate_processes()
        for pid in pids:
            if pid == os.getpid():
                continue
            print(f'[train_stop] Killing Tool.py pid={pid} ...')
            if kill_pid(pid, graceful=not args.force):
                stopped.append(pid)

    if stopped:
        print(f'[train_stop] Done. Stopped {len(stopped)} process(es): {stopped}')
    else:
        print('[train_stop] No running process found.')


if __name__ == '__main__':
    main()
