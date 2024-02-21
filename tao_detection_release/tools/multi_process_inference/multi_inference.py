# coding=utf-8
import os
import cv2
import sys
import tempfile
import math
import subprocess
import multiprocessing
import pdb
from argparse import ArgumentParser
from tqdm import tqdm

def dump_testfile(task, index, tmp_dir):
    filename = os.path.join(tmp_dir.name, "testfile_%d.txt" % index)
    fp = open(filename, "w+")
    for t in task:
        s = "%s\n" % t
        fp.write(s)
    fp.close()
    return  filename

def parse_textfile(testfile):
    with open(testfile) as fp:
        namelist = [x.strip() for x in fp.readlines()]
    return namelist

# split the arr into N chunks
def chunks(arr, m):
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]

def single_process(index, task, gpu, split):

    print(("任务%d处理%d张图片" % (index, len(task))))

    # 写文件
    tmp_dir = tempfile.TemporaryDirectory()
    filename = dump_testfile(task, index, tmp_dir)

    out_str = subprocess.check_output(["python", file, "--gpuid=%s" % str(gpu), "--img_list=%s" % filename, 
            "--out_dir=%s" % out_dir, "--batch_size=%d" % batch_size, "--split=%s" % split])

    print(("任务%d处理完毕！" % (index)))

def default_args():
    parser = ArgumentParser(
        description='Run Detector on TAO')
    parser.add_argument('split', type=str, help='[/train/val/test]')
    parser.add_argument('--out_dir', type=str, help='path to output dectection dir')
    parser.add_argument('--fps', type=int, help='fps to run detector at', default=1)

    args = parser.parse_args()
    return args

if "__main__" == __name__:
    args = default_args()

    gpu_list = [0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
    # gpu_list = [0,0,1,1]
    ###
    # process arguments
    ###
    split = args.split
    file = "tools/multi_process_inference/inference.py"
    
    fps_suffix = "" if args.fps == 1 else "_{}fps".format(args.fps)
    img_txt = f'./data/tao/{split}_img_list{fps_suffix}.txt'
    
    out_dir = args.out_dir
    batch_size = 1

    # create output dir
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # 解析dir
    img_list = parse_textfile(img_txt)
    print(f"总共{len(img_list)}张图片")

    # 分任务
    task_num = len(gpu_list)
    tasks = chunks(img_list, task_num)

    # 创建进程
    processes=list()
    print("Collect process...")
    for idx, (task, gpu) in tqdm(enumerate(zip(tasks, gpu_list))):
        processes.append(multiprocessing.Process(target=single_process,args=(idx, task, gpu, split)))

    print("Running detector...")
    for process in tqdm(processes):
        process.start()

    print("Merge results...")
    for process in tqdm(processes):
        process.join()
