import argparse
import json
import os.path as osp
from collections import defaultdict


from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description='Get img list txt for TAO')
    parser.add_argument('-t', '--tao', help='path of TAO json file')
    parser.add_argument('--fps', type=int, default=1, help='fps to run detector and tracker on')
    return parser.parse_args()

def get_img_lst(ann_file, out_file):
    '''
    * Get img_lst from ann_file and write the result to out_file
    '''
    with open(ann_file, 'r') as f:
        tao = json.load(f)
    
    with open(out_file, 'w') as f:
        for image in tqdm(tao['images']):
            f.write("{}\n".format(image['file_name']))

def main():
    args = parse_args()

    suffix = "" if args.fps == 1 else "_{}fps".format(args.fps)
    for file in [
            f'train_AOA{suffix}.json', f'validation_AOA{suffix}.json', f'test_AOA{suffix}.json'
    ]:
        print(f'Generate img list from {file}')
        prefix = file.split('.')[0].split('_')[0]
        out_file = f'{prefix}_img_list{suffix}.txt'
        get_img_lst(osp.join(args.tao, file), osp.join(args.tao, '..', out_file))

if __name__ == '__main__':
    main()