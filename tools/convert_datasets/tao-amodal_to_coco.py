import argparse
import os.path as osp
from collections import defaultdict

import mmcv
from tao.toolkit.tao import Tao
from tqdm import tqdm
import json

def parse_args():
    parser = argparse.ArgumentParser(
        description='Make annotation files for TAO')
    parser.add_argument('-t', '--tao', help='path of TAO json file')
    parser.add_argument('--fps', default=1, help='fps to run tracker at')
    parser.add_argument(
        '--filter-classes',
        action='store_true',
        help='whether filter 1230 classes to 482.')
    return parser.parse_args()


def get_classes(tao_path, filter_classes=True):
    '''
    Input: tao_path, the path to tao amodal annotation
    Output:
        a list contains all LVIS categories in TAO-Amodal, sorted by cat ids

    ******* Implementation Detail *******
    * Collect LVIS categories, the ID of LVIS categories in TAO-Amodal match with ID in LVIS
    * In training/evaluation, the order of categories inside tao["categories"] decide their corresponding label id
        e.g tao["categories] = [{id:1, name="aborn"}, {id:2, name:"animal"}]
        -> cat_id label_id(used in model prediction)
            1       0
            2       1
    * The tao["categories"](list) are sorted by cat_id, which has the same order as lvis["categories"]
        Thus, the label_id of each category will be the same across two datasets
    '''
    train = mmcv.load(osp.join(tao_path, 'train_with_freeform_amodal_boxes_may12_2022.json'))
    train["categories"] = [cat for cat in train["categories"]  if cat["synset"] != "unknown"]
    lvis_categories_id = set([cat["id"] for cat in train["categories"]  if cat["synset"] != "unknown"])

    train_classes = list(set([_['category_id'] for _ in train['annotations'] if _['category_id'] in lvis_categories_id]))
    print(f'TAO train set contains {len(train_classes)} categories.')

    val = mmcv.load(osp.join(tao_path, 'validation_with_freeform_amodal_boxes_Aug10_2022.json'))
    val_classes = list(set([_['category_id'] for _ in val['annotations'] if _['category_id'] in lvis_categories_id]))
    print(f'TAO val set contains {len(val_classes)} categories.')

    test = mmcv.load(osp.join(tao_path, 'test_with_freeform_amodal_boxes_Jan10_2023.json'))
    test_classes = list(set([_['category_id'] for _ in test['annotations'] if _['category_id'] in lvis_categories_id]))
    print(f'TAO test set contains {len(test_classes)} categories.')

    tao_classes = set(train_classes + val_classes + test_classes)
    print(f'TAO totally contains {len(tao_classes)} categories.')

    tao_classes = [_ for _ in train['categories'] if _['id'] in tao_classes]

    with open(osp.join(tao_path, 'tao_classes.txt'), 'wt') as f:
        for c in tao_classes:
            name = c['name']
            f.writelines(f'{name}\n')

    if filter_classes:
        return tao_classes
    else:
        return train['categories']

def vid_img_map(tao):
    '''
    * Return:
        vid_to_images[vid] = [image0, image1, image2]
    '''
    vid_to_images = defaultdict(list)
    for image in tao["images"]:
        vid = image["video_id"]
        vid_to_images[vid].append(image)
    return vid_to_images

def img_ann_map(tao):
    '''
    * Return:
        img_to_anns[image] = [ann0, ann1, ann2]
            ann{i} would be a dictionary containing attributes like bbox, area,
    '''
    img_to_anns = defaultdict(list)
    for ann in tao["annotations"]:
        img = ann["image_id"]
        img_to_anns[img].append(ann)
    return img_to_anns

def convert_tao(file, classes):
    raw = mmcv.load(file)
    vid_to_images = vid_img_map(raw)
    image_to_anns = img_ann_map(raw)

    out = defaultdict(list)
    out['tracks'] = raw['tracks'].copy()
    out['info'] = raw['info'].copy()
    out['licenses'] = raw['licenses'].copy()
    out['categories'] = classes  # only LVIS categories 

    for video in tqdm(raw['videos']):
        img_infos = vid_to_images[video['id']]
        img_infos = sorted(img_infos, key=lambda x: x['frame_index'])
        frame_range = img_infos[1]['frame_index'] - img_infos[0]['frame_index']
        video['frame_range'] = frame_range
        out['videos'].append(video)
        for i, img_info in enumerate(img_infos):
            img_info['frame_id'] = i
            img_info['neg_category_ids'] = video['neg_category_ids']
            img_info['not_exhaustive_category_ids'] = video[
                'not_exhaustive_category_ids']
            out['images'].append(img_info)
            ann_infos = image_to_anns[img_info['id']]
            for ann_info in ann_infos:
                ann_info['instance_id'] = ann_info['track_id']
                
                # Assign bbox with amodal bbox, so that we could evaluate with TAO toolkit
                ann_info["bbox"] = ann_info["amodal_bbox"]
                ann_info["area"] = ann_info["amodal_bbox"][2] * ann_info["amodal_bbox"][3] 
                out['annotations'].append(ann_info)

    print(len(raw['images']))
    print(len(out['images']))
    assert len(out['videos']) == len(raw['videos'])
    assert len(out['images']) == len(raw['images'])
    assert len(out['annotations']) == len(raw['annotations'])
    return out


def main():
    args = parse_args()

    classes = get_classes(args.tao, args.filter_classes)
    print(f'convert with {len(classes)} classes')
    suffix = "" if args.fps == 1 else "_{}fps".format(args.fps)
    for file in [
             f'validation_with_freeform_amodal_boxes_Aug10_2022_oof_visibility{suffix}.json', 
    ]:
        print(f'convert {file}')
        out = convert_tao(osp.join(args.tao, file), classes)
        c = '_482' if args.filter_classes else ''
        prefix = file.split('.')[0].split('_')[0]
        out_file = f'{prefix}{c}_AOA{suffix}.json'
        mmcv.dump(out, osp.join(args.tao, out_file))


if __name__ == '__main__':
    main()
