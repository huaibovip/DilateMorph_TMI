import json
import os
import os.path as osp

metainfo = dict(
    dataset_type='lpba_registration_dataset',
    task_name='registration',
    classes=[
        'Unknown', 'L-Superior-Frontal-Gyrus', 'R-Superior-Frontal-Gyrus',
        'L-Middle-Frontal-Gyrus', 'R-Middle-Frontal-Gyrus',
        'L-Inferior-Frontal-Gyrus', 'R-Inferior-Frontal-Gyrus',
        'L-Precentral-Gyrus', 'R-Precentral-Gyrus',
        'L-Middle-Orbitofrontal-Gyrus', 'R-Middle-Orbitofrontal-Gyrus',
        'L-Lateral-Orbitofrontal-Gyrus', 'R-Lateral-Orbitofrontal-Gyrus',
        'L-Gyrus-Rectus', 'R-Gyrus-Rectus', 'L-Postcentral-Gyrus',
        'R-Postcentral-Gyrus', 'L-Superior-Parietal-Gyrus',
        'R-Superior-Parietal-Gyrus', 'L-Supramarginal-Gyrus',
        'R-Supramarginal-Gyrus', 'L-Angular-Gyrus', 'R-Angular-Gyrus',
        'L-Precuneus', 'R-Precuneus', 'L-Superior-Occipital-Gyrus',
        'R-Superior-Occipital-Gyrus', 'L-Middle-Occipital-Gyrus',
        'R-Middle-Occipital-Gyrus', 'L-Inferior-Occipital-Gyrus',
        'R-Inferior-Occipital-Gyrus', 'L-Cuneus', 'R-Cuneus',
        'L-Superior-Temporal-Gyrus', 'R-Superior-Temporal-Gyrus',
        'L-Middle-Temporal-Gyrus', 'R-Middle-Temporal-Gyrus',
        'L-Inferior-Temporal-Gyrus', 'R-Inferior-Temporal-Gyrus',
        'L-Parahippocampal-Gyrus', 'R-Parahippocampal-Gyrus',
        'L-Lingual-Gyrus', 'R-Lingual-Gyrus', 'L-Fusiform-Gyrus',
        'R-Fusiform-Gyrus', 'L-Insular-Cortex', 'R-Insular-Cortex',
        'L-Cingulate-Gyrus', 'R-Cingulate-Gyrus', 'L-Caudate', 'R-Caudate',
        'L-Putamen', 'R-Putamen', 'L-Hippocampus', 'R-Hippocampus'
    ],
    palette=[[0, 0, 0], [255, 128, 128], [255, 128, 0], [0, 0, 255],
             [255, 128, 255], [128, 127, 0], [0, 0, 255], [0, 255, 255],
             [128, 0, 0], [255, 0, 128], [128, 128, 64], [128, 0, 255],
             [255, 128, 127], [128, 128, 255], [0, 255, 255], [0, 255, 0],
             [128, 128, 64], [0, 127, 128], [0, 255, 255], [255, 255, 0],
             [0, 128, 128], [255, 128, 255], [64, 0, 128], [128, 255, 128],
             [0, 0, 255], [127, 127, 255], [255, 127, 0], [255, 127, 128],
             [0, 127, 0], [160, 0, 0], [83, 166, 166], [255, 255, 127],
             [0, 0, 128], [128, 0, 64], [0, 128, 255], [92, 92, 237],
             [192, 128, 255], [17, 207, 255], [255, 0, 128], [105, 180, 31],
             [64, 255, 0], [255, 255, 0], [0, 255, 255], [35, 231, 216],
             [255, 0, 0], [0, 128, 255], [128, 255, 255], [64, 128, 255],
             [255, 0, 255], [0, 48, 255], [255, 0, 128], [128, 160, 48],
             [255, 128, 0], [133, 10, 31], [0, 255, 255]],
)


def write_ann(file, image_names, prefix=''):
    data_list = list()
    for i in range(len(image_names)):
        for j in range(len(image_names)):
            if i != j:
                src_path = f'{prefix}/{image_names[i]}'
                dst_path = f'{prefix}/{image_names[j]}'
                data = dict(source_path=src_path, target_path=dst_path)
                data_list.append(data)

    data_ann = dict()
    data_ann['metainfo'] = metainfo
    data_ann['data_list'] = data_list
    os.makedirs(os.path.dirname(file), exist_ok=True)

    print(f'Generate annotation files {osp.basename(file)}...')
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data_ann, f, ensure_ascii=False, sort_keys=False, indent=4)


if __name__ == '__main__':
    # dataset: https://github.com/ZAX130/RDP
    # https://drive.usercontent.google.com/download?id=1mFzZDn2qPAiP1ByGZ7EbsvEmm6vrS5WO&export=download&authuser=0

    root = 'data/reg/LPBA_data'

    names = os.listdir(osp.join(root, 'Train'))
    write_ann(
        osp.join('annotations', 'train_scan2scan.json'),
        image_names=names,
        prefix='Train')

    names = os.listdir(osp.join(root, 'Val'))
    write_ann(
        osp.join('annotations', 'test_scan2scan.json'),
        image_names=names,
        prefix='Val')
