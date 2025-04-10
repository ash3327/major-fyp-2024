"""
python scripts/datasets/prepare_dataset.py <dataset_name> [--holistic]
# only use --holistic if the dataset is guaranteed to contain the entire body and two hands, otherwise detection can be screwed up.
"""
import sys
sys.path.append('.')

import argparse

def get_info(dataset):
    data_dir = "data/raw"
    output_dir = "data/kpts"
    is_video = False
    others = dict()
    match dataset:
        case 'asl_alphabet':
            print('Fetching ASL Alphabet dataset...')
            dataset = "asl_alphabet"
            subfolders = dict(
                train="asl_alphabet_train/asl_alphabet_train",
                test="asl_alphabet_test/asl_alphabet_test"
            )
            dyn = False
        case 'lexset' | 'synthetic-asl-alphabet':
            print('Fetching Lexset (Synthetic ASL Alphabet) dataset...')
            dataset = "synthetic-asl-alphabet"
            subfolders = dict(
                train="Train_Alphabet",
                test="Test_Alphabet"
            )
            dyn = False
        case 'senz3d' | 'senz3d_dataset':
            print('Fetching Senz3D dataset...')
            dataset = "senz3d_dataset"
            subfolders = dict(
                acquisitions="acquisitions"
            )
            dyn = True
        case 'roboflowasl' | 'roboflow-asl-alphabet-1':
            print('Fetching Roboflow ASL Alphabet 1 dataset...')
            dataset = "roboflow-asl-alphabet-1"
            subfolders = dict(
                train="train",
                test="test",
                valid="valid"
            )
            dyn = False
        case 'hands' | 'hands_dataset':
            print('Fetching Hands dataset...')
            dataset = "hands_dataset"
            subfolders = dict(
                data=""
            )
            dyn = True
        case 'handshape' | 'ph2014-handshape':
            print('Fetching Handshapes Dataset...')
            dataset = 'ph2014-handshape'
            subfolders = dict(
                test="test/images",
                train="train/danish_nz_ph2014"
            )
            others['annotations'] = dict(
                test="test/3359-ph2014-MS-handshape-annotations.txt",
                train="train/1miohands-v2-trainingalignment.txt"
            )
            dyn = False
        case 'lsa64' | 'lsa64_raw':
            print('Fetching LSA64 Dataset')
            dataset = 'lsa64_raw'
            subfolders = dict(
                vid="all"
            )
            dyn = True
            is_video = True
        case 'lsa64p' | 'lsa64_preprocessed':
            print('Fetching LSA64 (preprocessed) Dataset (please do not use this)')
            dataset = 'lsa64_preprocessed'
            subfolders = dict(
                vid="lsa64_hand_videos"
            )
            dyn = True
            is_video = True
        case 'phoenix' | 'phoenix-2014-t':
            print('Fetching Phoenix-2014T Weather Dataset')
            dataset = 'phoenix-2014-t'
            subfolders = dict(
                dev="PHOENIX-2014-T/features/fullFrame-210x260px/dev",
                test="PHOENIX-2014-T/features/fullFrame-210x260px/test",
                # train="PHOENIX-2014-T/features/fullFrame-210x260px/train"
            )
            dyn = True
        case 'ipn' | 'IPN_Hand':
            print('Fetching IPN Dataset')
            dataset = 'IPN_Hand'
            subfolders = dict(
                vid="videos"
            )
            dyn = True
            is_video = True
        case 'ipn2' | 'IPN_Hand2':
            print('Fetching IPN Dataset')
            dataset = 'IPN_Hand'
            subfolders = dict(
                vid="frames"
            )
            dyn = True
            is_video = True
        case _:
            raise Exception("Such dataset is not defined within `prepare_dataset.py`.")
    return data_dir, dataset, subfolders, output_dir, dyn, is_video, others

def fetch_dataset(dataset, skip=False, holistic=False, only_holistic=False):
    if only_holistic:
        from scripts.datasets.feature_extractor_holistic import extract_features as extract_features_holistic
        data_dir, dataset, subfolders, output_dir, dyn, is_video, *rest = info = get_info(dataset)
        extract_features_holistic(*info, skip=skip)
    else:
        from scripts.datasets.feature_extractor import extract_features
        from scripts.datasets.feature_extractor_holistic import extract_features as extract_features_holistic

        data_dir, dataset, subfolders, output_dir, dyn, is_video, *rest = info = get_info(dataset)
        extract_features(*info, skip=skip)
        if holistic:
            from scripts.datasets._fix_holistic import fix_dataset
            fix_dataset(data_dir, dataset, subfolders, output_dir, is_video=is_video)
    from scripts.datasets._fix_order import fix_dataset as fix_order
    fix_order(data_dir, dataset, subfolders, output_dir, is_video=is_video)

if __name__ == '__main__':
    # choices = [
    #     'asl_alphabet',
    #     'lexset', 'synthetic-asl-alphabet',
    #     'senz3d', 'senz3d_dataset',
    #     'roboflowasl', 'roboflow-asl-alphabet-1',
    #     'hands', 'hands_dataset',
    #     'handshape', 'ph2014-handshape',
    #     # dynamic datasets
    #     'lsa64', 'lsa64_raw',
    #     'phoenix', 'phoenix-2014-t',
    #     'ipn'
    # ]
    parser = argparse.ArgumentParser(description='Prepare dataset by extracting features.')
    parser.add_argument('dataset', type=str, help='Path to the dataset file', default='lexset')
    parser.add_argument('--skip', action='store_true', help='Skip processing if already done.')
    parser.add_argument('--holistic', action='store_true', help='Update the items with holistic ones if missing.')
    parser.add_argument('--only_holistic', action='store_true', help='Update the items with holistic ones if missing.')
    args = parser.parse_args()

    fetch_dataset(args.dataset, skip=args.skip, holistic=args.holistic, only_holistic=args.only_holistic)
