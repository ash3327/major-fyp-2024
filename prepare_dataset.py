import argparse
from feature_extractor import extract_features

def get_info(dataset):
    data_dir = "data/raw"
    output_dir = "data/kpts"
    is_video = False
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
                train="train"
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
        case 'phoenix' | 'phoenix-2014-t':
            print('Fetching Phoenix-2014T Weather Dataset')
            dataset = 'phoenix-2014-t'
            subfolders = dict(
                dev="PHOENIX-2014-T/features/fullFrame-210x260px/dev",
                test="PHOENIX-2014-T/features/fullFrame-210x260px/test",
                train="PHOENIX-2014-T/features/fullFrame-210x260px/train"
            )
            dyn = True
        case _:
            raise Exception("Such dataset is not defined within `prepare_dataset.py`.")
    return data_dir, dataset, subfolders, output_dir, dyn, is_video

def fetch_dataset(dataset):
    extract_features(*get_info(dataset))

if __name__ == '__main__':
    choices = [
        'asl_alphabet',
        'lexset', 'synthetic-asl-alphabet',
        'senz3d', 'senz3d_dataset',
        'roboflowasl', 'roboflow-asl-alphabet-1',
        'hands', 'hands_dataset',
        'handshape', 'ph2014-handshape',
        # dynamic datasets
        'lsa64', 'lsa64_raw',
        'phoenix', 'phoenix-2014-t'
    ]
    parser = argparse.ArgumentParser(description='Prepare dataset by extracting features.')
    parser.add_argument('dataset', type=str, help='Path to the dataset file', default='lexset', choices=choices)
    args = parser.parse_args()

    fetch_dataset(args.dataset)
