import os
import numpy as np
from prepare_dataset import get_info
import argparse

# filepath: /d:/kht3327/_Projects/Major FYP/proj-structured/fixing.py


def recount_hands(dataset, split):
    """
    Recounts the number of hands in the dataset and updates the .npy file.
    """
    data_dir, dataset, subfolders, output_dir, dyn = get_info(dataset)
    splits = list(subfolders.keys())

    # Choose split if not provided or invalid
    if split is None or split not in splits:
        inpt = input(f"Choose from the splits: {splits}\n>>> ") if len(splits) > 1 else splits[0]
        if inpt in splits:
            split = inpt
            print(f"Fetching dataset with split {split}...")
        else:
            print("Invalid split. Terminating...")
            exit(1)
    else:
        print(f"Fetching dataset with split {split}...")

    npy_file = os.path.join(output_dir, dataset, f"record_{split}.npy")
    
    if not os.path.exists(npy_file):
        print(f"Error: {npy_file} does not exist.")
        return

    print(f"Loading data from {npy_file}...")
    data = np.load(npy_file, allow_pickle=True)

    problem_count = 0
    updated_data = []
    wrong_items = []
    count_dict = {}  # Dictionary to store counts for pairs of (num_hands, item[2])
    
    for item in data:
        landmarks = item[3]['hands']
        num_hands = sum(np.any(hand != 0) for hand in landmarks)
        if num_hands != item[2]:
            problem_count += 1
            wrong_items.append((item[0], num_hands, item[2]))  # Append wrong item details
            
            # Update the count dictionary
            pair = (num_hands, item[2])
            if pair in count_dict:
                count_dict[pair] += 1
            else:
                count_dict[pair] = 1
        
        item[2] = num_hands  # Update the number of hands
        updated_data.append(item)

    # Save wrong items to a separate .npy file
    wrong_items_file = os.path.join(output_dir, dataset, f"wrong_items_{split}.npy")
    np.save(wrong_items_file, np.array(wrong_items, dtype=object))
    print(f"Wrong items saved to {wrong_items_file}")

    # Save count dictionary to a separate .npy file
    print(count_dict)

    print(f"Number of wrong data: {problem_count} of {len(data)} ({problem_count/len(data)*100}%)")

    print(f"Saving updated data to {npy_file}...")
    np.save(npy_file, np.array(updated_data, dtype=object))
    print("Recounting complete!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Recount the number of hands in the dataset.")
    parser.add_argument('-d', '--dataset', type=str, required=True, help='Name of the dataset')
    parser.add_argument('-s', '--split', type=str, required=False, default=None, help='Name of the split (e.g., train, test)')
    args = parser.parse_args()

    recount_hands(args.dataset, args.split)