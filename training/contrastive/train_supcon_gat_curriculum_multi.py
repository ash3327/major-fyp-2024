"""
Running the code:
    python training/contrastive/train_supcon.py

Monitor with TensorBoard:
    tensorboard --logdir runs/hand_contrastive_learning_structured/v1
"""

# train_supcon_structured.py
import sys
sys.path.append('.')  # Ensure imports work from the project root

import os
import torch
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from datetime import datetime

from scripts.fake_data.contrastive_data_dataset import HandPoseContrastiveDataset
from scripts.hand_only_supervised.hand_supervised_dataset import LabelledHandDataset, CombinedLabelledHandDataset
from training.contrastive.augments import augment as augment_hand, augment_pair as augment_handpair
from training.contrastive.augments import generate_random_rotation_object, generate_random_scaling_vector, apply_transform, vectorized_apply_transform, normalize  # Make sure path is correct

from training.contrastive.model import HandEncoder, HandEncoder_6DOF
from training.contrastive.model_gat import HandEncoderGAT3dof, HandEncoderGAT6dof, graph_transform, graph_transform_complex
from training.contrastive.model_gcn import HandEncoderGCN3dof, HandEncoderGCN6dof
from training.contrastive.losses import info_nce_loss, supcon_loss, info_nce_loss_from_matrix
from training.contrastive.evals import extract_embeddings, evaluate_knn

from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
import math

# train info
#
version_id = 1
dump = False

current_time = datetime.now().strftime('%Y%m%d%H%M%S')
experiment_name = f'hand_contrastive_learning_structured' if not dump else 'hcls_dump'
train_path_root = f'runs/{experiment_name}/v{version_id}/{current_time}'

model_checkpoint_path = 'runs/hand_contrastive_learning/v4/20250401140023/checkpoints/best.pth'#None
start_epoch = 0
model_checkpoint_path = 'runs/hand_contrastive_learning_structured/v1/20250402190449/checkpoints/best.pth'
start_epoch = 1000
model_checkpoint_path = 'runs/hand_contrastive_learning_structured/v1/20250402210020/checkpoints/best.pth'
start_epoch = 11000
model_checkpoint_path = 'runs/hand_contrastive_learning_structured/v1/20250403104741/checkpoints/best.pth'
start_epoch = 13000
model_checkpoint_path = None
start_epoch = 0

# hyperparameters
batch_size = 256
grid_size = 4
n_aug_pregenerated = 32
embedding_dim = 128
initial_lr = 0.01
learning_rate = 0.01#1e-4
num_epochs = 10000  # Adjust as needed
temperature = 0.1
eval_interval = 10  # Evaluate every 10 epochs
k_neighbors = 5  # Number of neighbors for k-NN
num_samples_unsup = 50 * batch_size
num_samples_sup = 100 * batch_size
patience = 1000
do_sup = True

# 202504031040et
# added back empty landmarks
patience = 500
num_epochs = 2000
num_samples_unsup = 700 * batch_size

# 202504031508
initial_lr = 0.01
learning_rate = 1e-3
num_epochs = 5000
patience = 500
do_sup = True
do_unsup = False
lr_jump_epoch = 500

# GAT 202504051632
initial_lr = 1e-3#1e-4
learning_rate = 1e-4
eval_interval = 1
do_sup = True
do_unsup = False
eval_interval = 2
check_profile = False
do_pool = False
num_it_per_epoch = num_samples_unsup = 80 * batch_size

# curriculum
do_sup = True
do_unsup = True#False
fast_warmup_epochs = 1000
slow_warmup_epochs = 10000
large_angle = 2*np.pi 
small_angle = np.pi/6
proportion = lambda i,warmup: (1 if i > warmup else i/warmup)
angle_batch_schedule = lambda i: large_angle * (1 if i > fast_warmup_epochs else i/fast_warmup_epochs) # 0 -> 1
angle_aug_schedule = lambda i: small_angle * (1 if i > fast_warmup_epochs else i/fast_warmup_epochs) # 0 -> 1
angle_sup_aug_schedule = lambda i: large_angle * (1 if i > slow_warmup_epochs else i/slow_warmup_epochs) # 0 -> 1
scale_range_schedule = lambda i: (1-proportion(i,slow_warmup_epochs),1+4*proportion(i,slow_warmup_epochs)) # 1,1 -> 0,5

# unsup
max_dataset_size = 100 * batch_size#80 * batch_size

# 20250410
do_full = False#True
lexset_only = False#True

# device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epoch = 0

def structured_collate_fn_sup(batch_list):
    """Optimized collate function for supervised data."""
    global epoch
    # Extract joints and convert to tensor in one go
    labels_batch, joints_batch = zip(*batch_list) # [B, 21, 3], [B,]
    joints_batch = torch.stack(joints_batch)  
    labels_batch = torch.tensor(labels_batch, dtype=torch.long)  # [B,]
    
    B = len(batch_list)
    
    # Pre-compute transformations
    batch_rotation = generate_random_rotation_object(max_angle=angle_batch_schedule(epoch))
    rotations = generate_random_rotation_object(batch_size=grid_size, max_angle=angle_aug_schedule(epoch))
    rotations = [batch_rotation * r for r in rotations]
    scalings = torch.from_numpy(generate_random_scaling_vector(batch_size=grid_size, scale_range=scale_range_schedule(epoch))) # [grid_size, 3]
    
    # Pre-allocate output tensor on device
    output_batch = torch.zeros(B, grid_size, 21, 3, device=device) # B, G, 21, 3
    labels_batch = labels_batch.reshape(-1,1).tile(1,grid_size)# B, G,
    
    # Apply transformations in parallel for each grid position
    for j, (rotation_j, scaling_j) in enumerate(zip(rotations, scalings)):
        output_batch[:, j] = vectorized_apply_transform(joints_batch, rotation_j, scaling_j)
        
    # shape: [B, grid_size, 21, 3]
    output_batch = output_batch.view(-1, 21, 3)  # Flatten grid size [B*G,21,3]
    labels_batch = labels_batch.view(-1) # [B*G]
    return labels_batch, output_batch

def structured_collate_fn(batch_list):
    """Optimized collate function for unsupervised data."""
    global epoch
    B = len(batch_list)
    
    # Pre-compute transformations
    batch_rotation = generate_random_rotation_object(max_angle=angle_batch_schedule(epoch))
    rotations = [batch_rotation * generate_random_rotation_object(max_angle=angle_aug_schedule(epoch)) 
                 for _ in range(grid_size)]
    scalings = torch.from_numpy(generate_random_scaling_vector(batch_size=grid_size)) # [grid_size, 3]
    
    # Random indices for all batches at once
    rand_indices = torch.randint(0, n_aug_pregenerated, (B,))
    
    # Convert all poses to tensor and normalize in one go
    poses_batch = torch.stack([torch.from_numpy(batch_list[i][rand_indices[i]]) 
                             for i in range(B)])  # [B, 21, 3]
    
    # Pre-allocate output tensor on device
    output_batch = torch.zeros(B, grid_size, 21, 3, device=device)
    
    # Apply transformations in parallel for each grid position
    for j, (rotation_j, scaling_j) in enumerate(zip(rotations, scalings)):
        output_batch[:, j] = vectorized_apply_transform(poses_batch, rotation_j, scaling_j)
    
    # shape: [B, grid_size, 21, 3]
    output_batch = output_batch.view(-1, 21, 3)  # Flatten grid size
    return output_batch

if __name__ == '__main__':
    large_angle = np.pi*2
    small_angle = np.pi/6
    large_angle = 0
    small_angle = 0
    extra_text = f"[With Lexset, Handshape and Senz3d] {'<lexset only>' if lexset_only else '<all sup datasets>'} <more complex joints> Augmentation with linear curriculum scheduling (sup: use sup_aug_schedule: 0..2pi (10k ep), unsup: 0..2pi, 0..pi/6 (1k ep); do_norm_after_output=True), No pool, 4->3 layers"
    def aug_pair(*x):
        global epoch
        return augment_handpair(*x, maxangle=angle_batch_schedule(epoch))
    # aug_pair = lambda *x: augment_handpair(*x, max_angle=large_angle)
    def aug(x):
        global epoch
        x = augment_hand(x, max_angle=angle_sup_aug_schedule(epoch))
        return x
    # aug = lambda x: augment_hand(x, max_angle=small_angle)

    # initialize datasets and dataloaders
    # supervised dataset
    if lexset_only:
        dataset_sup = LabelledHandDataset(dataset_name='lexset', split='train', augment=aug)
    else:
        dataset_sup = CombinedLabelledHandDataset(
            dataset_names={
                'lexset': 'train',
                'handshape': 'train',
                'senz3d': 'acquisitions'
            }
        )
    # 
    dataloader_sup = DataLoader(
        dataset_sup, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=structured_collate_fn_sup,
        drop_last=True
    )

    # unsupervised dataset with structured augmentation
    dataset_unsup = HandPoseContrastiveDataset(num_samples=num_samples_unsup, npy_file=f'data/kpts/fake/augmented_gesture_groups_{n_aug_pregenerated}.npy')

    dataloader_unsup = DataLoader(dataset_unsup,
                                    batch_size=batch_size,  # Should match grid_size for structured batch
                                    shuffle=True,
                                    # num_workers=4,
                                    pin_memory=False,
                                    collate_fn=structured_collate_fn,
                                    drop_last=True)

    # test dataset
    dataset_eval_sup = LabelledHandDataset(dataset_name='lexset', split='train')
    dataloader_eval_sup = DataLoader(dataset_eval_sup, batch_size=batch_size, shuffle=False)

    dataset_test = LabelledHandDataset(dataset_name='lexset', split='test')
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    # initialize model and optimizer
    # model = HandEncoder(embedding_size=embedding_dim).to(device)
    model = HandEncoderGAT6dof(embedding_size=embedding_dim, do_norm_after_input=False, fn=graph_transform_complex).to(device)
    # model = HandEncoderGCN6dof(embedding_size=embedding_dim, do_norm_after_input=False, fn=graph_transform_complex).to(device)

    # load model from file
    if model_checkpoint_path:
        if os.path.exists(model_checkpoint_path):
            model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
            print(f"Model loaded from {model_checkpoint_path}")
            # You might want to load optimizer state as well for resuming training
            # optimizer.load_state_dict(torch.load(model_checkpoint_path.replace('best.pth', 'optimizer.pth')))
        else:
            print(f"Model file not found at {model_checkpoint_path}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, patience=patience)

    # initialize TensorBoard writer
    os.makedirs(train_path_root, exist_ok=True)
    writer = SummaryWriter(os.path.join(train_path_root,'logs'))
    writer.add_text('Model', f'Model class: {model.__class__.__name__} (train_supcon_gat_curriculum_multi) [dosup={do_sup}, dounsup={do_unsup}] {extra_text}', 0)

    # model save paths
    os.makedirs(train_path_root, exist_ok=True)
    model_save_path_root = os.path.join(train_path_root,'checkpoints')
    os.makedirs(model_save_path_root, exist_ok=True)
    best_model_path = os.path.join(model_save_path_root,'best.pth')
    last_model_path = os.path.join(model_save_path_root,'last.pth')

    # create the models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # initialize best loss tracker
    best_loss = float('inf')

    # training loop
    for param_group in optimizer.param_groups:
        param_group['lr'] = initial_lr

    for epoch in range(start_epoch, start_epoch+num_epochs):
        # =================== Profiling ===================
        if check_profile:
            import cProfile
            import pstats
            # Profile using cProfile
            profiler = cProfile.Profile()
            profiler.enable()
        # =================== Training ===================
        model.train()
        total_train_loss = 0.0
        total_supcon_loss = 0.0
        total_unsup_loss = 0.0
        sup_iter = iter(dataloader_sup)  # Iterator for labelled data
        unsup_iter = iter(dataloader_unsup)  # Iterator for unlabelled data

        dataset_size = 0
        if do_sup:
            dataset_size = len(dataloader_sup)
        if do_unsup:
            dataset_size = max(dataset_size, len(dataloader_unsup))
        dataset_size = min(dataset_size, max_dataset_size//batch_size)
        
        # FULL
        if do_full:
            dataset_size = len(dataloader_sup)
        for i in tqdm(range(dataset_size),
                        desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            loss = 0.0
            loss_supcon = 0.0
            loss_unsup = 0.0

            # supervised batch (if available)
            if do_sup:
                try:
                    labels, joints = next(sup_iter)
                    joints, labels = joints.to(device), labels.to(device)
                    features = model(joints)  # [B, D]
                    features = features.unsqueeze(1)  # [B, 1, D]
                    loss_supcon = supcon_loss(features, labels, device=device)
                    total_supcon_loss += loss_supcon.item()
                    loss = loss_supcon
                except StopIteration:
                    loss = 0.0

            # unsupervised batch (if available)
            if do_unsup:
                try:
                    structured_batch = next(unsup_iter)
                    h, w = batch_size, grid_size
                    n_samples = h * w
                    flat_input = structured_batch.view(n_samples, -1)
                    embeddings = model(flat_input)
                    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
                    similarity_matrix = torch.matmul(embeddings_norm, embeddings_norm.T)
                    unsupervised_labels = torch.arange(h).repeat_interleave(w).to(device)
                    mask = torch.eq(unsupervised_labels.unsqueeze(0), unsupervised_labels.unsqueeze(1))
                    mask = mask.fill_diagonal_(False)
                    loss_unsup = info_nce_loss_from_matrix(similarity_matrix, mask, temperature)
                    total_unsup_loss += loss_unsup.item()
                    loss = loss + loss_unsup if loss_supcon != 0.0 else loss_unsup
                except StopIteration:
                    loss_unsup = 0.0
                    if loss_supcon == 0.0:
                        loss = loss_unsup

            if loss == 0.0:
                continue
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / dataset_size
        avg_supcon_loss = total_supcon_loss / len(dataloader_sup) if len(dataloader_sup) > 0 else 0.0
        avg_unsup_loss = total_unsup_loss / len(dataloader_unsup) if len(dataloader_unsup) > 0 else 0.0

        # log metrics
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/train-supcon', avg_supcon_loss, epoch)
        writer.add_scalar('Loss/train-unsup', avg_unsup_loss, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        # curriculum
        writer.add_scalar('Curriculum/ang-batch', angle_batch_schedule(epoch), epoch)
        writer.add_scalar('Curriculum/ang-aug', angle_aug_schedule(epoch), epoch)
        writer.add_scalar('Curriculum/ang-sup-aug', angle_sup_aug_schedule(epoch), epoch)
        writer.add_scalar('Curriculum/scale-aug-L', scale_range_schedule(epoch)[0], epoch)
        writer.add_scalar('Curriculum/scale-aug-H', scale_range_schedule(epoch)[1], epoch)

        # save models
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with train loss: {best_loss:.4f}")
            # Optionally save optimizer state
            # torch.save(optimizer.state_dict(), best_model_path.replace('best.pth', 'optimizer.pth'))
        torch.save(model.state_dict(), last_model_path)
        # Optionally save optimizer state for the last model
        # torch.save(optimizer.state_dict(), last_model_path.replace('last.pth', 'optimizer.pth'))

        # update scheduler
        if epoch <= lr_jump_epoch:
            if epoch == lr_jump_epoch:
                initial_lr = learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_lr
        else:
            scheduler.step(avg_train_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, SupCon Loss: {avg_supcon_loss:.4f}, Unsup Loss: {avg_unsup_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # evaluation with k-NN
        if (epoch + 1) % eval_interval == 0:
            print(f"Evaluating on test set at epoch {epoch+1} using k-NN (k={k_neighbors})")

            # extract embeddings
            train_embeddings, train_labels = extract_embeddings(model, dataloader_eval_sup, device)
            test_embeddings, test_labels = extract_embeddings(model, dataloader_test, device)

            # perform k-NN evaluation
            accuracy, f1 = evaluate_knn(train_embeddings, train_labels, test_embeddings, test_labels, k=k_neighbors)

            # log evaluation metrics
            writer.add_scalar('Accuracy/test', accuracy, epoch)
            writer.add_scalar('F1/test', f1, epoch)
            print(f"Test Accuracy: {accuracy:.4f}, Test F1: {f1:.4f}")

        # == Profiling ==
        if check_profile:
            profiler.disable()

            # Print results sorted by cumulative time
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            stats.print_stats(30)  # Show top 30 results
            exit(0)

    print(f"Training completed. Model checkpoints saved at {model_save_path_root}.")
    if model_checkpoint_path:
        print(f"This model was loaded from {model_checkpoint_path}.")

    # clean up
    writer.close()