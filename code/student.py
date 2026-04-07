"""
Homework 5 - Vision Transformers and Self-Supervised Learning
CSCI1430 - Computer Vision
Brown University
"""

import os
import copy
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import hyperparameters as hp
from helpers import create_vit_tiny, get_attention_weights, DINODashboard
from hw4_code import train_loop, SceneDataset, CropRotationDataset

BANNER_ID = 2081693 # <- replace with your Banner ID; drop the 'B' prefix and any leading 0s.
torch.manual_seed(BANNER_ID)


# ========================================================================
#  TASK 0: Attention map visualization
#
#  Visualize what ViT attention heads "look at."
#  We extract [class]-to-patch attention from the last transformer layer
#  and display it in two styles: fade-to-black and grayscale heatmaps.
#  This function is reused throughout the homework (Tasks 0, 3, 4).
# ========================================================================

# Part A: Visualize attention maps
#
def visualize_attention(model, image_tensor, save_path, style='fade', device='cpu'):
    """Extract and visualize [class]-to-patch attention from a ViT.

    This function does two things:
      1. Extract attention: call get_attention_weights() to get the raw
         (num_heads, num_tokens, num_tokens) matrix from the last layer,
         then pull out [class]'s attention to each patch and reshape to 2D.
      2. Visualize: display the original image alongside per-head attention maps.

    Two visualization styles:
        'gray'  -- Nearest-neighbor upsample (preserves pixelated patch grid),
                   display as grayscale.  (Caron et al. DINO style)
        'fade'  -- Bilinear upsample to image resolution, multiply image by
                   attention. High-attention areas stay visible; low-attention
                   areas fade to black.  (Dosovitskiy et al. style)

    Rescaling for display: 
    Each head's [class]-to-patch attention is a probability distribution (softmax), 
    so the values sum to 1 across all patches. But, suppose we have a 
    14x14 = 196 patch grid, then a uniform attention gives ~0.005 per patch — 
    nearly black. Even a head that focuses on a single region might peak at 0.1. 
    
    To make the patterns visible, we can rescale each head's attention to [0, 1] 
    via (a - a.min()) / (a.max() - a.min()). This stretches whatever variation 
    exists to fill the display range. Note that this means a nearly-uniform head 
    can look structured — compare the range (max - min) across heads to judge 
    which are truly selective.

    Arguments:
        model        -- a timm ViT model (e.g., from create_vit_tiny())
        image_tensor -- (1, 3, H, W) tensor, values in [0, 1]
        save_path    -- where to save the PNG
        style        -- 'fade' or 'gray'
        device       -- torch device

    Hints:
        - get_attention_weights(model, image_tensor, device) returns shape
          (num_heads, num_tokens, num_tokens). Token 0 is [class].
        - model.num_prefix_tokens tells you how many non-patch tokens (usually 1).
        - For a 224px image with 16px patches: 14x14 = 196 patches.
        - Use F.interpolate to upsample, mode='bilinear' or mode='nearest'.
    """
    # TODO:
    #   Step 1: Extract attention maps
    #   1. Call get_attention_weights(model, image_tensor, device)
    #      -> (num_heads, num_tokens, num_tokens)
    #   2. Extract [class] row: attn[:, 0, num_prefix:]
    #      -> (num_heads, num_patches)
    #   3. Reshape to 2D grid. The grid shape depends on the image size:
    #          H_img, W_img = image_tensor.shape[2], image_tensor.shape[3]
    #          patch_size = 16  # ViT-Tiny uses 16x16 patches
    #          h, w = H_img // patch_size, W_img // patch_size
    #      -> cls_attn.reshape(num_heads, h, w)
    #
    #   Step 2: Build visualization panels
    #
    #   Convert image_tensor[0] to numpy (H, W, 3) for display.
    #
    #   Implement style='gray' first — simpler:
    #   4. For each head:
    #      a. Rescale attention to [0, 1] (see note above).
    #      b. Nearest-neighbor upsample: F.interpolate(..., mode='nearest')
    #      -> produces a list of upsampled attention maps (as numpy arrays)
    #
    #   Then add style='fade':
    #   4. For each head:
    #      a. Rescale attention to [0, 1] (see note above).
    #      b. Bilinear upsample: F.interpolate(..., mode='bilinear')
    #      c. Fade the input image = image * attn_up[:, :, np.newaxis]
    #      -> produces a list of faded images (as numpy arrays)
    #
    #   Step 3: Make figure and save
    #   5. fig, axes = plt.subplots(1, num_heads + 1, ...)
    #      First panel: original image. Remaining panels: attention maps.
    #   6. imshow each panel. For 'gray': cmap='gray', vmin=0, vmax=1.
    #   7. fig.savefig(save_path, ...) then plt.close(fig)
    # Step 1: Extract attention maps
    attn = get_attention_weights(model, image_tensor, device)  # (num_heads, num_tokens, num_tokens)
    num_heads = attn.shape[0]
    num_prefix = model.num_prefix_tokens  # usually 1

    H_img, W_img = image_tensor.shape[2], image_tensor.shape[3]
    patch_size = 16
    h, w = H_img // patch_size, W_img // patch_size

    # Extract [class] token's attention to each patch
    cls_attn = attn[:, 0, num_prefix:]          # (num_heads, num_patches)
    cls_attn = cls_attn.reshape(num_heads, h, w) # (num_heads, h, w)

    # Convert image tensor to numpy for display: (H, W, 3)
    img_np = image_tensor[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    img_np = np.clip(img_np, 0, 1)

    # Step 2: Build visualization panels
    panels = []
    for head_idx in range(num_heads):
        head_attn = cls_attn[head_idx]  # (h, w)

        # Rescale to [0, 1]
        a_min, a_max = head_attn.min(), head_attn.max()
        if a_max > a_min:
            head_attn_norm = (head_attn - a_min) / (a_max - a_min)
        else:
            head_attn_norm = torch.zeros_like(head_attn)

        # Shape for interpolation: (1, 1, h, w)
        attn_4d = head_attn_norm.unsqueeze(0).unsqueeze(0).float()

        if style == 'gray':
            attn_up = F.interpolate(attn_4d, size=(H_img, W_img), mode='nearest')
            attn_up_np = attn_up[0, 0].cpu().numpy()  # (H, W)
            panels.append(attn_up_np)

        elif style == 'fade':
            attn_up = F.interpolate(attn_4d, size=(H_img, W_img), mode='bilinear', align_corners=False)
            attn_up_np = attn_up[0, 0].cpu().numpy()  # (H, W)
            faded = img_np * attn_up_np[:, :, np.newaxis]  # (H, W, 3)
            faded = np.clip(faded, 0, 1)
            panels.append(faded)

    # Step 3: Make figure and save
    fig, axes = plt.subplots(1, num_heads + 1, figsize=(3 * (num_heads + 1), 3))

    # Original image
    axes[0].imshow(img_np)
    axes[0].set_title('Original', fontsize=9)
    axes[0].axis('off')

    for i, panel in enumerate(panels):
        if style == 'gray':
            axes[i + 1].imshow(panel, cmap='gray', vmin=0, vmax=1)
        else:
            axes[i + 1].imshow(panel)
        axes[i + 1].set_title(f'Head {i}', fontsize=9)
        axes[i + 1].axis('off')

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


# ========================================================================
#  TASK 3: Mini-DINO self-supervised pretraining
#
#  This is the main event. We implement DINO — a self-supervised method
#  that trains a ViT to produce semantically meaningful attention maps.
#  See the handout and the DINO diagram for the full picture.
# ========================================================================

# Part A: DINOMultiCropDataset
# Follow the same idea as in your HW4 CropRotationDataset function.
#
class DINOMultiCropDataset(Dataset):
    """Generate multi-crop views of images for DINO training.

    For each image, produce:
        - 2 global crops (large, covering 40-100% of image area)
        - N local crops (small, covering 5-40% of image area)

    The teacher sees only global crops (the big picture).
    The student sees all crops (including small local patches).
    This asymmetry forces the student to infer global semantics from local views.

    Hyperparameters are defined in hp.DINO_*

    Arguments:
        device           -- torch device for GPU operations
        data_dir         -- path to dataset root (images in data_dir/train/)
        global_crop_size -- pixel size of global crops (default: 224)
        local_crop_size  -- pixel size of local crops (default: 96)
        num_local_crops  -- number of local crops per image (default: 6)

    After construction, provides:
        .image_paths     -- list of image file paths
        len(dataset)     -- number of images
    """

    def __init__(self, device, data_dir, global_crop_size=hp.DINO_GLOBAL_CROP_SIZE,
                 local_crop_size=hp.DINO_LOCAL_CROP_SIZE,
                 num_local_crops=hp.DINO_NUM_LOCAL_CROPS):
        # TODO:
        #   1. Load all image paths from data_dir/train/ using ImageFolder.
        #      Store as self.image_paths.
        #   2. Store self.num_local_crops and self.device.
        #   3. Define self.global_transform and self.local_transform
        #      as transforms.Compose pipelines (see __getitem__ for usage).
        # 1. Load all image paths from data_dir/train/
        dataset = ImageFolder(os.path.join(data_dir, 'train'))
        self.image_paths = [s[0] for s in dataset.samples]

        # 2. Store attributes
        self.num_local_crops = num_local_crops
        self.device = device

        # 3. Define transforms
        # Global crops: cover 40-100% of the image
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                global_crop_size,
                scale=(0.4, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=int(global_crop_size * 0.1) | 1, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
        ])

        # Local crops: cover 5-40% of the image
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                local_crop_size,
                scale=(0.05, 0.4),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=int(local_crop_size * 0.1) | 1, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        # TODO
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Return a list of crops: [global_1, global_2, local_1, ..., local_N].

        Each crop is a (3, crop_size, crop_size) tensor in [0, 1].
        """
        # TODO:
        #   1. Load the image at self.image_paths[idx] as a PIL Image (RGB).
        #   2. Generate 2 global crops (large views of the image).
        #   3. Generate num_local_crops local crops (small views).
        #
        #   Hint: global crops should cover most of the image (40-100% area),
        #   local crops should be much smaller (5-40% area).
        #   See: transforms.RandomResizedCrop, transforms.Compose, transforms.ToTensor

        #   4. Optional augmentations (applied before ToTensor in each transform):
        #      horizontal flip, color jitter, random grayscale, gaussian blur.
        #   5. Return the list: [global_1, global_2, local_1, ..., local_N]
        # 1. Load image as PIL RGB
        img = Image.open(self.image_paths[idx]).convert('RGB')

        # 2. Generate 2 global crops
        global_crops = [self.global_transform(img) for _ in range(2)]

        # 3. Generate num_local_crops local crops
        local_crops = [self.local_transform(img) for _ in range(self.num_local_crops)]

        # 5. Return [global_1, global_2, local_1, ..., local_N]
        return global_crops + local_crops



# ========================================================================
# ViTEncoder — a ViT backbone with any head (provided for you)
# ========================================================================
#
class ViTEncoder(nn.Module):
    """ViT backbone with a head on the [class] token.

    Used for classification (Linear head), rotation (Linear head),
    DINO (MLP head), and DINOv3 (by passing encoder= to constructor).

    Arguments:
        head    -- nn.Module to apply to the [class] token embedding
        encoder -- optional external encoder (default: creates ViT-Tiny)

    After construction, provides:
        .encoder     -- the ViT backbone
        .encoder_dim -- embedding dimension (192 for ViT-Tiny, 384 for DINOv3)
        .head        -- the head module
    """

    def __init__(self, head, encoder=None):
        super().__init__()
        if encoder is None:
            self.encoder, self.encoder_dim = create_vit_tiny()
        else:
            self.encoder = encoder
            self.encoder_dim = encoder.embed_dim
        self.head = head

    def forward(self, x):
        tokens = self.encoder.forward_features(x)    # (B, N, D)
        cls_token = tokens[:, 0, :]                  # (B, D)
        return self.head(cls_token)                  # (B, out_dim)


# DINO training loop
#
def t3_dino_pretrain(dino_data, device, approaches):
    """Train a ViT-Tiny encoder with mini-DINO self-supervised learning.

    See the DINO diagram in the handout for the full architecture.

    Hyperparameters are defined in hp.DINO_*
    """
    # Reproducible initialization — do not remove
    torch.manual_seed(BANNER_ID)
    results_dir = os.path.dirname(approaches['dino'].weights)

    # --- Dashboard setup (provided, do not modify) -------------------------
    _pil = Image.open(dino_data.image_paths[0]).convert('RGB')
    sample_img = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
    ])(_pil).unsqueeze(0)
    dashboard = DINODashboard(save_dir=results_dir, sample_image=sample_img,
                              device=device)
    # -----------------------------------------------------------------------

    # TODO: Implement the DINO training loop. See the handout diagram.
    #
    #   Setup:
    #   1. Create student ViTEncoder with a 3-layer MLP projection head:
    #      e.g., 
    #      Linear(192, hidden) -> ReLU -> Linear(hidden, hidden) -> ReLU -> Linear(hidden, K)
    #      where hidden = hp.DINO_HIDDEN_DIM and K = hp.DINO_OUT_DIM.
    #
    #      - Using an MLP is critical — a single Linear head causes immediate collapse.
    #      - The DINO paper uses GeLU activations (a smooth fancier ReLU)
    #      - Choice of K is important -> we set it at 256 but smaller may give better results
    #        Training DINO on large data uses much larger K, e.g., 65,556
    #
    #   2. Create teacher as a frozen deep copy of the student (no gradients).
    #
    #   3. Create an optimizer.
    #
    #   4. Create a DataLoader for the multi-crop data.
    #      Normally, DataLoader's default collate stacks all samples into a
    #      single tensor via torch.stack. But here each sample is a list of
    #      crops with different sizes (224x224 globals + 96x96 locals), so
    #      stacking fails. However, the collate_fn input to DataLoader can 
    #      receive a list of samples and return a batch; so, we can use it:
    #      passing collate_fn=list makes it return the samples as-is 
    #      (list in, list out), skipping the stacking step.
    #
    #   Training loop (for each epoch, for each batch):
    #   5. Forward the teacher on global crops only (first 2), with no gradients.
    #      Forward the student on all crops (global + local).
    #
    #   6. DINO loss: for each cross-view pair (teacher crop i, student crop j,
    #      skipping i == j), compute cross-entropy where the target distribution
    #      is the teacher's sharpened softmax (divided by teacher_temp)
    #      and the predicted distribution is the student's sharpened softmax
    #      (divided by student_temp — less sharp than the teacher's).
    #      Average over all valid pairs.
    #      Remember: there is no gradient back from the loss for the teacher, so
    #      .detatch() on the teacher's softmax to prevent backprop through it.
    #
    #   7. Backprop and optimizer step.
    #
    #   8. EMA update: for each parameter pair,
    #      teacher = momentum * teacher + (1 - momentum) * student
    #
    #   After each epoch:
    #   9. Print average loss and call the dashboard:
    #       dashboard.update(epoch, avg_loss, student_out[0].detach(),
    #                        teacher_out[0].detach(),
    #                        center=torch.zeros(hp.DINO_OUT_DIM),
    #                        encoder=student.encoder,
    #                        ema_momentum=hp.DINO_EMA_MOMENTUM)
    #
    #   After training:
    #  10. Save encoder weights to approaches['dino'].weights
    #  11. Save loss curve to approaches['dino'].curve_train
    #  12. Call dashboard.save_attention_evolution()
    #  13. Visualize final attention maps using your visualize_attention():
    #          results_dir/attention_maps_fade.png      (style='fade')
    #          results_dir/attention_maps_grayscale.png (style='gray')

    # 1. Build student: ViTEncoder with 3-layer MLP head
    hidden = hp.DINO_HIDDEN_DIM
    K = hp.DINO_OUT_DIM
    encoder_dim = 192  # ViT-Tiny embedding dim

    student_head = nn.Sequential(
        nn.Linear(encoder_dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Linear(hidden, K),
    )
    student = ViTEncoder(head=student_head).to(device)

    # 2. Teacher = frozen deep copy of student
    teacher = copy.deepcopy(student).to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)

    # 3. Optimizer
    optimizer = torch.optim.Adam(student.parameters(), lr=hp.DINO_LR)

    # 4. DataLoader with list collate_fn
    loader = DataLoader(
        dino_data,
        batch_size=hp.DINO_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=list,
    )

    loss_history = []

    for epoch in range(hp.DINO_EPOCHS):
        student.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            # batch is a list of samples; each sample is a list of crop tensors
            # Transpose: batch[sample_idx][crop_idx] -> crops[crop_idx][sample_idx]
            num_crops = len(batch[0])
            crops = []
            for c in range(num_crops):
                crop_batch = torch.stack([batch[s][c] for s in range(len(batch))]).to(device)
                crops.append(crop_batch)

            global_crops = crops[:2]
            all_crops = crops  # 2 global + num_local

            # 5. Forward teacher on global crops only (no grad)
            with torch.no_grad():
                teacher_out = [teacher(g) for g in global_crops]  # list of (B, K)

            # Forward student on all crops
            student_out = [student(c) for c in all_crops]  # list of (B, K)

            # 6. DINO loss: cross-view cross-entropy
            teacher_temp = hp.DINO_TEACHER_TEMP
            student_temp = hp.DINO_STUDENT_TEMP

            loss = torch.tensor(0.0, device=device)
            n_pairs = 0
            for i, t_logit in enumerate(teacher_out):
                # Teacher sharpened softmax (detached)
                t_prob = F.softmax(t_logit / teacher_temp, dim=-1).detach()
                for j, s_logit in enumerate(student_out):
                    if i == j:
                        continue
                    # Student log-softmax (less sharp than teacher)
                    s_log_prob = F.log_softmax(s_logit / student_temp, dim=-1)
                    # Cross-entropy H(t_prob, s_log_prob)
                    loss += -(t_prob * s_log_prob).sum(dim=-1).mean()
                    n_pairs += 1

            loss = loss / n_pairs

            # 7. Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 8. EMA update of teacher
            momentum = hp.DINO_EMA_MOMENTUM
            with torch.no_grad():
                for s_param, t_param in zip(student.parameters(), teacher.parameters()):
                    t_param.data.mul_(momentum).add_((1.0 - momentum) * s_param.data)

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{hp.DINO_EPOCHS}]  Loss: {avg_loss:.4f}")

        # 9. Update dashboard
        dashboard.update(
            epoch, avg_loss,
            student_out[0].detach(),
            teacher_out[0].detach(),
            center=torch.zeros(hp.DINO_OUT_DIM),
            encoder=student.encoder,
            ema_momentum=hp.DINO_EMA_MOMENTUM,
        )

    # 10. Save encoder weights
    os.makedirs(results_dir, exist_ok=True)
    torch.save(student.encoder.state_dict(), approaches['dino'].weights)

    # 11. Save loss curve
    plt.figure()
    plt.plot(range(1, hp.DINO_EPOCHS + 1), loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('DINO Training Loss')
    plt.tight_layout()
    plt.savefig(approaches['dino'].curve_train, dpi=150)
    plt.close()

    # 12. Save attention evolution
    dashboard.save_attention_evolution()

    # 13. Final attention maps
    student.eval()
    with torch.no_grad():
        visualize_attention(
            student, sample_img.to(device),
            save_path=os.path.join(results_dir, 'attention_maps_fade.png'),
            style='fade', device=device,
        )
        visualize_attention(
            student, sample_img.to(device),
            save_path=os.path.join(results_dir, 'attention_maps_grayscale.png'),
            style='gray', device=device,
        )