import numpy as np

class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame =  self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame 
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames,1)).flatten()
        return mask 
    
class FrameMaskingGenerator:
    def __init__(self, input_size, mask_ratio, mask_list=None):
        """
        Full-frame masking generator.
        
        Args:
            input_size (tuple): (frames, height, width).
            mask_ratio (float): Probability of masking each frame (0 to 1).
            mask_list (list or None): If provided, must be a list of length
                                      `frames` with 0 or 1 indicating
                                      keep (0) or mask (1).
        """
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.mask_ratio = mask_ratio
        
        # Optional user-specified frame mask, e.g. [0, 1, 0, 1, ...]
        self.mask_list = mask_list
        if self.mask_list is not None:
            assert len(self.mask_list) == self.frames, (
                f"mask_list must have length {self.frames}, "
                f"but got {len(self.mask_list)}."
            )

    def __repr__(self):
        info = (
            f"FrameMaskingGenerator("
            f"frames={self.frames}, height={self.height}, width={self.width}, "
            f"mask_ratio={self.mask_ratio}, "
        )
        if self.mask_list is not None:
            masked_frames = sum(self.mask_list)
            info += f"user_masked_frames={masked_frames})"
        else:
            info += "random_masking=True)"
        return info

    def __call__(self):
        """
        Returns a 1D mask [frames * height * width], but ensures
        exactly N = round(mask_ratio * frames) frames are masked.
        """
        frames_to_mask = int(round(self.mask_ratio * self.frames))
        
        # For each sample in the batch, the code typically calls the mask generator once
        # But if your code calls it once per batch, you can do:
        frame_mask = np.zeros(self.frames, dtype=int)
        # Randomly choose 'frames_to_mask' distinct indices
        chosen_indices = np.random.choice(self.frames, frames_to_mask, replace=False)
        frame_mask[chosen_indices] = 1

        # Expand each frame's 0/1 to cover all patches
        mask = np.repeat(frame_mask, self.height * self.width)
        return mask
