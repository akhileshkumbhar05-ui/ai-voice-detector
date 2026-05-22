
# import torch
# import numpy as np
# import torch.nn.functional as F

# class SpectrogramGradCAM:
#     """Minimal Grad-CAM for a conv net that ends with global pooling + linear."""
#     def __init__(self, model, target_layer_name="features.6"):
#         self.model = model
#         self.gradients = None
#         self.activations = None
#         layer = dict([*model.named_modules()])[target_layer_name]
#         layer.register_forward_hook(self._forward_hook)
#         layer.register_full_backward_hook(self._backward_hook)

#     def _forward_hook(self, module, inp, out):
#         self.activations = out.detach()

#     def _backward_hook(self, module, grad_input, grad_output):
#         self.gradients = grad_output[0].detach()

#     def __call__(self, x, class_idx=None):
#         self.model.zero_grad()
#         logits = self.model(x)
#         if class_idx is None:
#             class_idx = logits.argmax(dim=1)
#         loss = logits[:, class_idx].sum()
#         loss.backward()
#         weights = self.gradients.mean(dim=(2, 3), keepdim=True)
#         cam = (weights * self.activations).sum(dim=1, keepdim=True)
#         cam = torch.relu(cam)
#         B, _, H, W = cam.shape
#         cam = cam.view(B, H, W)
#         cam = cam - cam.min()
#         if cam.max() > 0:
#             cam = cam / cam.max()
#         return cam.cpu().numpy(), logits.detach().cpu().numpy()


#----------------------------------------------------------------------------

# import torch
# import torch.nn.functional as F

# class SpectrogramGradCAM:
#     """Minimal Grad-CAM for conv nets ending with global pooling + linear."""
#     def __init__(self, model, target_layer_name: str = "features.6"):
#         self.model = model
#         self.gradients = None
#         self.activations = None

#         # grab the target layer
#         target_layer = dict(model.named_modules())[target_layer_name]
#         # register hooks
#         self._fh = target_layer.register_forward_hook(self._forward_hook)
#         self._bh = target_layer.register_full_backward_hook(self._backward_hook)

#     def _forward_hook(self, module, inp, out):
#         # keep on device; don’t detach (we only read it)
#         self.activations = out

#     def _backward_hook(self, module, grad_input, grad_output):
#         # grad w.r.t. the layer output
#         self.gradients = grad_output[0]

#     def __call__(self, x, class_idx=None):
#         """
#         x: Tensor [B,1,H,W] with requires_grad=True (set by caller).
#         Returns: (cam: np.ndarray [B,H,W], logits: np.ndarray [B,num_classes])
#         """
#         assert self.activations is None or self.activations.device == x.device

#         self.model.zero_grad(set_to_none=True)
#         logits = self.model(x)  # forward WITH grads

#         if class_idx is None:
#             class_idx = logits.argmax(dim=1)                # [B]
#         if isinstance(class_idx, int):
#             selected = logits[:, class_idx]                 # [B]
#         else:
#             selected = logits.gather(1, class_idx.view(-1, 1)).squeeze(1)  # [B]

#         # backprop from selected class score
#         self.model.zero_grad(set_to_none=True)
#         selected.sum().backward(retain_graph=True)

#         # weights: global-average of gradients over spatial dims
#         # shapes: gradients/activations -> [B, C, H, W]
#         weights = self.gradients.mean(dim=(2, 3), keepdim=True)
#         cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [B,1,H,W]
#         cam = F.relu(cam)

#         # normalize per-sample to [0,1]
#         B, _, H, W = cam.shape
#         cam = cam.view(B, H, W)
#         cam_min = cam.view(B, -1).min(dim=1, keepdim=True)[0].view(B, 1, 1)
#         cam_max = cam.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1)
#         cam = (cam - cam_min) / (cam_max - cam_min + 1e-6)

#         return cam.detach().cpu().numpy(), logits.detach().cpu().numpy()

#------------------------------------------------------------------------

import torch
import torch.nn.functional as F

class SpectrogramGradCAM:
    """Minimal Grad-CAM for conv nets ending with global pooling + linear."""
    def __init__(self, model, target_layer_name: str = "features.6"):
        self.model = model
        self.gradients = None
        self.activations = None

        # Get the target conv layer by name (e.g., "features.6")
        target_layer = dict(model.named_modules())[target_layer_name]

        # Register hooks
        self._fh = target_layer.register_forward_hook(self._forward_hook)
        self._bh = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        # Keep on-device tensors; we’ll only read them
        self.activations = out

    def _backward_hook(self, module, grad_input, grad_output):
        # Gradient w.r.t. the layer output
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        """
        x: Tensor [B,1,H,W] with requires_grad=True.
        Returns:
          cam: np.ndarray [B,H,W] in [0,1]
          logits: np.ndarray [B,num_classes]
        """
        assert x.requires_grad, "Input to Grad-CAM must require grad"

        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)  # forward WITH grads

        # Choose class per sample
        if class_idx is None:
            class_idx = logits.argmax(dim=1)  # [B]
        if isinstance(class_idx, int):
            selected = logits[:, class_idx]   # [B]
        else:
            selected = logits.gather(1, class_idx.view(-1, 1)).squeeze(1)  # [B]

        # Backprop from selected class score
        self.model.zero_grad(set_to_none=True)
        selected.sum().backward(retain_graph=True)

        # Build CAM from gradients & activations: [B,C,H,W]
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # GAP over H,W
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [B,1,H,W]
        cam = F.relu(cam)

        # Normalize per-sample to [0,1]
        B, _, H, W = cam.shape
        cam = cam.view(B, H, W)
        cam_min = cam.view(B, -1).min(dim=1, keepdim=True)[0].view(B, 1, 1)
        cam_max = cam.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-6)

        return cam.detach().cpu().numpy(), logits.detach().cpu().numpy()