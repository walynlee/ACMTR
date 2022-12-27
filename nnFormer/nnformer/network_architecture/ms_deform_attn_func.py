from operator import imod
from torch.autograd import Function

# import MultiScaleDeformableAttention as MSDA
import torch.nn.functional as F
import torch
from torch.autograd.function import once_differentiable

# class MSDeformAttnFunction(Function):
#     @staticmethod
#     def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
#         ctx.im2col_step = im2col_step
#         output = MSDA.ms_deform_attn_forward(
#             value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
#         ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
#         return output

#     @staticmethod
#     @once_differentiable
#     def backward(ctx, grad_output):
#         value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
#         grad_value, grad_sampling_loc, grad_attn_weight = \
#             MSDA.ms_deform_attn_backward(
#                 value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

#         return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None

#单尺度deformAttn
def deform_attn_core_pytorch_3D(value, value_spatial_shapes, sampling_locations, attention_weights):
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_,  P_, _ = sampling_locations.shape

    sampling_grids = 2 * sampling_locations - 1
    # sampling_grids = 3 * sampling_locations - 1
  
    T_, H_, W_ =  value_spatial_shapes
    value_l_ = value.flatten(2).transpose(1, 2).reshape(N_*M_, D_, T_, H_, W_)
    sampling_grid_l_ = sampling_grids[:, :, :].transpose(1, 2).flatten(0, 1)[:,None,:,:,:]
    
    sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_.to(dtype=value_l_.dtype), mode='bilinear', padding_mode='zeros', align_corners=False)[:,:,0]
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, P_)

    output = (sampling_value_l_ * attention_weights).sum(-1)    #attention_weights：只对q进行回归
    output =output.view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()