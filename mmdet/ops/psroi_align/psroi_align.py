from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
from torchvision.ops import ps_roi_align
class PsRoIAlign(nn.Module):

    def __init__(self,
                 out_size,
                 spatial_scale,
                 sample_num=0,
                 aligned=False):
        """
        Args:
            out_size (tuple): h, w
            spatial_scale (float): scale the input boxes by this number
            sample_num (int): number of inputs samples to take for each
                output sample. 2 to take samples densely for current models.
            use_torchvision (bool): whether to use roi_align from torchvision
            aligned (bool): if False, use the legacy implementation in
                MMDetection. If True, align the results more perfectly.

        Note:
            The implementation of RoIAlign when aligned=True is modified from
            https://github.com/facebookresearch/detectron2/

            The meaning of aligned=True:

            Given a continuous coordinate c, its two neighboring pixel
            indices (in our pixel model) are computed by floor(c - 0.5) and
            ceil(c - 0.5). For example, c=1.3 has pixel neighbors with discrete
            indices [0] and [1] (which are sampled from the underlying signal
            at continuous coordinates 0.5 and 1.5). But the original roi_align
            (aligned=False) does not subtract the 0.5 when computing
            neighboring pixel indices and therefore it uses pixels with a
            slightly incorrect alignment (relative to our pixel model) when
            performing bilinear interpolation.

            With `aligned=True`,
            we first appropriately scale the ROI and then shift it by -0.5
            prior to calling roi_align. This produces the correct neighbors;

            The difference does not make a difference to the model's
            performance if ROIAlign is used together with conv layers.
        """
        super(PsRoIAlign, self).__init__()
        self.out_size = _pair(out_size)
        # self.out_size = int(out_size)
        self.spatial_scale = float(spatial_scale)
        self.aligned = aligned
        self.sample_num = int(sample_num)


    def forward(self, features, rois):
        """
        Args:
            features: NCHW images
            rois: Bx5 boxes. First column is the index into N. The other 4
            columns are xyxy.
        """
        assert rois.dim() == 2 and rois.size(1) == 5

        return ps_roi_align(features, rois, self.out_size,
                                self.spatial_scale, self.sample_num)


    def __repr__(self):
        format_str = self.__class__.__name__
        format_str += '(out_size={}, spatial_scale={}, sample_num={}'.format(
            self.out_size, self.spatial_scale, self.sample_num)
        format_str += ', aligned={})'.format(
             self.aligned)
        return format_str
