###############################################################################
#   lazyflow: data flow based lazy parallel computation framework
#
#       Copyright (C) 2011-2025, the ilastik developers
#                                <team@ilastik.org>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the Lesser GNU General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# See the files LICENSE.lgpl2 and LICENSE.lgpl3 for full text of the
# GNU Lesser General Public License version 2.1 and 3 respectively.
# This information is also available on the ilastik web site at:
#          http://ilastik.org/license/
###############################################################################
import logging

import numpy as np
from skimage import transform as sk_transform

from lazyflow.graph import Operator, InputSlot, OutputSlot
from lazyflow.roi import enlargeRoiForHalo, roiToSlice

logger = logging.getLogger(__name__)


class opResize(Operator):
    """
    Resizes input image to desired output size.
    Cannot resize along channel axis (would be nonsense).
    Time is treated like space axes, so resize along t at your own risk.

    """

    RawImage = InputSlot()
    TargetShape = InputSlot()  # Tuple[int, ...]
    ResizedImage = OutputSlot()

    def setupOutputs(self):
        input_axiskeys = self.RawImage.meta.getAxisKeys()
        assert len(input_axiskeys) == len(
            self.TargetShape.value
        ), f"Input image ({self.RawImage.meta.shape}) and target shape ({self.TargetShape.value}) must have same axes"
        assert (
            "c" not in input_axiskeys
            or self.TargetShape.value[input_axiskeys.index("c")] == self.RawImage.meta.getTaggedShape["c"]
        ), "Cannot resize along channel axis"
        if (
            "t" in input_axiskeys
            and self.TargetShape.value[input_axiskeys.index("t")] != self.RawImage.meta.getTaggedShape["t"]
        ):
            logger.warning("Resizing along time axis. Are you sure this is what you want?")
        self.ResizedImage.meta.assignFrom(self.RawImage.meta)
        self.ResizedImage.meta.shape = self.TargetShape.value

    def execute(self, slot, subindex, roi, result):
        assert slot is self.ResizedImage, "Unknown output slot"
        shape_in = self.RawImage.meta.shape
        shape_out = self.ResizedImage.meta.shape
        # @haesleinhuepf sigmas for antialiasing (cf BioImageAnalysisNotebooks downscaling and denoising)
        # scaling_factors = np.divide(shape_out, shape_in)
        # sigmas = (1 / scale / 4 for scale in scaling_factors)
        # skimage sigmas for antialiasing (cf skimage.transform.resize)
        downsampling_factors = np.divide(shape_in, shape_out)
        sigmas = np.maximum(0, (downsampling_factors - 1) / 2)
        axes_to_enlarge = downsampling_factors > 1
        roi_scaled_shape = np.divide(roi.stop - roi.start, downsampling_factors)
        roi_with_halo, result_roi = enlargeRoiForHalo(
            roi.start, roi.stop, shape_in.values(), sigma=sigmas, enlarge_axes=axes_to_enlarge, return_result_roi=True
        )
        raw_image = self.RawImage[roiToSlice(*roi_with_halo)].wait()
        scaled_image = sk_transform.resize(
            raw_image,
            roi_scaled_shape,
            anti_aliasing=True,
            anti_aliasing_sigma=sigmas,
        )
        result[...] = scaled_image[roiToSlice(*result_roi)]
