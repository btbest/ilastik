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
from collections import OrderedDict

from lazyflow.graph import Operator, InputSlot, OutputSlot


class opResize(Operator):
    RawImage = InputSlot()
    TargetShape = InputSlot()
    ResizedImage = OutputSlot()

    def setupOutputs(self):
        assert len(self.RawImage.meta.shape) == len(
            self.TargetShape.value
        ), f"Input image ({self.RawImage.meta.shape}) and target shape ({self.TargetShape.value}) must have same axes"
        self.ResizedImage.meta.assignFrom(self.RawImage.meta)
        self.ResizedImage.meta.shape = self.TargetShape.value

    def execute(self, slot, subindex, roi, result):
        assert slot is self.ResizedImage, "Unknown output slot"
        axes = self.RawImage.meta.getAxisKeys()
        shape_in = self.RawImage.meta.getTaggedShape()
        shape_out = self.ResizedImage.meta.getTaggedShape()
        scaling_factors = OrderedDict([(a, shape_out[a] / shape_in[a]) for a in axes])
        required_halo = (0 for _ in axes)  # TODO: Compute halo based on scaling factors
        # roi_in = enlargeRoiForHalo(roi.start, roi.stop, shape_in.values(), sigma, window=3.5, enlarge_axes=None, return_result_roi=True)
