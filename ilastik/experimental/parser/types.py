###############################################################################
#   ilastik: interactive learning and segmentation toolkit
#
#       Copyright (C) 2011-2024, the ilastik developers
#                                <team@ilastik.org>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# In addition, as a special exception, the copyright holders of
# ilastik give you permission to combine ilastik with applets,
# workflows and plugins which are not covered under the GNU
# General Public License.
#
# See the LICENSE file for details. License information is also available
# on the ilastik web site at:
#          http://ilastik.org/license.html
###############################################################################
# pyright: strict
from collections import OrderedDict
from pathlib import Path
from typing import Annotated, Dict, List, Literal, Optional, Tuple, Type

import annotated_types
import numpy
import numpy.typing as npt
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, StringConstraints

from ilastik.applets.base.appletSerializer.legacyClassifiers import (
    deserialize_classifier,
    deserialize_classifier_factory,
)
from ilastik.applets.base.appletSerializer.serializerUtils import deserialize_string_from_h5
from ilastik.experimental.parser._h5helpers import (
    deserialize_arraylike_from_h5,
    deserialize_axistags_from_h5,
    deserialize_str_list_from_h5,
)
from lazyflow.classifiers import LazyflowVectorwiseClassifierABC, LazyflowVectorwiseClassifierFactoryABC

NDShape = Annotated[Tuple[int, ...], annotated_types.Len(2, 6)]
LaneName = Annotated[str, StringConstraints(pattern=r"lane\d{4}")]


class VigraAxisTags(BaseModel):
    key: str
    typeFlags: int
    resolution: int
    description: str


class DatasetInfo(BaseModel):
    axistags: Annotated[List[VigraAxisTags], BeforeValidator(deserialize_axistags_from_h5)]
    shape: Annotated[
        NDShape, BeforeValidator(lambda x: tuple(x.tolist())), BeforeValidator(deserialize_arraylike_from_h5)
    ]


class InputData(BaseModel):
    role_names: Annotated[List[str], BeforeValidator(deserialize_str_list_from_h5)] = Field(alias="Role Names")
    infos: OrderedDict[
        LaneName, Dict[str, Annotated[Optional[DatasetInfo], BeforeValidator(lambda x: None if len(x) == 0 else x)]]
    ]

    @property
    def _last_lane(self):
        last_lane_name = next(reversed(self.infos))
        return self.infos[last_lane_name]

    @property
    def axis_order(self) -> str:
        """Returns the axis keys of the last lane as a string"""
        last_lane = self._last_lane
        base_role_name = self.role_names[0]  # should usually be 'Raw Data'
        base_info = last_lane[base_role_name]
        assert base_info
        return "".join(ax.key for ax in base_info.axistags)

    @property
    def num_channels(self) -> int:
        """Returns the number for channels in the last lane"""
        last_lane = self._last_lane
        base_role_name = self.role_names[0]  # should usually be 'Raw Data'
        base_info = last_lane[base_role_name]
        assert base_info
        n_channels = 1
        for ind, at in enumerate(base_info.axistags):
            if at.key == "c":
                n_channels = base_info.shape[ind]
                break
        return n_channels

    @property
    def spatial_axes(self):
        return "".join(ax for ax in self.axis_order if ax in "xyz")


class FeatureMatrix(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    names: Annotated[List[str], BeforeValidator(deserialize_arraylike_from_h5)] = Field(alias="FeatureIds")
    scales: Annotated[List[float], BeforeValidator(deserialize_arraylike_from_h5)] = Field(alias="Scales")
    selections: Annotated[npt.NDArray[numpy.bool_], BeforeValidator(deserialize_arraylike_from_h5)] = Field(
        alias="SelectionMatrix"
    )
    compute_in_2d: Annotated[npt.NDArray[numpy.bool_], BeforeValidator(deserialize_arraylike_from_h5)] = Field(
        alias="ComputeIn2d"
    )


class Classifier(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    classifier_factory: Annotated[
        LazyflowVectorwiseClassifierFactoryABC, BeforeValidator(deserialize_classifier_factory)
    ] = Field(alias="ClassifierFactory")
    classifier: Annotated[
        LazyflowVectorwiseClassifierABC,
        BeforeValidator(deserialize_classifier),
    ] = Field(alias="ClassifierForests")
    label_names: List[str] = Field(alias="LabelNames")

    @property
    def label_count(self) -> int:
        return len(self.label_names)
