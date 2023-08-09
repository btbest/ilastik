from typing import Optional

import dask.array as daskarray
from napari.components import ViewerModel as NapariViewerModel
from napari.layers import Image as NapariImageLayer
from volumina.pixelpipeline.datasources import createDataSource
from volumina.layer import GrayscaleLayer as VoluminaGrayscaleLayer

from lazyflow.slot import Slot


class DaskSlotAdapter:
    """
    Dask array that uses a lazyflow slot as its data source to provide lazy-loading.
    Slots lazy-load by returning Requests when sliced, which the consumer needs to .wait() for to load.
    Napari doesn't know to do that of course, but it does know how to lazy-load from dask arrays.
    So we let Napari lazy-load through dask for us and turn off lazy-loading in the slot by awaiting the Request.
    """

    def __init__(self, slot: Slot):
        self.slot = slot
        self.shape = slot.meta.shape
        self.ndim = len(self.shape)
        self.dtype = slot.meta.dtype

    def __getitem__(self, item):
        return self.slot[item].wait()

    def __getattr__(self, item):
        return getattr(self.slot, item)


class NapariImageAdapter(VoluminaGrayscaleLayer):
    """
    A virtual layer that provides ilastik with the API it expects from a viewer layer.
    In particular, ilastik assumes that viewer layers can lazy-load image data from slots it gives them.
    Requests image data and creates actual Napari image layers as necessary when the layer is made visible,
    as Napari images expect their data to be passed in at construction time.
    """

    def __init__(self, viewer: NapariViewerModel, slot: Slot, name: str, opacity: float, visible: bool):
        super().__init__(createDataSource(slot))
        self.napari_image: Optional[NapariImageLayer] = None
        self.viewer = viewer
        self.slot = slot
        self.name = name or slot.name
        self.opacity = opacity
        self.visible = visible

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, value: bool):
        self._visible = value
        if value:
            self._create_napari_image_and_add_to_viewer()
        elif self.napari_image is not None:
            self._destroy_napari_image()

    def _create_napari_image_and_add_to_viewer(self):
        c_index = self.slot.meta.axistags.index("c")
        image_data = daskarray.from_array(DaskSlotAdapter(self.slot), chunks=256)
        self.napari_image = self.viewer.add_image(image_data, opacity=self.opacity, channel_axis=c_index)[0]
        self.napari_image.name = self.name

    def _destroy_napari_image(self):
        self.viewer.layers.remove(self.napari_image)
        self.napari_image = None
