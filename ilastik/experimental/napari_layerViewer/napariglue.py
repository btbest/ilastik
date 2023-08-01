import numpy as np
from napari.components import ViewerModel as NapariViewerModel
from napari.layers import Image

from lazyflow.slot import Slot


class NapariImageAdapter(Image):
    """
    A virtual layer that provides ilastik with the API it expects from a viewer layer.
    In particular, ilastik assumes that viewer layers can lazy-load image data from slots it gives them.
    Requests image data and creates actual Napari image layers as necessary when the layer is made visible,
    as Napari images expect their data to be passed in at construction time.
    """

    def __init__(self, viewer: NapariViewerModel, slot: Slot, name: str, opacity: float, visible: bool):
        super().__init__(np.array([[0]]))
        self.viewer = viewer
        self.slot = slot
        self.name = name or slot.name
        self.opacity = opacity
        self.visible = visible
        self.napari_image: Image = None

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, value: bool):
        self._visible = value
        if value:
            self._create_napari_image_and_add_to_viewer()
        else:
            self.napari_image = None

    def _create_napari_image_and_add_to_viewer(self):
        c_index = self.slot.meta.axistags.index("c")
        image_data = self.slot.value
        self.napari_image = self.viewer.add_image(image_data, opacity=self.opacity, channel_axis=c_index)[0]
        self.napari_image.name = self.name
