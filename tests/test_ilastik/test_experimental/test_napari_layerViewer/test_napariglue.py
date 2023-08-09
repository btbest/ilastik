from unittest.mock import MagicMock, call, Mock

import dask.array
import numpy as np

from ilastik.experimental.napari_layerViewer.napariglue import DaskSlotAdapter
from lazyflow.rtype import SubRegion


class TestDaskSlotAdapter:
    def test_does_not_request_from_slot_until_dask_compute_triggered(self):
        def call_get_with_expected_subregion_for_roi(roi):
            if roi == null_slicing:
                return slot.get(null_region)
            elif roi == all_slicing:
                return slot.get(all_region)
            else:
                raise AssertionError(f"DaskAdapterSlot sliced with unexpected ROI: {roi}")

        slot = MagicMock()
        slot.__getitem__.side_effect = call_get_with_expected_subregion_for_roi
        slot.meta.dtype = np.uint8
        slot.meta.shape = (1, 1, 1)
        slot.get = Mock()
        null_slicing = (slice(0, 0, None), slice(0, 0, None), slice(0, 0, None))
        all_slicing = (slice(0, 1, None), slice(0, 1, None), slice(0, 1, None))
        null_region = SubRegion(slot, pslice=null_slicing)
        all_region = SubRegion(slot, pslice=all_slicing)
        dask_array = dask.array.from_array(DaskSlotAdapter(slot), chunks=1)

        # When
        sliced = dask_array[all_slicing]  # Slicing should not call slot.get

        # Then
        slot.get.assert_called_once_with(null_region)  # from_array apparently does this
        assert call(all_region) not in slot.get.mock_calls

        # When
        np.asarray(sliced)  # Triggers dask compute and hence slot.get

        # Then
        assert slot.get.call_args_list == [call(null_region), call(all_region)]
