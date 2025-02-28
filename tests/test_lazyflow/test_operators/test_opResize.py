from unittest import mock

import numpy
import vigra
from skimage.transform import resize as sk_resize

from lazyflow.operators.opResize import OpResize
from lazyflow.operators.opSplitRequestsBlockwise import OpSplitRequestsBlockwise


def test_resize_matches_skimage(graph):
    arr = numpy.random.randint(0, 256, (10, 10, 10), dtype="uint8")
    data = vigra.taggedView(arr, "zyx")

    op = OpResize(graph=graph)
    op.RawImage.setValue(data)
    op.TargetShape.setValue((5, 5, 5))

    op_resized = op.ResizedImage[:].wait()
    sk_resized = sk_resize(data, (5, 5, 5), anti_aliasing=True, preserve_range=True).astype(numpy.uint8)

    numpy.testing.assert_allclose(op_resized, sk_resized)


def test_resize_handles_blocks(graph):
    # arr = numpy.random.randint(0, 256, (50, 50), dtype="uint8")
    # arr = numpy.indices((50, 50)).sum(0)
    import tifffile

    # filename = "idr-13457227-s4"
    # arr = tifffile.imread(f"C:/Users/root/Code/ilastik-group/sample-data/idr-13457227-s4.tif")
    # data = vigra.taggedView(arr[0,0,2,:,:], "yx")
    # filename = "grid-128x128"
    # arr = tifffile.imread(f"C:/Users/root/Code/ilastik-group/sample-data/grid-128x128.tif")
    # data = vigra.taggedView(arr[:,:,0], "yx")
    filename = "indices-128x128"
    arr = numpy.indices((128, 128)).sum(0)
    data = vigra.taggedView(arr, "yx")
    scaling_target_shape = (50, 50)
    scaling_block_shape = (10, 20)

    op = OpResize(graph=graph)
    op.RawImage.setValue(data)
    op.TargetShape.setValue(scaling_target_shape)
    for halo_factor in numpy.arange(2.5, 9.0, 0.1):
        halo_factor = round(halo_factor, 5)
        op.HaloFactor.setValue(halo_factor)

        split_op = OpSplitRequestsBlockwise(True, graph=graph)
        split_op.Input.connect(op.ResizedImage)
        split_op.BlockShape.setValue(scaling_block_shape)  # Tiny blocks to exacerbate rounding and halo errors

        with mock.patch.object(OpResize, "execute", wraps=op.execute) as mock_execute:
            op_resized = split_op.Output[:].wait().astype(numpy.float64)
            # assert mock_execute.call_count == 16, "splitting into 16 blocks not working"

        sk_resized = sk_resize(data, scaling_target_shape, anti_aliasing=True, preserve_range=True)
        diff = op_resized - sk_resized
        tifffile.imwrite(f"C:/Users/root/Code/ilastik-group/sample-data/{filename}-scipy-1orig.tif", data)
        tifffile.imwrite(f"C:/Users/root/Code/ilastik-group/sample-data/{filename}-scipy-2skscaled.tif", sk_resized)
        tifffile.imwrite(f"C:/Users/root/Code/ilastik-group/sample-data/{filename}-scipy-3opscaled.tif", op_resized)
        tifffile.imwrite(f"C:/Users/root/Code/ilastik-group/sample-data/{filename}-scipy-4diff-{halo_factor}.tif", diff)
        print(
            f"{halo_factor}: halo sigma, {round(0.78*halo_factor,5)}, halo, {numpy.ceil(3.5*0.78*halo_factor)}, mean diff, {round(diff.mean(), 2)}, diff range, {round(diff.min(), 2)}, {round(diff.max(),2)}"
        )
    # numpy.testing.assert_allclose(op_resized, sk_resized, atol=1)


def test_anisotropic_scaling(graph):
    pass
