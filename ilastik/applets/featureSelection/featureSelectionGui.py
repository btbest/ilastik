###############################################################################
#   ilastik: interactive learning and segmentation toolkit
#
#       Copyright (C) 2011-2014, the ilastik developers
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
import logging
import numpy
import os

from functools import partial

# PyQt
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QApplication, QAbstractItemView, QFileDialog, QMessageBox

# lazyflow
import lazyflow.operators.filterOperators as filterOps
from ilastik.experimental.napari_layerViewer.napariglue import NapariImageAdapter
from lazyflow.operators.generic import OpSubRegion

# volumina
from volumina.utility import preferences
from volumina.widgets.layercontextmenu import layercontextmenu

# ilastik
from ilastik.widgets.featureTableWidget import FeatureEntry
from ilastik.widgets.featureDlg import FeatureDlg
from ilastik.utility import bind
from ilastik.experimental.napari_layerViewer.layerViewerGui import LayerViewerGui
from ilastik.config import cfg as ilastik_config

from ilastik.applets.base.applet import DatasetConstraintError

from ilastik.applets.featureSelection.opFeatureSelection import OpFeatureSelection
from lazyflow.slot import Slot

logger = logging.getLogger(__name__)


# ===---------------------------------------------------------------------------------------------------------------===
# === FeatureSelectionGui                                                                                           ===
# ===---------------------------------------------------------------------------------------------------------------===
class FeatureSelectionGui(LayerViewerGui):
    """"""

    # ##########################################
    # ## AppletGuiInterface Concrete Methods ###
    # ##########################################

    def appletDrawer(self):
        return self.drawer

    def viewerControlWidget(self):
        return self._viewerControlWidget

    def stopAndCleanUp(self):
        super(FeatureSelectionGui, self).stopAndCleanUp()

        # Unsubscribe to all signals
        for fn in self.__cleanup_fns:
            fn()

    def __init__(self, parentApplet, topLevelOperatorView):
        """"""
        self.topLevelOperatorView = topLevelOperatorView
        super(FeatureSelectionGui, self).__init__(parentApplet, topLevelOperatorView, crosshair=False)
        self.parentApplet = parentApplet

        self.__cleanup_fns = []

        self.topLevelOperatorView.InputImage.notifyDirty(bind(self.onFeaturesSelectionsChanged))
        self.topLevelOperatorView.SelectionMatrix.notifyDirty(bind(self.onFeaturesSelectionsChanged))
        self.topLevelOperatorView.FeatureListFilename.notifyDirty(bind(self.onFeaturesSelectionsChanged))
        self.__cleanup_fns.append(
            partial(self.topLevelOperatorView.SelectionMatrix.unregisterDirty, bind(self.onFeaturesSelectionsChanged))
        )
        self.__cleanup_fns.append(
            partial(
                self.topLevelOperatorView.FeatureListFilename.unregisterDirty, bind(self.onFeaturesSelectionsChanged)
            )
        )

        # Init feature dialog
        self.initFeatureDlg()

        self.onFeaturesSelectionsChanged()

    def initAppletDrawerUi(self):
        """
        Load the ui file for the applet drawer, which we own.
        """
        localDir = os.path.split(__file__)[0]
        # (We don't pass self here because we keep the drawer ui in a separate object.)
        self.drawer = uic.loadUi(localDir + "/featureSelectionDrawer.ui")
        self.drawer.SelectFeaturesButton.clicked.connect(self.onFeatureButtonClicked)
        self.drawer.UsePrecomputedFeaturesButton.clicked.connect(self.onUsePrecomputedFeaturesButtonClicked)
        dbg = ilastik_config.getboolean("ilastik", "debug")
        if not dbg:
            self.drawer.UsePrecomputedFeaturesButton.setHidden(True)

    def initViewerControlUi(self):
        """
        Load the viewer controls GUI, which appears below the applet bar.
        In our case, the viewer control GUI consists mainly of a layer list.

        TODO: Right now we manage adding/removing entries to a plain listview
              widget by monitoring the layerstack for changes.
              Ideally, we should implement a custom widget that does this for us,
              which would be initialized with the layer list model (like volumina.layerwidget)
        """
        self._viewerControlWidget = uic.loadUi(os.path.split(__file__)[0] + "/viewerControls.ui")

        layerListWidget = self._viewerControlWidget.featureListWidget
        layerListWidget.setSelectionMode(QAbstractItemView.SingleSelection)

        # Need to handle data changes because the layerstack model hasn't
        # updated his data yet by the time he calls the rowsInserted signal
        def handleLayerStackDataChanged(startIndex, stopIndex):
            row = startIndex.row()
            layerListWidget.item(row).setText(self.layerstack[row].name)

        def handleSelectionChanged(row):
            # Only one layer is visible at a time
            for i, layer in enumerate(self.layerstack):
                layer.visible = i == row

        def handleInsertedLayers(parent, start, end):
            for i in range(start, end + 1):
                layerListWidget.insertItem(i, self.layerstack[i].name)
                if layerListWidget.model().rowCount() == 1:
                    layerListWidget.item(0).setSelected(True)

        def handleRemovedLayers(parent, start, end):
            for i in reversed(list(range(start, end + 1))):
                layerListWidget.takeItem(i)

        self.layerstack.dataChanged.connect(handleLayerStackDataChanged)
        self.layerstack.rowsRemoved.connect(handleRemovedLayers)
        self.layerstack.rowsInserted.connect(handleInsertedLayers)
        layerListWidget.currentRowChanged.connect(handleSelectionChanged)

        # Support the same right-click menu as 'normal' layer list widgets
        def showLayerContextMenu(pos):
            idx = layerListWidget.indexAt(pos)
            layer = self.layerstack[idx.row()]
            layercontextmenu(layer, layerListWidget.mapToGlobal(pos), layerListWidget)

        layerListWidget.customContextMenuRequested.connect(showLayerContextMenu)
        layerListWidget.setContextMenuPolicy(Qt.CustomContextMenu)

    def setupLayers(self):
        if hasattr(self.drawer, "feature2dBox"):
            # Z may have been added or removed
            if "z" in self.topLevelOperatorView.InputImage.meta.original_axistags:
                self.drawer.feature2dBox.setHidden(False)
            else:
                self.drawer.feature2dBox.setHidden(True)

        opFeatureSelection = self.topLevelOperatorView
        inputSlot = opFeatureSelection.InputImage

        layers = []

        if inputSlot.ready():
            rawDataLayer = self.createStandardLayerFromSlot(inputSlot, name="Raw Data (display only)")
            layers.append(rawDataLayer)

        featureMultiSlot = opFeatureSelection.FeatureLayers
        if inputSlot.ready() and featureMultiSlot.ready():
            for featureIndex, featureSlot in enumerate(featureMultiSlot):
                assert featureSlot.ready()
                layers += self.get_feature_layers(inputSlot, featureSlot)

        layers[0].visible = True
        return layers

    def get_feature_layers(self, inputSlot: Slot, featureSlot: Slot) -> list[NapariImageAdapter]:
        layers = []

        channel_index: int = inputSlot.meta.axistags.channelIndex
        assert channel_index == featureSlot.meta.axistags.channelIndex
        input_channels: int = inputSlot.meta.shape[channel_index]
        feature_channels: int = featureSlot.meta.shape[channel_index]

        features_per_input = feature_channels // input_channels
        if not 0 < features_per_input <= 3:
            logger.warning(
                "The feature selection Gui does not yet support features with more than three channels per "
                "input channel. Some features will not be displayed entirely."
            )

        for input_channel in range(input_channels):
            feature_name = featureSlot.meta.description
            assert feature_name is not None
            if 2 <= input_channels <= 3:
                rgb = ["R", "G", "B"]
                feature_name += " (" + rgb[input_channel] + ")"
            if input_channels > 3:
                feature_name += " (Ch. {})".format(input_channel)

            opSubRegion = OpSubRegion(parent=self.topLevelOperatorView.parent)
            opSubRegion.Input.connect(featureSlot)
            start = [0] * len(featureSlot.meta.shape)
            start[channel_index] = input_channel * features_per_input
            stop = list(featureSlot.meta.shape)
            stop[channel_index] = (input_channel + 1) * features_per_input

            opSubRegion.Roi.setValue((tuple(start), tuple(stop)))

            feature_layer = self.createStandardLayerFromSlot(
                opSubRegion.Output, name=feature_name, opacity=1.0, visible=False
            )

            layers.append(feature_layer)

        return layers

    def initFeatureDlg(self):
        """
        Initialize the feature selection widget.
        """
        self.featureDlg = FeatureDlg(parent=self)
        self.featureDlg.setWindowTitle("Features")
        try:
            size = preferences.get("featureSelection", "dialog size")
            self.featureDlg.resize(*size)
        except TypeError:
            pass

        def saveSize():
            size = self.featureDlg.size()
            s = (size.width(), size.height())
            preferences.set("featureSelection", "dialog size", s)

        self.featureDlg.accepted.connect(saveSize)
        self.featureDlg.setImageToPreView(None)
        self.featureDlg.accepted.connect(self.onNewFeaturesFromFeatureDlg)

    def onUsePrecomputedFeaturesButtonClicked(self):
        options = QFileDialog.Options()
        if ilastik_config.getboolean("ilastik", "debug"):
            options |= QFileDialog.DontUseNativeDialog

        filenames, _filter = QFileDialog.getOpenFileNames(self, "Open Feature Files", ".", options=options)

        # Check if file exists
        if not filenames:
            return

        for filename in filenames:
            if not os.path.exists(filename):
                QMessageBox.critical(self, "Open Feature List", "File '%s' does not exist" % filename)
                return

        num_lanes = len(self.parentApplet.topLevelOperator.FeatureListFilename)
        if num_lanes != len(filenames):
            QMessageBox.critical(
                self,
                "Wrong number of feature files",
                "You must select all pre-computed feature files at once (shift-click).\n"
                "You selected {} file(s), but there are {} image(s) loaded".format(len(filenames), num_lanes),
            )
            return

        for filename, slot in zip(filenames, self.parentApplet.topLevelOperator.FeatureListFilename):
            slot.setValue(filename)

        # Create a dummy SelectionMatrix, just so the operator knows it is configured
        # This is a little hacky.  We should really make SelectionMatrix optional,
        # and then handle the choice correctly in setupOutputs, probably involving
        # the Output.meta.NOTREADY flag
        dummy_matrix = numpy.zeros((6, 7), dtype=bool)
        dummy_matrix[0, 0] = True
        self.parentApplet.topLevelOperator.SelectionMatrix.setValue(True)

        # Notify the workflow that some applets may have changed state now.
        # (For example, the downstream pixel classification applet can
        #  be used now that there are features selected)
        self.parentApplet.appletStateUpdateRequested()

    def onFeatureButtonClicked(self):
        # Remove all pre-computed feature files
        for slot in self.parentApplet.topLevelOperator.FeatureListFilename:
            slot.disconnect()

        # The first time we open feature selection, the minimal set of features should be set. Afterwards, if the
        # data input is changed, the feature selection dialog should appear, if adjustments are necessary
        if not self.topLevelOperatorView.SelectionMatrix.ready():
            self.topLevelOperatorView.SelectionMatrix.setValue(self.topLevelOperatorView.MinimalFeatures)

        # Other slots need to be ready (they also should, as they have default values)
        assert self.topLevelOperatorView.FeatureIds.ready()
        assert self.topLevelOperatorView.Scales.ready()
        assert self.topLevelOperatorView.ComputeIn2d.ready(), self.topLevelOperatorView.ComputeIn2d.value

        # Refresh the dialog data in case it has changed since the last time we were opened
        # (e.g. if the user loaded a project from disk)
        # This also ensures to restore the selection after previously canceling the feature dialog
        opFeatureSelection = self.topLevelOperatorView

        # Map from groups of feature IDs to groups of feature NAMEs
        groupedNames = []
        for group, featureIds in opFeatureSelection.FeatureGroups:
            featureEntries = []
            for featureId in featureIds:
                featureName = opFeatureSelection.FeatureNames[featureId]
                availableFilterOps = {
                    key[2:]: value for key, value in filterOps.__dict__.items() if key.startswith("Op")
                }
                minimum_scale = availableFilterOps[featureId].minimum_scale
                featureEntries.append(FeatureEntry(featureName, minimum_scale))
            groupedNames.append((group, featureEntries))
        self.featureDlg.createFeatureTable(
            groupedNames,
            opFeatureSelection.Scales.value,
            opFeatureSelection.ComputeIn2d.value,
            opFeatureSelection.WINDOW_SIZE,
        )
        # update feature dialog to show/hide z dimension specific 'compute in 2d' flags
        if self.topLevelOperatorView.InputImage.ready() and self.topLevelOperatorView.ComputeIn2d.value:
            ts = self.topLevelOperatorView.InputImage.meta.getTaggedShape()
            hide = ("z" not in ts or ts["z"] == 1) and all(self.topLevelOperatorView.ComputeIn2d.value)
            self.featureDlg.setComputeIn2dHidden(hide)

        matrix = opFeatureSelection.SelectionMatrix.value
        featureOrdering = opFeatureSelection.FeatureIds.value

        # Re-order the feature matrix using the loaded feature ids
        reorderedMatrix = numpy.zeros(matrix.shape, dtype=bool)
        newrow = 0
        for group, featureIds in OpFeatureSelection.FeatureGroups:
            for featureId in featureIds:
                oldrow = featureOrdering.index(featureId)
                reorderedMatrix[newrow] = matrix[oldrow]
                newrow += 1

        self.featureDlg.set_selectionMatrix(reorderedMatrix)

        # Now open the feature selection dialog
        self.featureDlg.exec_()

    def onNewFeaturesFromFeatureDlg(self):
        opFeatureSelection = self.topLevelOperatorView
        if opFeatureSelection is not None:
            # Save previous settings
            old_scales = opFeatureSelection.Scales.value
            old_computeIn2d = opFeatureSelection.ComputeIn2d.value
            old_features = opFeatureSelection.SelectionMatrix.value

            # Disable gui
            self.parentApplet.busy = True
            self.parentApplet.appletStateUpdateRequested()
            QApplication.instance().setOverrideCursor(QCursor(Qt.WaitCursor))

            try:
                # Apply new settings
                # Disconnect an input (used like a transaction slot)
                opFeatureSelection.SelectionMatrix.disconnect()

                opFeatureSelection.Scales.setValue(self.featureDlg.get_scales())
                opFeatureSelection.ComputeIn2d.setValue(self.featureDlg.get_computeIn2d())
                # set disconnected slot at last (used like a transaction slot)
                opFeatureSelection.SelectionMatrix.setValue(self.featureDlg.get_selectionMatrix())
            except (DatasetConstraintError, RuntimeError) as ex:
                # The user selected some scales that were too big.
                if isinstance(ex, DatasetConstraintError):
                    QMessageBox.critical(self, "Invalid selection", ex.message)
                else:
                    QMessageBox.critical(self, "Invalid selection", "You selected the exact same feature twice.")

                # Restore previous settings
                opFeatureSelection.SelectionMatrix.disconnect()
                opFeatureSelection.Scales.setValue(old_scales)
                opFeatureSelection.ComputeIn2d.setValue(old_computeIn2d)
                opFeatureSelection.SelectionMatrix.setValue(old_features)

            # Re-enable gui
            QApplication.instance().restoreOverrideCursor()
            self.parentApplet.busy = False

            # Notify the workflow that some applets may have changed state now.
            # (For example, the downstream pixel classification applet can
            #  be used now that there are features selected)
            self.parentApplet.appletStateUpdateRequested()

    def onFeaturesSelectionsChanged(self):
        """
        Handles changes to our top-level operator's ImageInput and matrix of feature selections.
        """
        # Update the drawer caption
        fff = (
            self.topLevelOperatorView.FeatureListFilename.ready()
            and len(self.topLevelOperatorView.FeatureListFilename.value) != 0
        )

        if not self.topLevelOperatorView.SelectionMatrix.ready() and not fff:
            self.drawer.caption.setText("(No features selected)")
            self.layerstack.clear()
        elif fff:
            self.drawer.caption.setText("(features from files)")
        else:
            nr_feat = self.topLevelOperatorView.SelectionMatrix.value.sum()
            self.drawer.caption.setText(f"(Selected {nr_feat} features)")
