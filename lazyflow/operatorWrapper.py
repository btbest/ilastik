###############################################################################
#   lazyflow: data flow based lazy parallel computation framework
#
#       Copyright (C) 2011-2014, the ilastik developers
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
# 		   http://ilastik.org/license/
###############################################################################
# Python
import functools

# lazyflow
from lazyflow.operator import Operator


class OperatorWrapper(Operator):
    name = "OperatorWrapper"

    def __init__(
        self,
        operatorClass,
        operator_args=None,
        operator_kwargs=None,
        parent=None,
        graph=None,
        promotedSlotNames=None,
        broadcastingSlotNames=None,
        write_logs=False,
    ):
        """Constructs a wrapper for the given operator. That is,
        manages a list of copies of the original operator, and
        provides access to these inner operators' slots via external
        multislots.

        :param operatorClass: An operator type that can be constructed
          with the given args and kwargs

        :param operator_args: Positional arguments for operator's
          constructor. Note: Do not include 'parent' and 'graph'
          arguments.

        :param operator_kwargs: Keyword arguments for the operator's
          constructor. Note: Do not include 'parent' and 'graph'
          arguments

        :param parent: The parent of the OperatorWrapper

        :param graph: the graph operator to init each inner operator with

        :param promotedSlotNames: If provided, only those slots will be promoted when
            replicated. All other slots will be replicated without
            promotion, and their input values will be broadcasted to
            all inner operators.
          If not provided (i.e. promotedSlotNames=None), the default
            behavior is to promote ALL replicated slots.
          Note: Outputslots are always promoted, regardless of whether
            or not they appear in the promotedSlotNames argument.

        :param write_logs: Debugging feature. The wrapper and wrapped ops will write debug logs if True.
            The wrapped operator's __init__ must also accept the write_logs kwarg.
            Make sure the `lazyflow.op_debug` logger has level=DEBUG in the logging config.

        """
        super(OperatorWrapper, self).__init__(parent=parent, graph=graph, write_logs=write_logs)
        if operator_args is None:
            operator_args = ()
        if operator_kwargs is None:
            operator_kwargs = {}
        assert isinstance(operator_args, (tuple, list))
        assert isinstance(operator_kwargs, dict)
        if write_logs:
            operator_kwargs.update({"write_logs": write_logs})
        self._createInnerOperator = functools.partial(operatorClass, parent=self, *operator_args, **operator_kwargs)

        self._initialized = False

        self.name = "Wrapped " + operatorClass.name
        if self._debug_logger:
            self._debug_logger.debug(f"Wrapper {id(self)} name={self.name}")

        allInputSlotNames = set([s.name for s in operatorClass.inputSlots])

        if promotedSlotNames is not None:
            assert (
                broadcastingSlotNames is None
            ), "Please specify either the promoted slots or the broadcasting slots, not both."
            for name in promotedSlotNames:
                assert (
                    name in allInputSlotNames
                ), "Didn't recognize slot name to promote: '{}' is not a valid input slot name for this operator".format(
                    name
                )

            # 'Promoted' slots will be exposed as multi-slots
            # All others will be broadcasted
            promotedSlotNames = set(promotedSlotNames)

        elif broadcastingSlotNames is not None:
            # 'Broadcasting' slots are NOT exposed as multi-slots.
            # Each is exposed as a single slot that is shared by all
            # inner operators.

            for name in broadcastingSlotNames:
                assert (
                    name in allInputSlotNames
                ), "Didn't recognize slot name to broadcast: '{}' is not a valid input slot name for this operator".format(
                    name
                )

            # set difference
            promotedSlotNames = allInputSlotNames - set(broadcastingSlotNames)

        else:
            # No slots specified: All original slots are promoted by
            # default
            promotedSlotNames = set(slot.name for slot in operatorClass.inputSlots)

        # All Outputs are always promoted
        promotedSlotNames |= set(slot.name for slot in operatorClass.outputSlots)

        self.promotedSlotNames = promotedSlotNames

        self.innerOperators = []

        # replicate input slot definitions
        for innerSlot in sorted(operatorClass.inputSlots, key=lambda s: s._global_slot_id):
            level = innerSlot.level
            if innerSlot.name in self.promotedSlotNames:
                level += 1
            outerSlot = innerSlot._getInstance(self, level=level)
            self.inputs[outerSlot.name] = outerSlot
            setattr(self, outerSlot.name, outerSlot)

        # replicate output slot definitions
        for innerSlot in sorted(operatorClass.outputSlots, key=lambda s: s._global_slot_id):
            level = innerSlot.level + 1
            outerSlot = innerSlot._getInstance(self, level=level)
            self.outputs[outerSlot.name] = outerSlot
            setattr(self, outerSlot.name, outerSlot)

        # register callbacks for inserted and removed input subslots
        for s in list(self.inputs.values()):
            if s.name in self.promotedSlotNames:
                s.notifyInserted(self._callbackInserted)
                s.notifyRemove(self._callbackPreRemove)
                s.notifyRemoved(self._callbackPostRemoved)
                s._notifyConnect(self._callbackConnect)

        # register callbacks for inserted and removed output subslots
        for s in list(self.outputs.values()):
            s.notifyInserted(self._callbackInserted)
            s.notifyRemove(self._callbackPreRemove)
            s.notifyRemoved(self._callbackPostRemoved)

        for s in list(self.inputs.values()):
            assert len(s) == 0
        for s in list(self.outputs.values()):
            assert len(s) == 0

    def __getitem__(self, key):
        return self.innerOperators[key]

    def __len__(self):
        return len(self.innerOperators)

    def __iter__(self):
        return self.innerOperators.__iter__()

    def _callbackInserted(self, slot, index, size):
        self._insertInnerOperator(index, size)

    def _callbackPreRemove(self, slot, index, length):
        # Prepare for disconnect which will occur in _removeInnerOperator
        # (Can't properly disconnect a slot if backpropagate_values is True)
        for mslot in list(self.outputs.values()):
            if len(mslot) > length:
                mslot[index].backpropagate_values = False
                mslot[index].unregisterDisconnect(self.handleEarlyDisconnect)

    def _callbackPostRemoved(self, slot, index, size):
        self._removeInnerOperator(index, size)

    def _callbackConnect(self, slot):
        slot.resize(len(self.innerOperators))
        for index, innerOp in enumerate(self.innerOperators):
            innerOp.inputs[slot.name].connect(slot[index])

    def propagateDirty(self, slot, subindex, roi):
        # Nothing to do: All inputs are directly connected to internal
        # operators.
        pass

    def _insertInnerOperator(self, index, length):
        if len(self.innerOperators) >= length:
            return self.innerOperators[index]
        if self._debug_logger:
            self._debug_logger.debug(f"Inserting inner operator at index {index}")
        op = self._createInnerOperator()

        # If anyone calls setValue() on one of these slots,
        # forward the setValue call to the slot's upstream_slot (the
        # outer slot on the operator wrapper)
        for slot in list(op.inputs.values()):
            slot.backpropagate_values = True
            slot.notifyDisconnect(self.handleEarlyDisconnect)

        self.innerOperators.insert(index, op)

        # Connect the inner operator's inputs to our outer input
        # slots
        for key, outerSlot in list(self.inputs.items()):
            # Only connect to a subslot if it was promoted during
            # wrapping
            if outerSlot.name in self.promotedSlotNames:
                outerSlot.insertSlot(index, length)
                upstream_slot = outerSlot[index]
            else:
                upstream_slot = outerSlot
            if op.inputs[key].upstream_slot is not None:
                msg = (
                    "Can't set up OperatorWrapper connections."
                    " Input slot {} is already connected to a"
                    " upstream_slot (must have happened in {}'s"
                    " constructor".format(key, op.name)
                )
                raise RuntimeError(msg)
            op.inputs[key].connect(upstream_slot)

        # Connect our outer output slots to the inner operator's output slots.
        for key, mslot in list(self.outputs.items()):
            mslot.insertSlot(index, length)
            mslot[index].backpropagate_values = True
            mslot[index].connect(op.outputs[key])
            mslot[index].notifyDisconnect(self.handleEarlyDisconnect)
            # mslot[index]._changed()
        return op

    def handleEarlyDisconnect(self, slot):
        assert self._cleaningUp, "You aren't allowed to disconnect the internal connections of an operator wrapper."

    def _removeInnerOperator(self, index, length):
        if len(self.innerOperators) <= length:
            return
        assert index < len(self.innerOperators)

        op = self.innerOperators.pop(index)
        for slot in list(op.inputs.values()):
            slot.backpropagate_values = False
            slot.unregisterDisconnect(self.handleEarlyDisconnect)

        for oslot in list(self.outputs.values()):
            oslot.removeSlot(index, length)

        for key, outerSlot in list(self.inputs.items()):
            if outerSlot.name in self.promotedSlotNames:
                outerSlot.removeSlot(index, length)

        op.cleanUp()

    def _setupOutputs(self):
        pass

    def execute(self, slot, subindex, roi, result):
        # this should never be called !!!
        assert False, (
            "OperatorWrapper execute() function should never be called.  "
            "You can only ask for data from SUBslots, not the outer multi-slots themselves."
        )

    def setInSlot(self, slot, subindex, key, value):
        # Nothing to do here. Calls to Slot.setitem are already
        # forwarded to all downstream partners.
        pass
