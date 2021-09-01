import numpy as np
import rtmidi


class XTouch:
    def __init__(self, hold_thresh=0.5):
        self.press_dict = {}
        self.arrays = {
            'a-buttons', np.arange(24).reshape((3, 8)),
            'b-buttons', np.arange(24, 48).reshape((3, 8)),
            'a-rotary', np.arange(1, 9),
            'b-rotary', np.arange(11, 19),
            'a-slider', np.array([9]),
            'b-slider', np.array([10]),
        }
        for array in self.arrays.values():
            for idx in np.ndindex(*array.shape):
                num = array[idx]
                self.press_dict[num] = (array, idx)

    def receive_set(self, message_time_tup, data):
        message, time = message_time_tup
        msg_type, control_id, value = message
        if msg_type == 186:  # rotary or slider
            pass
        elif msg_type == 154:  # button press down
            pass
        elif msg_type == 138:  # button press up
            pass

    def receive_rotary(self, control_id, )
