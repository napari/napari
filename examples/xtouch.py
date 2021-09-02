import numpy as np
import pandas as pd
import rtmidi.midiutil


class XTouch:
    def __init__(self, viewer, hold_thresh=0.5):
        self.viewer = viewer
        self.midi_in, _ = rtmidi.midiutil.open_midiinput(0)
        self.midi_in.set_callback(self.receive_set)
        self.midi_out, _ = rtmidi.midiutil.open_midioutput(0)
        table = []
        press_dict = {}
        control_ids = {
            'a-button': np.arange(24).reshape((3, 8)),
            'b-button': np.arange(24, 48).reshape((3, 8)),
            'a-rotary': np.arange(1, 9),
            'b-rotary': np.arange(11, 19),
            'a-slider': np.array([9]),
            'b-slider': np.array([10]),
        }
        for name, array in control_ids.items():
            layer, control_type = name.split('-')
            for idx in np.ndindex(*array.shape):
                control_id = array[idx]
                press_dict[control_id] = (array, idx)
                virtual2raw = 127 if control_type == 'button' else 1
                virtual2data = 1
                value = 0
                table.append((
                    layer,
                    control_type,
                    control_id,
                    idx,
                    value,
                    virtual2raw,
                    virtual2data,
                    None,
                    None,
                ))
        self.table = pd.DataFrame(
            table,
            columns=[
                'layer',  # midi control layer, 'a' or 'b'
                'type',  # 'rotary', 'button', or 'slider'
                'id',  # midi control ID
                'index',  # position of the control in logical space
                'value',  # value of the control in 0-127
                'virtual2raw',  # conv. factor when number of levels not 128
                'raw2data',  # conv factor to actual slider numbers
                'fw',  # function to apply slider values to viewer
                'inv',  # function to apply viewer values to slider
            ],
        )
        self.table.set_index('id', drop=False)

    def receive_set(self, message_time_tup, data):
        message, time = message_time_tup
        msg_type, control_id, value = message
        if msg_type == 186:  # rotary or slider
            self.receive_continuous(control_id, value)
        elif msg_type == 154:  # button press down
            pass
        elif msg_type == 138:  # button press up
            pass

    def receive_continuous(self, control_id, value):
        control = self.table.loc[control_id]
        if control['fw'] is not None:
            control['fw'](value)

    def send_continuous(self, control_id, value):
        self.midi_out.send_message((186, control_id, value))

    def bind_current_step(self, layer, axis, rotary0, rotary1):
        cond0 = (
            (self.table['index'] == (rotary0,))
            & (self.table['type'] == 'rotary')
            & (self.table['layer'] == layer)
        )
        rotary0_id = int(self.table.loc[cond0]['id'])
        cond1 = (
            (self.table['index'] == (rotary1,))
            & (self.table['type'] == 'rotary')
            & (self.table['layer'] == layer)
        )
        rotary1_id = int(self.table.loc[cond1]['id'])

        nsteps = self.viewer.dims.nsteps[axis]
        rot0_nsteps = min(nsteps, 13)
        self.table.loc[rotary0_id, 'virtual2raw'] = (v2r := 127 / rot0_nsteps)
        self.table.loc[rotary0_id, 'raw2data'] = (r2d := nsteps / 127)
        self.table.loc[rotary1_id, 'virtual2raw'] = (v2r1 := 127 / v2r)

        def process_current_step_event(ev):
            current_step = ev.value[axis]
            r0_value = int(current_step / r2d)
            self.table.loc[rotary0_id, 'value'] = r0_value
            self.send_continuous(rotary0_id, r0_value)
            r1_value = (current_step % v2r) * v2r1
            self.table.loc[rotary1_id, 'value'] = r1_value
            self.send_continuous(rotary1_id, r1_value)

        self.viewer.dims.events.current_step.connect(process_current_step_event)

        def fw0(value):
            prev_value = self.table.loc[rotary0_id, 'value']
            down = (prev_value > value) or (prev_value == value and value == 0)
            increment = int(np.round(v2r))
            if down:
                increment = -increment
            next_step = list(self.viewer.dims.current_step)
            next_step[axis] = np.clip(next_step[axis] + increment, 0, nsteps - 1)
            self.viewer.dims.current_step = tuple(next_step)

        self.table.loc[rotary0_id, 'fw'] = fw0

        def fw1(value):
            prev_value = self.table.loc[rotary1_id, 'value']
            up = (prev_value < value) or (prev_value == value and value == 127)
            next_step = list(self.viewer.dims.current_step)
            increment = 1 if up else -1
            next_step[axis] = np.clip(next_step[axis] + increment, 0, nsteps - 1)
            self.viewer.dims.current_step = tuple(next_step)

        self.table.loc[rotary1_id, 'fw'] = fw1

    def bind_slider(self, layer):
        table = self.table
        cond = (table['layer'] == layer) & (table['type'] == 'slider')
        slider_id = table.loc[cond, 'id']

        def change_opacity(value):
            vlayer = self.viewer.layers[-1]
            vlayer.opacity = value / 127

        table.loc[slider_id, 'fw'] = change_opacity

if __name__ == '__main__':
    import napari
    image = np.random.random((100, 200, 200))
    v = napari.view_image(image)
    xt = XTouch(v)
    xt.bind_current_step('b', 0, 0, 1)
    xt.bind_slider('b')
    napari.run()
