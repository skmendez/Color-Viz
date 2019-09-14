import time

import cv2
import numpy as np

import config
from capture import ThreadedVideoCapture
from colors import Color, ColorWheel, table_from_cmap
from dsp import ExpFilter
from stream import stream
from filter import butter_bandpass_filter

r = Color(255, 0, 0)
g = Color(0, 255, 0)
b = Color(0, 0, 255)
c = Color(0, 255, 255)
m = Color(255, 0, 255)
y = Color(255, 255, 0)

m1 = Color(39, 103, 191)
m2 = Color(32, 175, 219)
m3 = Color(53, 232, 206)
wheel = ColorWheel({0: m1, 54: m2, 118: m3, 138: m3, 202: m2})
conv = lambda v: int(v * 256/360)

wheel2 = ColorWheel({conv(v): c for v, c in {0: r, 60: y, 120: g, 180: c, 240: b, 300: m}.items()})
print("here!")

class Colorama:
    def __init__(self, filename, table):
        self.filename = filename
        #self.lattice = cv2.cvtColor(cv2.imread(f"images/{filename}"), cv2.COLOR_RGB2GRAY)
        self.table = table
        self._generated = False

    def create_window(self):
        if not self._generated:
            cv2.namedWindow(self.filename, cv2.WINDOW_NORMAL)
            self._generated = True


lut_list = [table_from_cmap("hsv", False),
            wheel.generate_lookup(),
            table_from_cmap("plasma", True),
            table_from_cmap("cool", True),
            table_from_cmap("twilight_shifted", False),
            table_from_cmap("spring", True)]

lut = lut_list[0]


class Loudness:
    def __init__(self):
        self.rolling = np.random.rand(4, config.FRAMES_PER_BUFFER) / 1e16

    def update(self, y):
        self.rolling[:-1] = self.rolling[1:]
        self.rolling[-1, :] = np.copy(y)
        y_data = np.concatenate(self.rolling, axis=0)
        inter = (y_data.astype(float)**2)
        power = np.mean(inter)
        return power

class LoudnessMax:
    def update(self, y):
        return np.abs(np.max(y))/20

def level(cur, max_val, size):
    arr = np.zeros(size).astype(np.uint8)
    largest_idx = int(round((cur / max_val) * size[0]))
    arr[:largest_idx] = 255
    return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

class App:
    actions = {"[": lambda self: self.mod_repeat(1/1.1),
               "]": lambda self: self.mod_repeat(1.1)}

    def mod_repeat(self, val):
        self.repeat *= val

    def __init__(self, lut_list):
        self.title = "viz"
        self.resolution = (720, 1280)
        self.total = 0
        self.power_filter = ExpFilter(alpha_decay=0.05, alpha_rise=0.2)
        self.loud = LoudnessMax()
        self.repeat = 1
        cv2.namedWindow(self.title, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.key_ready = time.time()-1
        self.lut_list = lut_list
        self.cur_idx = 0

    @property
    def lut(self):
        return self.lut_list[self.cur_idx]

    @property
    def x(self):
        return self.resolution[0]

    @property
    def y(self):
        return self.resolution[1]

    def run(self):
        with ThreadedVideoCapture(1) as cap:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.y)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.x)
            while True:
                y = np.mean(
                    np.frombuffer(stream.read(config.FRAMES_PER_BUFFER, exception_on_overflow=False), dtype=np.int16).reshape(-1, config.CHANNELS),
                    axis=1)/2 ** 16
                stream.read(stream.get_read_available(), exception_on_overflow=False)
                y = butter_bandpass_filter(y, 100, config.RATE)
                power = self.loud.update(y)
                power *= 10 ** config.GAIN
                # print("{:4.1f} {:4.1f}".format(power, val))
                val = self.power_filter.update(power)
                self.total += power


                ret, frame = cap.read()
                arr = cv2.cvtColor(frame[:, ::-1], cv2.COLOR_BGR2RGB)
                frame = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                # frame = cv2.Canny(frame, 60, 120)
                process = (frame*self.repeat+int(self.total)).astype(np.uint8)
                # process = np.maximum(process, cv2.Canny(frame, 60, 120))
                out = self.lut.process_image(process)
                # out = np.maximum(out, np.repeat(cv2.Canny(frame, 60, 120)[:,:,np.newaxis], 3, axis=2))
                alpha = 1
                # cv2.imshow("viz", cv2.cvtColor(cv2.addWeighted(arr, 1-alpha, out, alpha, 0), cv2.COLOR_BGR2RGB))
                cv2.imshow(self.title, cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
                cv2.imshow("level", level(power, 20, size=(400, 20)))
                cv2.imshow("level_stab", level(val, 20, size=(400, 20)))

                val = chr(cv2.waitKey(5) & 0xFF)
                if val == "q":
                    cv2.destroyAllWindows()
                    break
                elif "0" <= val <= "9":
                    idx = (ord(val)-49) % 10
                    if idx < len(self.lut_list):
                        self.cur_idx = idx
                elif val in self.actions:
                    if time.time() - self.key_ready > .5:
                        self.actions[val](self)
                        self.key_ready = time.time()

if __name__ == '__main__':
    a = App(lut_list)
    a.run()