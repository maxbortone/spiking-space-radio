import numpy as np
from scipy.interpolate import interp1d

class AsynchronousDeltaModulator():

    def __init__(self, thrup, thrdn, resampling_factor):
        self.thrup = thrup
        self.thrdn = thrdn
        self.resampling_factor = resampling_factor
        self.time_length = None
        self.time_resampled = None
        self.vin = None
        self.rec = None
        self.up = None
        self.dn = None
        self.time_step = None

    def interpolate(self, time, vin):
        self.time_resampled, self.time_step = np.linspace(np.min(time), np.max(time), num=len(vin)*self.resampling_factor, endpoint=True, retstep=True)
        self.vin = interp1d(time, vin, kind='linear')(self.time_resampled)
        self.time_length = len(self.vin)
 
    def encode(self):
        self.up = np.zeros(self.time_length, dtype=bool)
        self.dn = np.zeros(self.time_length, dtype=bool)
        actual_dc = self.vin[0]
        for i in range(self.time_length):
            if (actual_dc + self.thrup) < self.vin[i]:
                self.up[i] = True
                actual_dc = self.vin[i]
            elif (actual_dc - self.thrdn) > self.vin[i]:
                self.dn[i] = True
                actual_dc = self.vin[i]

    def decode(self):
        actual_dc = self.vin[0]
        self.rec = np.zeros_like(self.vin)
        for i in range(self.time_length):
            if self.up[i]:
                actual_dc = actual_dc + self.thrup
            if self.dn[i]:
                actual_dc = actual_dc - self.thrdn
            self.rec[i] = actual_dc
