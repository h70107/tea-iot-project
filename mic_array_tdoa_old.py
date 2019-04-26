#!/usr/bin/env python3

## FILE: mic_array.py
## 

import pyaudio
import threading
import numpy as np
from gcc_phat import gcc_phat
import math
from beamformer import DAS

try:
    # python2 
    import Queue
except:
    # python3
    import queue as Queue

SOUND_SPEED = 343.2

MIC_DISTANCE_6P1 = 0.064
MAX_TDOA_6P1 = MIC_DISTANCE_6P1 / float(SOUND_SPEED)


class MicArray(object):

    def __init__(self, rate=16000, channels=8, chunk_size=None):
        self.pyaudio_instance = pyaudio.PyAudio()
        self.queue = Queue.Queue()
        self.sep_queue = Queue.Queue()

        self.quit_event = threading.Event()
        self.channels = channels
        self.sample_rate = rate
        self.chunk_size = chunk_size if chunk_size else rate / 100
        
        # self.beamformer = DAS(eps=0.01, nfft=1024)

        device_index = None
        for i in range(self.pyaudio_instance.get_device_count()):
            dev = self.pyaudio_instance.get_device_info_by_index(i)
            name = dev['name'].encode('utf-8')
            print(i, name, dev['maxInputChannels'], dev['maxOutputChannels'])
            if dev['maxInputChannels'] == self.channels:
                print('Use {}'.format(name))
                device_index = i
                break

        if device_index is None:
            raise Exception('can not find input device with {} channel(s)'.format(self.channels))

        self.stream = self.pyaudio_instance.open(
            input=True,
            start=False,
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=int(self.sample_rate),
            frames_per_buffer=int(self.chunk_size),
            stream_callback=self._callback,
            input_device_index=device_index,
        )

        # build tdoa matrix
        mic_theta           = np.arange(0, 2*np.pi, 2*np.pi/6)
        self.tdoa_matrix    = np.array([np.cos(mic_theta[1:]), np.sin(mic_theta[1:])]).T

        self.tdoa_matrix -= np.ones((len(mic_theta)-1, 2)) \
                             * np.array([np.cos(mic_theta[0]), np.sin(mic_theta[0])])

        self.tdoa_measures  = np.ones(((len(mic_theta)-1, ))) \
                                * SOUND_SPEED / (MIC_DISTANCE_6P1/2)
        

    def _callback(self, in_data, frame_count, time_info, status):
        self.queue.put(in_data)
        return None, pyaudio.paContinue

    def start(self):
        self.queue.queue.clear()
        self.stream.start_stream()


    def read_chunks(self):
        self.quit_event.clear()
        while not self.quit_event.is_set():
            frames = self.queue.get()
            if not frames:
                break

            yield frames

    def stop(self):
        self.quit_event.set()
        self.stream.stop_stream()
        self.queue.put('')

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        if value:
            return False
        self.stop()

    def get_direction(self, buf):
        best_guess = None
        MIC_GROUP_N = 5
        MIC_GROUP = [[1+i, 1] for i in range(1, MIC_GROUP_N+1)]
        tau = [0] * MIC_GROUP_N

        # estimate each group of delay 
       
        for i, v in enumerate(MIC_GROUP):
           tau[i] = gcc_phat(buf[v[0]::8], buf[v[1]::8], fs=self.sample_rate, max_tau=MAX_TDOA_6P1, interp=10)
        #############################
        
        
        #############################
        # least square solution of (cos, sin)
        sol = np.linalg.pinv(self.tdoa_matrix).dot( \
              (self.tdoa_measures * np.array(tau)).reshape(MIC_GROUP_N, 1)) 

        # found out theta
        # another 180.0 for positive value, 30.0 for respeaker architecture
        angle = math.atan2(sol[1],sol[0])
        fstoa=(angle/np.pi*180.0 + 210) %360
 
 ## 3D_POINT_SOURCE ##
        
#        phi = np.linalg.pinv(self.tdoa_matrix.dot(sol)).dot((self.tdoa_measures * np.array(tau)).reshape(MIC_GROUP_N, 1))
      #  print(sol.shape)
        a = min(sol[1] / math.sin(angle),1)
        sstoa = 90 -  np.rad2deg( math.asin(a))
      
      #  print(phi)
      #  sstoa= math.asin(phi[0])/np.pi*180

        return [fstoa,sstoa]


 ##GEMETRY METHOD ##
#        num =round( (fstoa + 30)/60. )
#        beta=1.125
#        tauc = [0] * 3
#        if num <= 3:
#            micg = [[0,num],[num+3,num],[num+3,0]] #tau1 tau2 tau3
#            for i, v in enumerate(micg):
#                tauc[i] = gcc_phat(buf[v[0]::8], buf[v[1]::8], fs=self.sample_rate, max_tau=MAX_TDOA_6P1, interp=16)
#            a1 = np.abs((tauc[0]*34320*beta))
#            a2 = np.abs((tauc[1]*34320*beta))
#            len1=np.abs((20.48+(2* math.pow(a1,2))-math.pow(a1,2))/((2*a2)-(4*a1)) )
#            check = min(1,(((2*len1*a1)+math.pow(a1,2)+10.24)/(6.4*(len1+a1))))
#            sstoa = math.acos(check)
#        elif num == 7: 
#            micg = [[0,1],[4,1],[4,0]] #tau1 tau2 tau3 
#            for i, v in enumerate(micg):
#                tauc[i] = gcc_phat(buf[v[0]::8], buf[v[1]::8], fs=self.sample_rate, max_tau=MAX_TDOA_6P1, interp=16)
#            a1 = np.abs((tauc[0]*34320*beta))
#            a2 = np.abs((tauc[1]*34320*beta))  
#            len1=np.abs((20.48+(2* math.pow(a1,2))-math.pow(a1,2))/((2*a2)-(4*a1)) )
#            check = min(1,(((2*len1*a1)+math.pow(a1,2)+10.24)/(6.4*(len1+a1))))
#            sstoa = math.acos(check)
#        else:
#            micg = [[0,num],[num-3,num],[num-3,0]] #tau1 tau2 tau3 
#            for i, v in enumerate(micg):
#                tauc[i] = gcc_phat(buf[v[0]::8], buf[v[1]::8], fs=self.sample_rate, max_tau=MAX_TDOA_6P1, interp=16)
#            a1 = np.abs((tauc[0]*34320*beta))
#            a2 = np.abs((tauc[1]*34320*beta))
#            len1=np.abs((20.48+(2* math.pow(a1,2))-math.pow(a1,2))/((2*a2)-(4*a1)) )
#            check = min(1,(((2*len1*a1)+math.pow(a1,2)+10.24)/(6.4*(len1+a1))))
#            sstoa = math.acos(check)
#        return  [num,fstoa,np.rad2deg(sstoa)]







##    def separationAt(self, buff, directions):
            
            
            
##        """
##        Arg:
##        --------------------------------------------------------
##        buff      [numpy.ndarray]   : shape(#frames, #channels)
##        direction [array of floats] : stores separatino directions
##
##        Return:
##        --------------------------------------------------------
##        en_speech [numpy.ndarray]   : shape(#frames, len(directions))
##        """
##        # buff.shape = -1, 8
##
##        self.sep_queue.put(self.beamformer.sound_separation(buff[:, 1:7], directions))



def test_8mic():
    import signal
    import time
    from pixel_ring import pixel_ring

    is_quit = threading.Event()
    

    def signal_handler(sig, num):
        is_quit.set()
        print('Quit')

    signal.signal(signal.SIGINT, signal_handler)
 
    en_speech = np.zeros((1,))
    raw = np.zeros((1, 8))
    with MicArray(16000, 8, 16000 / 8)  as mic:
        for frames in mic.read_chunks():
            chunk = np.fromstring(frames, dtype='int16')
            direction = mic.get_direction(chunk)
            #############################################
            #if (direction[2]==1):
            #    print('Warnning')
            
            
            print(chunk[0::8].shape)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            ############################################
            pixel_ring.set_direction(direction[0])
            
            print('@ {:.2f}'.format(direction[0]),'@ {:.2f}'.format(direction[1]))
           #  chunk = chunk / (2**15)
           #  chunk.shape = -1, 8

           #  raw = np.concatenate((raw, chunk), axis=0)
           #  
           #  mic.separationAt(chunk, [direction])
           #  en_speech = np.concatenate((en_speech, mic.sep_queue.get()),axis=0)
            if is_quit.is_set():
                break

    pixel_ring.off()
    return raw, en_speech


if __name__ == '__main__':
    # test_4mic()
    raw, en_speech = test_8mic()
