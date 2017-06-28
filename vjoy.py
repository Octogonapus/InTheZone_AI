import ctypes
import struct
import time
import numpy as np

CONST_DLL_VJOY = "C:\\Program Files\\vJoy\\x64\\vJoyInterface.dll"


class vJoy(object):
    def __init__(self, reference=1):
        self.handle = None
        self.dll = ctypes.CDLL(CONST_DLL_VJOY)
        self.reference = reference
        self.acquired = False

    def open(self):
        if self.dll.AcquireVJD(self.reference):
            self.acquired = True
            return True
        return False

    def close(self):
        if self.dll.RelinquishVJD(self.reference):
            self.acquired = False
            return True
        return False

    def gen_joy_pos(self,
                    wthrottle=0, wrudder=0, waileron=0,
                    waxis_x=0, waxis_y=0, waxis_z=0,
                    waxis_x_rot=0, waxis_y_rot=0, waxis_z_rot=0,
                    wslider=0, wdial=0, wwheel=0,
                    waxis_vx=0, waxis_vy=0, waxis_vz=0,
                    waxis_vbrx=0, waxis_vbry=0, waxis_vbrz=0,
                    lbuttons=0, bhats=0, bhats_ex1=0, bhats_ex2=0, bhats_ex3=0):
        """
        typedef struct _JOYSTICK_POSITION
        {
            BYTE    bDevice; // Index of device. 1-based
            LONG    wThrottle;
            LONG    wRudder;
            LONG    wAileron;
            LONG    wAxisX;
            LONG    wAxisY;
            LONG    wAxisZ;
            LONG    wAxisXRot;
            LONG    wAxisYRot;
            LONG    wAxisZRot;
            LONG    wSlider;
            LONG    wDial;
            LONG    wWheel;
            LONG    wAxisVX;
            LONG    wAxisVY;
            LONG    wAxisVZ;
            LONG    wAxisVBRX;
            LONG    wAxisVBRY;
            LONG    wAxisVBRZ;
            LONG    lButtons;   // 32 buttons: 0x00000001 means button1 is pressed, 0x80000000 -> button32 is pressed
            DWORD   bHats;      // Lower 4 bits: HAT switch or 16-bit of continuous HAT switch
                        DWORD   bHatsEx1;   // 16-bit of continuous HAT switch
                        DWORD   bHatsEx2;   // 16-bit of continuous HAT switch
                        DWORD   bHatsEx3;   // 16-bit of continuous HAT switch
        } JOYSTICK_POSITION, *PJOYSTICK_POSITION;
        """
        joyPosFormat = "BlllllllllllllllllllIIII"
        pos = struct.pack(joyPosFormat, self.reference, wthrottle, wrudder,
                          waileron, waxis_x, waxis_y, waxis_z, waxis_x_rot, waxis_y_rot,
                          waxis_z_rot, wslider, wdial, wwheel, waxis_vx, waxis_vy, waxis_vz,
                          waxis_vbrx, waxis_vbry, waxis_vbrz, lbuttons, bhats, bhats_ex1, bhats_ex2, bhats_ex3)
        return pos

    def update(self, joy_pos):
        if self.dll.UpdateVJD(self.reference, joy_pos):
            return True
        return False

    # Not working, send buttons one by one
    def send_buttons(self, bstate):
        joyPosition = self.gen_joy_pos(lbuttons=bstate)
        return self.update(joyPosition)

    def set_button(self, index, state):
        if self.dll.SetBtn(state, self.reference, index):
            return True
        return False


vj = vJoy()


# valueX, valueY between -1.0 and 1.0
# scale between 0 and 16000
def set_joy(val_x, val_y, scale=10000.0, val_axis_x_rot=0, val_axis_y_rot=0, val_axis_z_rot=0):
    xPos = int(val_x * scale)
    yPos = int(val_y * scale)
    joystickPosition = vj.gen_joy_pos(waxis_x=16000 + xPos, waxis_y=16000 + yPos, waxis_x_rot=val_axis_x_rot,
                                      waxis_y_rot=val_axis_y_rot, waxis_z_rot=val_axis_z_rot)
    vj.update(joystickPosition)


def test():
    print("vj opening", flush=True)
    vj.open()
    time.sleep(2)
    print("sending axes", flush=True)
    for i in range(0, 1000, 1):
        xPos = int(10000.0 * np.sin(2.0 * np.pi * i / 1000))
        yPos = int(10000.0 * np.sin(2.0 * np.pi * i / 100))
        print(xPos, flush=True)
        joystickPosition = vj.gen_joy_pos(waxis_x=16000 + xPos, waxis_y=16000 + yPos)
        vj.update(joystickPosition)
        time.sleep(0.01)
    joystickPosition = vj.gen_joy_pos(waxis_x=16000, waxis_y=16000)
    vj.update(joystickPosition)
    vj.send_buttons(0)
    print("vj closing", flush=True)
    vj.close()


if __name__ == '__main__':
    test()
