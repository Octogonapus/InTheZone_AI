import psutil
from ctypes import *
import ctypes


class HeapScanner(object):
    def __init__(self, base=0x00000000):
        self.byte_shifts = [0, 8, 16, 24, 36, 48, 60]
        self.base = base

    @staticmethod
    def get_pid(name):
        for proc in psutil.process_iter():
            if str(name) in str(proc.name):
                return proc.pid

    def string6_from_uint(self, value):
        return "".join(map(chr, [(value >> i) & 0xff for i in self.byte_shifts]))

    def print_memory(self, address, value):
        print("Raw memory: ", str(hex(address)), value)
        mem_bytes = [(value >> i) & 0xff for i in self.byte_shifts]
        print("Bytes: ", mem_bytes)
        print("As string[4]: ", "".join(map(chr, mem_bytes)))

    def scan_memory(self, text):
        text_sorted = sorted(text)
        PID = self.get_pid('inthezone.exe')
        PROCESS_QUERY_INFORMATION = 0x0400
        PROCESS_VM_READ = 0x0010

        process = windll.kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, PID)
        readprocess = windll.kernel32.ReadProcessMemory
        rdbuf = c_ubyte()
        bytread = c_ulong(0)

        addr = self.base
        counter = 0
        substring_length = len(text)
        mem_bytes = [0] * substring_length
        while True:
            try:
                if readprocess(process,
                               ctypes.c_void_p(addr),
                               ctypes.byref(rdbuf),
                               ctypes.sizeof(rdbuf),
                               ctypes.byref(bytread)):
                    mem_bytes[counter % substring_length] = rdbuf.value
                    if text_sorted == sorted("".join(map(chr, mem_bytes))):
                        print("Found", text, "at address", hex(addr))
                        return addr

                # if addr % 100000 == 0:
                #     print("Progress: ", hex(addr))

                addr = addr + 1
                counter = counter + 1
            except Exception:
                print("Failure! Last addr: ", hex(addr))
                break
