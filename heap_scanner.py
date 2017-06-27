import psutil
from ctypes import *
import ctypes


class HeapScanner(object):
    def __init__(self, base=0x00000000):
        self.byte_shifts = [0]
        self.base = base

        PID = self.get_pid('inthezone.exe')
        PROCESS_QUERY_INFORMATION = 0x0400
        PROCESS_VM_READ = 0x0010
        self.process = windll.kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, PID)
        self.readprocess = windll.kernel32.ReadProcessMemory
        self.rdbuf = c_ubyte()
        self.byteread = c_ulong(0)

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

    def read_memory(self, address, length):
        addr = address
        counter = 0
        mem_bytes = [0] * length

        while counter < length:
            try:
                if self.readprocess(self.process,
                                    ctypes.c_void_p(addr),
                                    ctypes.byref(self.rdbuf),
                                    ctypes.sizeof(self.rdbuf),
                                    ctypes.byref(self.byteread)):
                    mem_bytes[counter % length] = self.rdbuf.value

                addr += 1
                counter += 1
            except Exception:
                print("Failure! Last addr: ", hex(address))

        result = [""] * length
        for i in range(0, length):
            result[i] = self.string6_from_uint(mem_bytes[i])
        return "".join(map(chr, mem_bytes))

    def scan_memory(self, text):
        text_sorted = sorted(text)
        counter = 0
        substring_length = len(text)
        mem_bytes = [0] * substring_length

        addr = self.base
        while True:
            try:
                if self.readprocess(self.process,
                                    ctypes.c_void_p(addr),
                                    ctypes.byref(self.rdbuf),
                                    ctypes.sizeof(self.rdbuf),
                                    ctypes.byref(self.byteread)):
                    mem_bytes[counter % substring_length] = self.rdbuf.value
                    if text_sorted == sorted("".join(map(chr, mem_bytes))):
                        print("Found", text, "at address", hex(addr))
                        return addr - substring_length + 1

                addr += 1
                counter += 1
            except Exception:
                print("Failure! Last addr: ", hex(addr))
                break
