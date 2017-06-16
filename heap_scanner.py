import psutil
from ctypes import *
import ctypes
from struct import *


def get_pid(name):
    for proc in psutil.process_iter():
        if str(name) in str(proc.name):
            print(name, 'pid = ', proc.pid)
            return proc.pid


def read_memory():
    PID = get_pid('inthezone.exe')
    PROCESS_QUERY_INFORMATION = 0x0400
    PROCESS_VM_READ = 0x0010

    process = windll.kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, PID)
    readprocess = windll.kernel32.ReadProcessMemory
    rdbuf = c_uint()
    bytread = c_ulong(0)
    base = 0x010FEBA4

    for addr in range(base, base + 1): # range(base, base + 11):
        try:
            if readprocess(process, ctypes.c_void_p(addr), ctypes.byref(rdbuf), ctypes.sizeof(rdbuf),
                           ctypes.byref(bytread)):
                print("Raw memory: ", str(hex(addr)), rdbuf.value)
                mem_bytes = [(rdbuf.value >> i) & 0xff for i in [0, 8, 16, 24]]
                print("Bytes: ", mem_bytes)
                print("As string[4]: ", "".join(map(chr, mem_bytes)))
        except:
            None
