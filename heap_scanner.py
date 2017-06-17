import psutil
from ctypes import *
import ctypes

byte_shifts = [0, 8, 16, 24, 36, 48, 60]


def get_pid(name):
    for proc in psutil.process_iter():
        if str(name) in str(proc.name):
            print(name, 'pid = ', proc.pid)
            return proc.pid


def string6_from_uint(value):
    return "".join(map(chr, [(value >> i) & 0xff for i in byte_shifts]))


def print_memory(address, value):
    print("Raw memory: ", str(hex(address)), value)
    mem_bytes = [(value >> i) & 0xff for i in byte_shifts]
    print("Bytes: ", mem_bytes)
    print("As string[4]: ", "".join(map(chr, mem_bytes)))


def scan_memory(text):
    text_sorted = sort(text)
    PID = get_pid('inthezone.exe')
    PROCESS_QUERY_INFORMATION = 0x0400
    PROCESS_VM_READ = 0x0010

    process = windll.kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, PID)
    readprocess = windll.kernel32.ReadProcessMemory
    rdbuf = c_ubyte()
    bytread = c_ulong(0)
    base = 0x00000000

    addr = base
    counter = 0
    mem_bytes = [0] * len(text)
    while True:
        try:
            if readprocess(process,
                           ctypes.c_void_p(addr),
                           ctypes.byref(rdbuf),
                           ctypes.sizeof(rdbuf),
                           ctypes.byref(bytread)):
                mem_bytes[counter % 6] = rdbuf.value
                if text_sorted == sorted("".join(map(chr, mem_bytes))):
                    print("Found", text, "at address", hex(addr))
                    return addr

            if addr % 100000 == 0:
                print("Progress: ", hex(addr))

            addr = addr + 1
            counter = counter + 1
        except Exception:
            print("Failure! Last addr: ", hex(addr))
            break
