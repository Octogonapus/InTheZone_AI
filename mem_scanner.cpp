#include <string>
#include <Windows.h>
#include <iostream>

class MemScanner {
public:
  MemScanner(int pid):
    m_pid(pid) {}

  virtual ~MemScanner() {}

  void scan(const std::string& text) {
    DWORD address = 0x100579C;
  	int value = 0;
  	DWORD pid;
  	HWND hwnd = FindWindow(NULL,L"Minesweeper");

  	if (!hwnd) {
    	cout <<"Window not found!\n";
    	cin.get();
  	} else {
    	GetWindowThreadProcessId(hwnd,&pid);
    	HANDLE phandle = OpenProcess(PROCESS_VM_READ,0,pid);
  		if (!phandle) {
    		cout <<"Could not get handle!\n";
    		cin.get();
  		} else {
    		while(1) {
      		ReadProcessMemory(phandle,(void*)address,&value,sizeof(value),0);
      		cout << value << "\n";
      		Sleep(1000);
    		}
    		return 0;
  		}
  	}
  }
private:
  DWORD m_pid;
};
