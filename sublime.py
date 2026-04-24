import os
import sys

# Structure: {offset: (original_byte, patch_byte)}
PATCH_MAP = {
    0x001C6CE4: (0x01, 0x00), # Example: change 01 to 00
    0x001C6CFB: (0x01, 0x00),
}

def patch_exe(input_file):
    if not os.path.exists(input_file):
        print(f"[-] Error: File '{input_file}' not found.")
        return

    backup_file = input_file + ".bak"
    
    try:
        # Create a backup first
        if not os.path.exists(backup_file):
            import shutil
            shutil.copy2(input_file, backup_file)
            print(f"[+] Backup created: {backup_file}")

        with open(input_file, "r+b") as f:
            for offset, (original, patch) in PATCH_MAP.items():
                f.seek(offset)
                current_byte = f.read(1)
                
                if current_byte == bytes([original]):
                    f.seek(offset)
                    f.write(bytes([patch]))
                    print(f"[+] Offset {hex(offset)}: Patched!")
                elif current_byte == bytes([patch]):
                    print(f"[*] Offset {hex(offset)}: Already patched.")
                else:
                    print(f"[!] Offset {hex(offset)}: Unexpected byte ({current_byte.hex()}). Patch aborted.")
                    
    except PermissionError:
        print("[-] Error: Permission denied. Please run as Administrator.")
    except Exception as e:
        print(f"[-] An error occurred: {e}")

if __name__ == "__main__":
    target = r"C:\Program Files\Sublime Text 3\sublime_text.exe"
    # Allow override via command line
    file_to_patch = sys.argv[1] if len(sys.argv) > 1 else target
    patch_exe(file_to_patch)
        
 # python .\sublime.py "C:\Program Files\Sublime Text\sublime_text.exe"
 # python -c "importar shutil; fn='sublime_text.exe'; off=0x46B80; old=bytes.fromhex('0F B6 51 05 83 F2 01'); new=bytes.fromhex('C6 41 05 01 B2 00 90'); shutil.copy2(fn, fn+'.old'); f=open(fn,'r+b'); f.seek(off); d=f.read(len(old)); (d==old) e (f.seek(off),f.write(new),print('[+] Patch aplicado')) ou print('[!] Bytes diferentes, não corrigidos');
 # 0F B6 51 05 83 F2 01 with C6 41 05 01 B2 00 90