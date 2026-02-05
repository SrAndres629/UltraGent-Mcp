import os
import re
from pathlib import Path

def clean_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        new_lines = []
        modified = False
        for line in lines:
            # Match patterns like "197: ", "213 | ", "213 | >>> "
            cleaned = re.sub(r"^\d+\s*[:|]\s*(>>>\s*)?", "", line)
            if cleaned != line:
                modified = True
            new_lines.append(cleaned)
        
        if modified:
            print(f"CLEANED: {file_path}")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            return True
    except Exception as e:
        print(f"ERROR cleaning {file_path}: {e}")
    return False

def main():
    # Targets
    targets = [
        r'C:\Users\acord\AppData\Local\Programs\Python\Python313\Lib\site-packages\openhands',
        r'c:\Users\acord\OneDrive\Desktop\Biblioteca MCP\Ultragent'
    ]
    
    extensions = {'.py', '.j2', '.txt', '.sh', '.md', '.json'}
    
    count = 0
    for target in targets:
        print(f"Scanning target: {target}")
        for root, dirs, files in os.walk(target):
            # Skip noise dirs
            if any(x in root for x in ('.git', '__pycache__', 'node_modules', '.ai')):
                continue
                
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    if clean_file(os.path.join(root, file)):
                        count += 1
    
    print(f"\nFinished. Total files cleaned: {count}")

if __name__ == "__main__":
    main()
