# fix_import.py
import fileinput
import sys

# 修复 basicsr 的导入
file_path = "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/basicsr/data/degradations.py"

with fileinput.FileInput(file_path, inplace=True) as file:
    for line in file:
        if "from torchvision.transforms.functional_tensor import" in line:
            line = line.replace("functional_tensor", "functional")
        sys.stdout.write(line)

print("修复完成！")