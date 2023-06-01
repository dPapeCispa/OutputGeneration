import os
from typing import List
import subprocess

def get_prompt_function_lines(prompt: str) -> List[int]:
    with open('temp.c', 'w') as f:
        f.write(prompt)
    cmd = "ctags -x --c-kinds=fp " + 'temp.c' 
    output = subprocess.getoutput(cmd)
    lines = output.splitlines()
    line_nums = list()
    for line in lines:
        if line.strip()!="":
            line = line.split(" ")
            line = list(filter(None, line))
            line_nums.append(int(line[2])-1)
    os.remove('temp.c')
    return line_nums
            
def extract_function_by_lines(output: str, line_nums: List[int]) -> List[str]:
    functions = list()
    split_output = output.split('\n')
    for line_num in line_nums:
        function = ""
        cnt_braket = 0
        found_start = False
        for i, line in enumerate(split_output):
            if(i >= line_num):
                function = function + line + '\n'
                if line.count("{") > 0:
                    found_start = True
                    cnt_braket += line.count("{")
                if line.count("}") > 0:
                    cnt_braket -= line.count("}")
                if cnt_braket == 0 and found_start == True:
                    functions.append(function)
                    break
        if(cnt_braket != 0):
            functions.append(None)
    return functions



def extract_c_function(decoded_outputs: List[str], prompt: str) -> List[str]:
    prompt_function_lines = get_prompt_function_lines(prompt)
    new_outputs = list()
    main_function = ["int main(){\n\t\n}"]
    for output in decoded_outputs:
        functions = extract_function_by_lines(output, prompt_function_lines)
        for function in functions:
            if(function is not None):
                new_outputs.append("\n".join(functions + main_function))
    return new_outputs
        