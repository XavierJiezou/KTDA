import json
from glob import glob
# 读取JSON文件内容
import json

def json_to_markdown(json_file_path, md_file_path):

    classes_mapping = {
        "0":"Clear",
        "1":"Cloud Shadow",
        "2":"Thin Cloud",
        "3":"Cloud"
    }
    # 读取JSON文件内容
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # 创建一个Markdown文件
    with open(md_file_path, 'w') as f:
        # 写入标题
        f.write("# Class Evaluation Metrics\n\n")
        
        # 写入表格头
        f.write("| Class | IoU    | Acc    | Dice   | Fscore | Precision | Recall  |\n")
        f.write("|-------|--------|--------|--------|--------|-----------|---------|\n")
        
        # 写入每一行数据
        for item in data:
            f.write(f"| {classes_mapping[item['Class']]} | {item['IoU']:.2f} | {item['Acc']:.2f} | {item['Dice']:.2f} | {item['Fscore']:.2f} | {item['Precision']:.2f} | {item['Recall']:.2f} |\n")

    print(f"Markdown file has been saved to {md_file_path}.")

jsons = glob("eval_result/cloud/*.json")

for json_file in jsons:
    md_file_path = json_file.replace(".json", ".md")
    json_to_markdown(json_file, md_file_path)
