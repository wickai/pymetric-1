import re
import json


def extract_test_epoch_and_rngs(log_text):
    # 提取 test_epoch 行的信息，并解析 epoch, top1_err, top5_err
    test_epoch_pattern = re.compile(
        r'json_stats: \{"_type": "test_epoch".*?"epoch": "(\d+/\d+)",.*?"top1_err": ([\d\.]+), "top5_err": ([\d\.]+).*?\}'
    )
    test_epoch_matches = [
        {"epoch": match[0], "top1_err": float(match[1]), "top5_err": float(match[2])}
        for match in test_epoch_pattern.findall(log_text)
    ]

    # 提取 rngs
    rngs_pattern = re.compile(r"eval on random fix rngs:\[(.*?)\]")
    rngs_matches = rngs_pattern.findall(log_text)

    # 解析 rngs 为整数列表
    rngs_lists = [list(map(int, rngs.split(", "))) for rngs in rngs_matches]

    return test_epoch_matches, rngs_lists


def process_log_file(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        log_text = f.read()

    test_epoch_data, rngs_data = extract_test_epoch_and_rngs(log_text)

    output_data = {"test_epoch": test_epoch_data, "rngs": rngs_data}

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)


# 示例调用
input_log_file = (
    "useful_log/202502080957_supermodel_300sub-branch_eval_only.log"  # 替换为你的日志文件路径
)
output_json_file = "output_eval.json"
process_log_file(input_log_file, output_json_file)
