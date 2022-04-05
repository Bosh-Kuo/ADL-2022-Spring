import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
import os
import sys


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        help="Directory to the dataset.",
        default="./data/train.json",
    )
    parser.add_argument(
        "--output_dir",
        help="Directory to save the processed file.",
        default="./data/multiple_choice_train.json",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # 建立資料夾與 json 檔
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    # 原資料路徑
    data_dir = Path(args.data_dir)
    # 讀入資料
    list_data = json.loads(data_dir.read_text())
    # 將 List 轉成 dict 儲存
    json_data = {"data": list_data}
    json.dump(json_data, open(args.output_dir, 'w'),indent=2, ensure_ascii=False)  # ensure_ascii=False 包含中文
