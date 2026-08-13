"""将 development 证据决策写入 final 配置（脚本化，防手改）。

读取 outputs/tables/window_decision.json（窗口）与
neural_search_decision.json（M3/M5 超参），更新配置的
window.primary 与 models.M3/M5 对应字段；随后 freeze_final.py
生成冻结清单。任何选择都有 development 证据文件可追溯。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import yaml

from spyvar.config import content_sha256


def main() -> None:
    p = argparse.ArgumentParser(description="把 development 决策写入最终配置")
    p.add_argument("--config", default="configs/final.yaml")
    p.add_argument("--out-root", default="outputs")
    args = p.parse_args()

    out_root = Path(args.out_root)
    dev_dir = out_root / "development"
    win_dec = json.loads((dev_dir / "window_decision.json").read_text())
    search_dec = json.loads((dev_dir / "neural_search_decision.json").read_text())
    cfg_path = Path(args.config)
    raw = yaml.safe_load(cfg_path.read_text())

    raw["window"]["primary"] = int(win_dec["chosen_window"])
    for model_id, dec in search_dec.items():
        for key, value in dec["best"].items():
            raw["models"][model_id][key] = value
    cfg_path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"配置更新: primary window={raw['window']['primary']}")
    for model_id, dec in search_dec.items():
        print(f"  {model_id} 超参: {dec['best']}")
    print(f"config SHA256: {content_sha256(cfg_path.read_text())}")


if __name__ == "__main__":
    main()
