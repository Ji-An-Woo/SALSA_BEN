import json
import re
import argparse
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import seaborn as sns


def configure_font():
    candidate_fonts = [
        ("Malgun Gothic", Path(r"C:/Windows/Fonts/malgun.ttf")),
        ("Malgun Gothic", Path(r"C:/Windows/Fonts/malgunbd.ttf")),
        ("NanumGothic", Path(r"C:/Windows/Fonts/NanumGothic.ttf")),
        ("NanumGothic", Path(r"C:/Windows/Fonts/NanumGothicBold.ttf")),
        ("AppleGothic", Path("/System/Library/Fonts/AppleGothic.ttf")),
    ]

    chosen = None
    chosen_path = None
    for font_name, font_path in candidate_fonts:
        if font_path.exists():
            try:
                fm.fontManager.addfont(str(font_path))
            except Exception:
                pass
            chosen = font_name
            chosen_path = font_path
            break

    if chosen is None:
        chosen = 'DejaVu Sans'

    plt.rc('font', family=chosen)
    plt.rcParams['font.family'] = [chosen]
    plt.rcParams['font.sans-serif'] = [chosen]
    plt.rcParams['font.serif'] = [chosen]
    plt.rcParams['font.monospace'] = [chosen]
    plt.rcParams['axes.unicode_minus'] = False

    if chosen_path and chosen_path.exists():
        return fm.FontProperties(fname=str(chosen_path))
    return fm.FontProperties(family=chosen)


FONT_PROP = configure_font()

ROOT = Path(__file__).resolve().parent
DEFAULT_LOG_PATH = Path(r"C:\checkpoint\user\dumped\final_test\zgm2ws92ej\train.log")
OUTPUT_DIR = ROOT / 'graphs'
OUTPUT_DIR.mkdir(exist_ok=True)


def parse_recovery_metrics(log_path: Path) -> pd.DataFrame:
    if not log_path.exists():
        raise FileNotFoundError(f"로그 파일을 찾을 수 없습니다: {log_path}")

    pattern = re.compile(r"\{.*\}")
    rows: List[Dict] = []

    with log_path.open('r', encoding='utf-8') as f:
        for line in f:
            if '"recover/matched"' not in line:
                continue
            match = pattern.search(line)
            if not match:
                continue
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                continue

            rows.append(
                {
                    'epoch': data.get('recover/epoch', 0),
                    'loss': data.get('recover/loss', float('nan')),
                    'acc1': data.get('recover/acc1', float('nan')),
                    'acc5': data.get('recover/acc5', float('nan')),
                    'matched': bool(data.get('recover/matched', False)),
                }
            )

    if not rows:
        raise ValueError('로그에서 복구 메트릭을 찾지 못했습니다.')

    df = pd.DataFrame(rows)
    df.sort_values('epoch', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def create_recovery_line_plot(df: pd.DataFrame, output_path: Path):
    df['success_flag'] = df['matched'].astype(int)
    df['cumulative_success'] = df['success_flag'].cumsum()

    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(8, 5))

    plt.plot(
        df['epoch'],
        df['cumulative_success'],
        marker='o',
        linewidth=2,
        color='#8e44ad',
        label='누적 비밀키 복구 횟수',
    )

    plt.scatter(df['epoch'], df['success_flag'], color='#e74c3c', s=80, label='해당 Epoch 성공 여부 (0/1)')

    for _, row in df.iterrows():
        status = '성공' if row['success_flag'] else '실패'
    plt.text(row['epoch'], row['cumulative_success'] + 0.05, f"{int(row['cumulative_success'])}",
                 ha='center', va='bottom', fontsize=10, color='#2c3e50', fontweight='bold', fontproperties=FONT_PROP)
    plt.text(row['epoch'], row['success_flag'] + 0.05, status,
                 ha='center', va='bottom', fontsize=9, color='#e74c3c', fontweight='bold', fontproperties=FONT_PROP)

    plt.xlabel('Epoch', fontsize=12, fontproperties=FONT_PROP)
    plt.ylabel('횟수 / 성공 여부', fontsize=12, fontproperties=FONT_PROP)
    plt.title('SALSA 비밀키 복구 누적 추이', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
    plt.xticks(df['epoch'])
    plt.ylim(0, df['cumulative_success'].max() + 1)
    plt.legend(prop=FONT_PROP)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return df[['epoch', 'success_flag', 'cumulative_success', 'loss', 'acc1', 'acc5']]


def safe_tag(tag: str | None) -> str | None:
    if not tag:
        return None
    return re.sub(r'[^0-9A-Za-z._-]+', '_', tag)


def main(log_path: Path = DEFAULT_LOG_PATH, tag: str | None = None, output_dir: Path = OUTPUT_DIR):
    log_path = Path(log_path)
    output_dir = Path(output_dir)
    if not log_path.exists():
        raise FileNotFoundError(f'로그 파일을 찾을 수 없습니다: {log_path}')

    tag = safe_tag(tag)

    print(f'📄 복구 로그 파싱 중: {log_path}')
    df = parse_recovery_metrics(log_path)

    output_dir.mkdir(exist_ok=True, parents=True)

    img_name = f'secret_recovery_line_{tag}.png' if tag else 'secret_recovery_line.png'
    csv_name = f'secret_recovery_summary_{tag}.csv' if tag else 'secret_recovery_summary.csv'

    output_path = output_dir / img_name
    print('📈 비밀키 복구 누적 그래프 생성 중...')
    summary_df = create_recovery_line_plot(df, output_path)
    print(f'✅ 그래프 저장 완료: {output_path}')

    csv_path = output_dir / csv_name
    summary_df.to_csv(csv_path, index=False)
    print(f'📝 지표 데이터 저장 완료: {csv_path}')


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='SALSA 비밀키 복구 그래프를 생성합니다.')
    parser.add_argument('--log', type=Path, default=DEFAULT_LOG_PATH, help='recover 로그가 포함된 train.log 경로')
    parser.add_argument('--tag', type=str, default=None, help='출력 파일에 추가할 태그')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR, help='그래프 및 CSV 출력 디렉터리')
    return parser


if __name__ == '__main__':
    args = build_argparser().parse_args()
    main(log_path=args.log, tag=args.tag, output_dir=args.output_dir)
