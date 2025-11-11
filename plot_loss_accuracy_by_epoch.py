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
    """Windows 환경에서 한글이 깨지지 않도록 폰트를 강제 등록."""
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


def parse_training_metrics(log_path: Path) -> pd.DataFrame:
    """SALSA train.log 파일에서 epoch별 학습 메트릭을 추출."""
    if not log_path.exists():
        raise FileNotFoundError(f"로그 파일을 찾을 수 없습니다: {log_path}")

    rows: List[Dict] = []
    pattern = re.compile(r"\{.*\}")

    with log_path.open('r', encoding='utf-8') as f:
        for line in f:
            if '"train/acc1"' not in line:
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
                    'epoch': data.get('train/epoch', 0),
                    'step': data.get('train/step', 0),
                    'loss': data.get('train/loss', float('nan')),
                    'acc1': data.get('train/acc1', float('nan')),
                    'acc5': data.get('train/acc5', float('nan')),
                }
            )

    if not rows:
        raise ValueError('로그에서 학습 메트릭을 찾지 못했습니다.')

    df = pd.DataFrame(rows)
    df.sort_values(['epoch', 'step'], inplace=True)
    return df.reset_index(drop=True)


def aggregate_by_epoch(df: pd.DataFrame) -> pd.DataFrame:
    """에폭별 평균 Loss와 Accuracy를 계산."""
    grouped = (
        df.groupby('epoch')[['loss', 'acc1', 'acc5']]
        .mean()
        .reset_index()
        .sort_values('epoch')
    )
    # Accuracy를 백분율로 변환
    grouped['acc1_pct'] = grouped['acc1'] * 100
    grouped['acc5_pct'] = grouped['acc5'] * 100
    return grouped


def plot_loss_curve(epoch_df: pd.DataFrame, output_path: Path):
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_df['epoch'], epoch_df['loss'], color='#e74c3c', marker='o', linewidth=2, label='Loss')
    plt.xlabel('Epoch', fontsize=12, fontproperties=FONT_PROP)
    plt.ylabel('Loss', fontsize=12, fontproperties=FONT_PROP)
    plt.title('SALSA Training Loss by Epoch', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
    plt.xticks(epoch_df['epoch'])
    plt.legend(prop=FONT_PROP)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_accuracy_curve(epoch_df: pd.DataFrame, output_path: Path):
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(8, 5))
    plt.plot(
        epoch_df['epoch'],
        epoch_df['acc1_pct'],
        color='#f39c12',
        marker='s',
        linewidth=2,
        label='Top-1 Accuracy (%)',
    )
    plt.xlabel('Epoch', fontsize=12, fontproperties=FONT_PROP)
    plt.ylabel('Accuracy (%)', fontsize=12, fontproperties=FONT_PROP)
    plt.title('SALSA Training Accuracy by Epoch', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
    plt.xticks(epoch_df['epoch'])
    plt.legend(prop=FONT_PROP)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


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

    print(f'📄 로그 파싱 중: {log_path}')
    df = parse_training_metrics(log_path)
    epoch_df = aggregate_by_epoch(df)

    output_dir.mkdir(exist_ok=True, parents=True)

    loss_name = f'loss_by_epoch_{tag}.png' if tag else 'loss_by_epoch.png'
    acc_name = f'accuracy_by_epoch_{tag}.png' if tag else 'accuracy_by_epoch.png'
    csv_name = f'train_epoch_metrics_{tag}.csv' if tag else 'train_epoch_metrics.csv'

    loss_path = output_dir / loss_name
    acc_path = output_dir / acc_name

    print('📉 Loss 그래프 생성 중...')
    plot_loss_curve(epoch_df, loss_path)
    print(f'✅ Loss 그래프 저장 완료: {loss_path}')

    print('📈 Accuracy 그래프 생성 중...')
    plot_accuracy_curve(epoch_df, acc_path)
    print(f'✅ Accuracy 그래프 저장 완료: {acc_path}')

    csv_path = output_dir / csv_name
    epoch_df[['epoch', 'loss', 'acc1_pct', 'acc5_pct']].to_csv(csv_path, index=False)
    print(f'📝 에폭별 요약 CSV 저장 완료: {csv_path}')


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='SALSA 학습 로그에서 Loss/Accuracy 그래프를 생성합니다.')
    parser.add_argument('--log', type=Path, default=DEFAULT_LOG_PATH, help='train.log 파일 경로')
    parser.add_argument('--tag', type=str, default=None, help='출력 파일 이름에 사용할 태그')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR, help='그래프 및 CSV 출력 디렉터리')
    return parser


if __name__ == '__main__':
    args = build_argparser().parse_args()
    main(log_path=args.log, tag=args.tag, output_dir=args.output_dir)
