# SALSA_BEN: SALSA Attack on Maclaurin-Obfuscated LWE

이 프로젝트는 **Maclaurin 다항식 변환을 통한 LWE(Learning With Errors) 방어**에 대한 SALSA 공격의 효과를 평가합니다.

## 📋 프로젝트 구조

```
SALSA_BEN/
├── src/                          # 핵심 데이터 생성 코드
│   ├── data_gen_obfuscate_fixed.py   # 데이터셋 생성 (Baseline + Idea)
│   ├── utils.py                      # 공통 유틸리티
│   └── run_salsa_connected.py        # SALSA 실행 스크립트
├── external/
│   └── LWE-benchmarking/             # 외부 SALSA 구현 (별도 클론 필요)
├── data/
│   └── precomputed/                  # 생성된 데이터셋 저장
│       ├── baseline_n10/
│       ├── baseline_n30/
│       ├── idea_n10/
│       └── idea_n30/
├── results/
│   └── salsa_runs/                   # SALSA 실행 결과
├── idea/                             # 평가 및 분석 코드
│   ├── evaluate_and_plot.py          # 결과 평가
│   ├── run_salsa_connected.py        # SALSA 실행 (심링크)
│   └── utils.py                      # 유틸리티 (심링크)
└── README.md
```

## 🔑 핵심 개념

### Baseline (원본 s)
```
생성: s → gen_lwe_samples → (A, b = A·s + e)
평가: SALSA 예측값 vs 원본 s
기대: ~95% 복구 (공격 성공)
```

### Idea (Maclaurin 변환된 s')
```
생성: s → obfuscate_maclaurin → s' = c₁·s + c₃·s³ + c₅·s⁵ (mod q)
      → gen_lwe_samples → (A2, b2 = A2·s' + e2)

평가1: SALSA 예측값 vs 원본 s
기대: ~0% 복구 (방어 성공!)

평가2: SALSA 예측값 vs s'
기대: ~70-80% 복구 (SALSA 여전히 강력)
```

## 📊 전체 실행 흐름

### 1️⃣ 데이터셋 생성 (약 1분)

```bash
cd /Users/seungmin/Desktop/SALSA_BEN
python3 src/data_gen_obfuscate_fixed.py
```

**수행 작업:**
- `baseline_n10/`, `baseline_n30/`: 원본 비밀키 기반 LWE 샘플
- `idea_n10/`, `idea_n30/`: Maclaurin 변환된 비밀키 기반 LWE 샘플

**생성 파일:**
```
data/precomputed/{baseline,idea}_n{10,30}/
├── A.npy              # LWE 행렬 (m × n)
├── b.npy              # LWE 결과 벡터 (m,)
├── e.npy              # 에러 벡터 (m,)
└── meta.json          # 메타데이터
    ├── "s": [...]     # 원본 비밀키
    ├── "s_prime": [...] # Maclaurin 변환된 비밀키 (idea만)
    ├── "coeffs": {...} # 다항식 계수 (idea만)
    └── "params": {...} # LWE 파라미터
```

### 2️⃣ 외부 SALSA 저장소 클론

```bash
cd external
git clone https://github.com/[SALSA_REPO].git LWE-benchmarking
cd ..
```

**필수 사항:**
- `external/LWE-benchmarking/src/salsa/train_and_recover.py` 존재 확인

### 3️⃣ SALSA 모델 학습 및 비밀키 복구 (약 20-30분)

```bash
cd idea
python3 run_salsa_connected.py
```

**수행 작업:**
- 각 데이터셋(baseline_n10, baseline_n30, idea_n10, idea_n30)에 대해:
  1. SALSA 모델 학습 (5 에포크)
  2. 비밀키 예측 (첫 번째 추측)
  3. 결과 저장

**생성 파일:**
```
results/salsa_runs/{baseline,idea}_n{10,30}/
├── run_meta.json             # 실행 메타데이터
├── run_stdout.json           # 실행 로그
└── predicted_secrets.json    # SALSA의 예측된 비밀키
    └── "guesses": [[...]]    # n차원 배열
```

**모델 파라미터:**
```python
enc_emb_dim: 512         # 임베딩 차원
n_enc_heads: 4           # 어텐션 헤드 수
n_enc_layers: 2          # 인코더 레이어 수
max_epoch: 5             # 학습 에포크
train_batch_size: 32     # 학습 배치 크기
val_batch_size: 64       # 검증 배치 크기
```

### 4️⃣ 결과 평가 (약 1초)

```bash
python3 idea/evaluate_and_plot.py
```

**수행 작업:**
- 각 폴더의 SALSA 예측값을 두 가지 방식으로 평가:
  - **vs 원본 s**: 모든 데이터셋 (방어 효과 측정)
  - **vs s'**: idea 데이터셋만 (SALSA 수렴 능력 측정)

**생성 파일:**
```
results/salsa_runs/salsa_summary.csv
  ┌─────────────┬──────────────────┬──────────────────┬──────────────────┬──────────────────┐
  │ folder      │ exact_vs_s       │ bitwise_vs_s     │ exact_vs_s_prime │ bitwise_vs_s_pri │
  ├─────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┤
  │baseline_n10 │ 1                │ 0.95             │ None             │ None             │
  │baseline_n30 │ 1                │ 0.92             │ None             │ None             │
  │idea_n10     │ 0                │ 0.0              │ 1                │ 0.75             │
  │idea_n30     │ 0                │ 0.0              │ 1                │ 0.70             │
  └─────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┘

results/salsa_runs/salsa_summary.json  # CSV와 동일한 내용 (JSON 형식)
```

## 📈 결과 해석

### 메트릭 설명

| 메트릭 | 의미 | 범위 | 예시 |
|--------|------|------|------|
| **exact** | 모든 비트가 정확히 일치 | 0 또는 1 | 1 = 완벽 복구, 0 = 부분 실패 |
| **bitwise** | 일치하는 비트의 비율 | 0.0 ~ 1.0 | 0.95 = 95% 유사 |

### 해석 예시

```
Baseline n=10:
  exact_vs_s = 1       → SALSA가 s를 정확히 복구함 ✓
  bitwise_vs_s = 0.95  → 95% 유사도

Idea n=10:
  exact_vs_s = 0       → SALSA가 원본 s를 못 복구 (방어 성공!) ✓✓
  bitwise_vs_s = 0.0   → 원본과 완전히 무관
  exact_vs_s_prime = 1 → SALSA가 s'을 정확히 학습 (방어의 원리) ✓
  bitwise_vs_s_prime = 0.75 → s'과 75% 유사
```

## 🔧 커스터마이징

### 모델 차원 변경

`idea/run_salsa_connected.py` 라인 37:
```python
'--enc_emb_dim', str(flags.get('enc_emb_dim',512))  # 512 → 다른 값으로 변경
```

권장값:
- `128`: 빠른 실행, 낮은 정확도
- `256`: 균형잡힌 성능
- `512`: 현재 설정 (추천)
- `1024`: 높은 정확도, 느린 속도 (GPU 메모리 필요)

### 에포크 수 변경

`idea/run_salsa_connected.py` 라인 45:
```python
'--max_epoch', str(flags.get('epochs',5))  # 5 → 다른 값으로 변경
```

### LWE 파라미터 변경

`src/data_gen_obfuscate_fixed.py`에서 직접 수정:
```python
datasets = [
    {'n': 10, 'q': 842779, 'm': 500, 'sigma': 3.0, 'hamming': 3, ...},
    {'n': 30, 'q': 842779, 'm': 2000, 'sigma': 3.0, 'hamming': 3, ...},
]
```

## 📝 실행 체크리스트

- [ ] 데이터셋 생성: `python3 src/data_gen_obfuscate_fixed.py`
- [ ] 외부 저장소 클론: `git clone ... external/LWE-benchmarking`
- [ ] SALSA 실행: `cd idea && python3 run_salsa_connected.py`
- [ ] 결과 평가: `python3 idea/evaluate_and_plot.py`
- [ ] 결과 확인: `cat results/salsa_runs/salsa_summary.csv`

## �� 트러블슈팅

### "train_and_recover.py not found" 에러
```bash
# 외부 저장소 경로 확인
ls -la external/LWE-benchmarking/src/salsa/
# 없으면 저장소 다시 클론
```

### GPU 메모리 부족
```python
# run_salsa_connected.py에서 enc_emb_dim 값 감소
'--enc_emb_dim', str(flags.get('enc_emb_dim',256))  # 512 → 256
```

### 데이터 파일 없음
```bash
# data/precomputed 폴더 확인
ls -la data/precomputed/
# 없으면 Step 1 다시 실행
python3 src/data_gen_obfuscate_fixed.py
```

## 📚 주요 코드 파일

### `src/data_gen_obfuscate_fixed.py`
- `sample_secret(n, hamming, seed)`: 이진 비밀키 생성
- `circular_wrap(conv_result, n, q)`: 환 위의 합성곱 래핑
- `obfuscate_maclaurin(s, q, degrees, coeffs, coeff_choices)`: Maclaurin 변환
- `gen_lwe_samples(n, q, m, sigma, s, seed)`: LWE 샘플 생성

### `idea/run_salsa_connected.py`
- `build_cmd(data_path, exp_name, seed)`: SALSA 실행 명령 생성
- 메인 루프: 모든 데이터셋에 대해 SALSA 실행

### `idea/evaluate_and_plot.py`
- `load_json(p)`: JSON 파일 로드
- `compute_recovery(true_s, pred_s)`: 복구율 계산 (exact, bitwise)
- 메인 루프: 결과 수집 및 평가

## 📌 주의사항

1. **데이터셋 생성은 한 번만**: 재생성하면 기존 데이터 덮어씀
2. **SALSA는 GPU 강력 권장**: CPU만으로는 매우 느림
3. **결과 해석**: baseline과 idea 결과를 함께 봐야 방어 효과를 알 수 있음
4. **메타데이터 중요**: `meta.json`에 s와 s'가 저장되어 있어야 평가 가능

## 🎯 예상 결과

```
전체 실행 시간: ~30분 (GPU 포함)

baseline_n10: exact_vs_s ≈ 1.0, bitwise_vs_s ≈ 0.95
baseline_n30: exact_vs_s ≈ 1.0, bitwise_vs_s ≈ 0.92
idea_n10:    exact_vs_s ≈ 0.0, bitwise_vs_s ≈ 0.0,
             exact_vs_s_prime ≈ 1.0, bitwise_vs_s_prime ≈ 0.75
idea_n30:    exact_vs_s ≈ 0.0, bitwise_vs_s ≈ 0.0,
             exact_vs_s_prime ≈ 1.0, bitwise_vs_s_prime ≈ 0.70
```

### 해석
- **Baseline**: SALSA가 원본 비밀키를 거의 완벽하게 복구 (95%+)
- **Idea**: SALSA가 원본 비밀키는 못 복구(0%)하지만, Maclaurin 변환된 비밀키는 70-75% 복구
  - 이는 **방어가 작동**함을 의미 (원본을 숨김)
  - 하지만 **SALSA는 여전히 강력** (변환된 버전은 학습)

---

**문의 및 버그 보고**: Issues 탭 참고
