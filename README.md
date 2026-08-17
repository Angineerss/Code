# Binance tick → 달러 바 → 트리플 베리어

학습용 샘플은 **바이낸스 현물 하루치 aggTrades(틱)** 를 받아 **달러 바**로 시간을 자르고, 그 바 안의 달러 불균형으로 방향을 정한 뒤 트리플 베리어 메타 라벨을 붙입니다.

## 연구 가설 (잠금)

**거래대금으로 바를 자른 뒤, 그 바의 달러 불균형이 셀 때 그 방향으로 베팅하는 것이 유리하다.**

시계와 방향 공식을 나누는 이유: 바를 `|θ| ≥ E[θ]`로 자르면, 닫힌 바는 이미 “불균형이 충분하다”는 뜻이라 primary `sign(θ)`가 같은 말을 한 번 더 하게 된다.

| | 본실험 (`--bar-type dollar`, 기본) | 대조군 (`--bar-type dollar_imbalance`) |
| --- | --- | --- |
| 정보 구조 (언제 자를지) | 거래대금이 T$에 닿으면 자른다 | `|θ|`가 E[θ]에 닿으면 자른다 (원래 샘플러) |
| Primary (어느 쪽) | `sign(θ)` = `sign(signed_flow)` | 동일 |
| Meta 게이트 | `|θ| ≥ E[θ]` (`require_strong_imbalance`) | 동일. 대조군은 대부분의 바가 이미 이 조건으로 닫힘 |
| 겹침 | 시계와 방향이 분리됨 | 시계와 방향이 같은 불균형 공식 |

운영 규칙:

- 정보 구조 = **달러 바**. 시드 `D = (어제까지 365 UTC일 일별 quote 평균) / 650`. T$는 달러로 닫힐 때마다 EWMA, `[0.5D, 2D]` 클립. E[θ]는 같은 바에서 계산만 하고 자르는 데는 안 씀
- Primary = **방향** = `sign(θ)`. θ, E[θ], `|2b-1|`, E[T], E[size], EWMA span=50은 예전 달러 불균형 수식 그대로
- **Meta 게이트:** `|θ| ≥ E[θ]` 인 바만 (`max_ticks`이면서 약한 θ는 폐기). `close_reason == imbalance`가 아님
- **Meta 피처:**
  - 세기: `flow_strength = |θ|/E[θ]`
  - 맥락: `tick_rel = tick_count/E[T]` (바 종료 시점 E[T], EWMA 갱신 전), `sigma` (베리어와 동일 EWM σ)
  - 미포함: 가격 확인 강도 `cusum_excess_ratio`, 원시 `tick_count`, `duration_s`, 장 시간대, 모멘텀
- CUSUM은 기본 경로가 아님 (`--event-mode cusum` 대조 실험용)
- 검증 = 트리플 베리어 메타 라벨 (`y_meta`). 인과(“가격이 taker 때문인가”)는 이 파이프라인의 범위 밖
- **대조군 EWMA/바는 본실험과 섞지 말 것.** `--out-dir`를 따로 쓴다 (기본값에 `bar_type`이 들어감)

## 확정 스펙

### 원본 데이터


| 항목       | 결정                                                                |
| -------- | ----------------------------------------------------------------- |
| 거래소      | Binance spot                                                      |
| 심볼       | `BTCUSDT` (기본값, CLI로 변경)                                          |
| 기간       | Vision 아카이브 **2017-08-17 ~ 2026-08-13** UTC. 워밍업 / IS / OOS는 아래 컷 |
| 원본       | `aggTrades` (가격·시각이 같은 체결을 묶은 틱)                                  |
| 연구 컷     | 아카이브 전체를 쓰되, D 시드용 1년 워밍업 후 IS → OOS                              |
| 가격       | 체결가 그대로. 주식 adjusted/분할 개념 없음                                     |
| 거래량      | base `qty`, 달러 흐름은 `price * qty`                                  |
| 공격자      | `is_buyer_maker=True` → 적극 매도(`side=-1`), False → 적극 매수(`+1`)     |
| 타임스탬프 저장 | UTC                                                               |
| 세션 필터    | 없음 (크립토 24/7)                                                     |
| 결측       | 체결이 없으면 바를 만들지 않음 (forward-fill 없음)                               |




### 바 / 이벤트 / 라벨


| 단계         | 결정                                                                                                                                                                                                                                                           |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 바          | **달러 바** (본실험). 시드 `D = (슬라이딩 365 UTC일 일별 quote 거래대금 평균) / 650`. 창은 **어제까지**. T$는 달러로 닫힐 때마다 EWMA, 그날 슬라이딩 D의 `[0.5D, 2D]`로만 클립. E[θ]는 기록만. `init_T = 20,000`, `max_ticks = 50,000`, `init_b = 0.5`. 첫날만 `init_T` 틱 워밍업(라벨 제외). 대조군은 `--bar-type dollar_imbalance` (닫힘 = `|θ| ≥ E[θ]`) |
| Primary    | 규칙 기반. `sign(signed_flow)` = `sign(θ)`. 바를 자르는 규칙이 아님. `--primary rule_cusum_sign`은 대조 실험용                                                                                                                   |
| Primary 목표 | recall 우선 (precision은 Meta가 회수)                                                                                                                                                                                                                              |
| 이벤트       | 달러 바 종가. `require_strong_imbalance=True`이면 `|θ| ≥ E[θ]`인 바만. `--event-mode cusum`은 대조 실험용                                                                                                                 |
| 트리플 베리어    | `pt=sl=2σ`, 수직장벽 **τ=30** (운영값. 1σ는 다음 바에 종료, 3σ는 ~24바가 필요. CPCV 로그로스 최솟값이 아님; `results/pt_sl_tau_cpcv_2018-08-17_2021-01-17.json`). 경로는 바 high/low |
| Meta 타깃    | `y=1` 익절 선터치, `y=0` 손절·타임아웃·동시터치                                                                                                                                                                                                                             |
| Meta 피처    | `flow_strength`, `tick_rel=tick_count/E[T]`, `sigma`. 가격 확인 강도(`cusum_excess_ratio`)·원시 tick·duration·시간대·모멘텀 제외 |
| Meta 모델    | Random Forest (이 레포는 라벨+피처까지; 학습기는 다음 단계)                                                                                                                                                                                                                                   |


바이낸스 `BTCUSDT`는 2017년에 상장되어 **2017-01-15 기준 1년은 존재하지 않습니다.** 기준일은 틱 데이터의 UTC 날짜입니다. 예: `--date 2024-01-16` → 평균 구간 `2023-01-16` ~ `2024-01-15` (고정 연도가 아니라 어제에서 끝나는 슬라이딩 365일).

### 검증

- 전체 시계를 시간 순으로만 분할. 무작위 shuffle 없음
- **Warmup** `2017-08-17` ~ `2018-08-16` (365 UTC일). D 시드(슬라이딩 365일) + EWMA만. 연구 라벨/하이퍼파라미터에 쓰지 않음
- **IS** `2018-08-17` ~ `2024-12-31` (2329 UTC일). 하이퍼파라미터·메타 학습·교차검증은 여기만
- **IS 안 학습 vs CV**: 별도 연도 holdout이 아니라 **CPCV**. 라벨 이벤트를 시간 순 **6개 contiguous 그룹**으로 나누고, 그중 **2개**를 CV-test로 쓰는 경로 `C(6,2)=15`. 각 경로의 나머지 그룹 = train (겹치면 **Purge**, 테스트 종료 후 **Embargo**). **정책 A (잠금):** Purge + Embargo = **1τ** 바씩 (`τ=30` → 각 30바; `purge_bars=embargo_bars=None`이면 `vertical_bars`를 따름). OOS는 CV에 넣지 않음
- **OOS** `2025-01-01` ~ `2026-08-13` (590 UTC일). 컷 확정 후 손대지 않음. 이후 공개분은 OOS 끝에만 붙임
- **OOS 가드:** 학습·구조화·튜닝에서 OOS는 코드가 거부 (`assert_learning_range` / `assert_not_oos_day`). **모든 모델·피처·임계값이 IS에서 확정된 뒤** 최종 성적만 `--allow-oos`
- **메타 학습 샘플:** `split==is`만. 워밍업은 EWMA/D 전용(메타 샘플 제외). 선택/MDA/하이퍼/메타 임계값은 **CPCV만** (`selection_method=cpcv_only`)
- **IS↔OOS 경계 (AFML, 정책 A = 1τ):** Purge = `t1`이 OOS 직전 1τ 창에 들어오면 제거. Embargo = 이벤트 시각이 OOS 직전 1τ 이내면 제거. `filter_meta_learning_samples`



## 파이프라인

```text
Binance aggTrades
  → aggressor-signed ticks
  → day 1: first init_T=20,000 ticks seed E[size], b stays 0.5 (not labeled)
  → later days: load previous ewma_state.json (skip warmup bar)
  → bars capped at max_ticks=50,000
  → D_seed = sliding 365d average daily quote notional ending yesterday / 650
  → dollar bars close on T$; E[θ] is recorded, not used to close
  → event = bar close with |θ| ≥ E[θ] (require_strong_imbalance)
  → primary side = sign(θ) = sign(signed dollar flow)
  → control clock: --bar-type dollar_imbalance (separate out-dir)
  → meta features: flow_strength, tick_rel, sigma
  → triple-barrier meta labels
  → CPCV
```



## 실행

```bash
pip install -r requirements.txt
pytest
# Vision aggTrades archive from listing (monthly zips + open-month dailies; ~58GB+)
python scripts/download_aggtrades_archive.py --data-dir data/aggtrades
# Learning range only (warmup bars → IS labels). OOS forbidden. CV = CPCV on IS.
python scripts/run_learning_range.py --skip-existing
# Control clock (original dollar-imbalance bars). Separate out-dir / EWMA.
python scripts/run_learning_range.py --bar-type dollar_imbalance --skip-existing
python scripts/summarize_is_cpcv.py --run-root data/runs/learning_dollar_2017-08-17_2024-12-31
python -m src --symbol BTCUSDT --date 2024-01-15
python -m src --symbol BTCUSDT --date 2024-01-16
python -m src --symbol BTCUSDT --date 2024-01-15 --bar-type dollar_imbalance
python -m src --symbol BTCUSDT --date 2024-01-15 --event-mode cusum
python scripts/compare_vertical_tau.py --run-root data/runs/<is_run>
# Same CPCV scripts work on a control run-root (do not mix bar types in one folder).
```

산출물 (`data/`):

- `aggtrades/monthly/{SYMBOL}-aggTrades-YYYY-MM.zip` — 상장 이후 월간 아카이브
- `aggtrades/daily/{SYMBOL}-aggTrades-YYYY-MM-DD.zip` — 미마감 월 일간
- `{SYMBOL}_{day}_bars.csv`
- `{SYMBOL}_{day}_labels.csv`
- `{SYMBOL}_{day}_ewma_state.json` (다음 날 EWMA 이어받기)
- `{SYMBOL}_{day}_summary.json`

