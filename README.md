# Binance tick → 불균형 바 → CUSUM → 트리플 베리어

학습용 샘플은 **바이낸스 현물 하루치 aggTrades(틱)** 를 받아 **달러 불균형 바**로 샘플링하고, **CUSUM**으로 이벤트를 고른 뒤 트리플 베리어 메타 라벨을 붙입니다.

## 확정 스펙

### 원본 데이터

| 항목 | 결정 |
| --- | --- |
| 거래소 | Binance spot |
| 심볼 | `BTCUSDT` (기본값, CLI로 변경) |
| 기간 | 전체 시계 **2024-01-01 ~ 2026-08-13** UTC (Vision aggTrades). IS/OOS는 아래 컷 |
| 원본 | `aggTrades` (가격·시각이 같은 체결을 묶은 틱). Vision 아카이브는 **2017-08-17**부터 수집 가능 |
| 연구 컷 | IS/OOS는 아래. 아카이브 전체와 연구 구간은 별개 |
| 가격 | 체결가 그대로. 주식 adjusted/분할 개념 없음 |
| 거래량 | base `qty`, 달러 흐름은 `price * qty` |
| 공격자 | `is_buyer_maker=True` → 적극 매도(`side=-1`), False → 적극 매수(`+1`) |
| 타임스탬프 저장 | UTC |
| 세션 필터 | 없음 (크립토 24/7) |
| 결측 | 체결이 없으면 바를 만들지 않음 (forward-fill 없음) |

### 바 / 이벤트 / 라벨

| 단계 | 결정 |
| --- | --- |
| 바 | **달러 불균형 바**. 시드 `D = (슬라이딩 365 UTC일 일별 quote 거래대금 평균) / 650`. 창은 **어제까지**. `E[θ]`는 불균형으로 닫힐 때마다 EWMA로 갱신하고, 다음 날 `ewma_state.json`에서 이어받음. 그날 슬라이딩 D의 `[0.5D, 2D]`로만 클립. `init_T = 20,000`, `max_ticks = 50,000`, `init_b = 0.5`. 첫날만 `init_T` 틱 워밍업(라벨 제외) |
| Primary | 규칙 기반. CUSUM과 별개. 기본은 이벤트 바의 체결 불균형 부호 `sign(signed_flow)`. `--primary rule_cusum_sign`이면 필터 방향을 1차로 쓸 수 있음 |
| Primary 목표 | recall 우선 (precision은 Meta가 회수) |
| 이벤트 필터 | 불균형 바 종가 경로에 대칭 CUSUM. 1차 **이전** 필터. `S±`는 AFML 식 그대로, 넘은 쪽만 0으로 리셋. `h = 1σ`. `--event-mode every_bar`면 필터 없이 바마다 이벤트 |
| 트리플 베리어 | `pt=sl=1σ`, 수직장벽 `τ=20` 바, 경로는 바 high/low |
| Meta 타깃 | `y=1` 익절 선터치, `y=0` 손절·타임아웃·동시터치 |
| Meta 모델 | Random Forest (이 레포는 라벨까지) |

바이낸스 `BTCUSDT`는 2017년에 상장되어 **2014-01-15 기준 1년은 존재하지 않습니다.** 기준일은 틱 데이터의 UTC 날짜입니다. 예: `--date 2024-01-16` → 평균 구간 `2023-01-16` ~ `2024-01-15` (고정 연도가 아니라 어제에서 끝나는 슬라이딩 365일).

### 검증

- 전체 시계를 시간 순으로만 분할. 무작위 shuffle 없음
- **IS** `2024-01-01` ~ `2025-12-31` (731 UTC일). 하이퍼파라미터·메타 학습·**CPCV**는 여기만
- IS 안은 별도 연도 holdout이 아니라 **CPCV**: 5그룹 중 2그룹 테스트, `C(5,2)=10` 경로. Train/test 라벨이 겹치면 **Purge**, 테스트 종료 이후 **Embargo**. 길이는 수직장벽 `τ=20` 바
- **OOS** `2026-01-01` ~ `2026-08-13` (225 UTC일, Vision 공개분). 컷 확정 후 손대지 않음. 이후 공개되는 2026일은 OOS 끝에만 붙임
- IS 라벨의 트리플 베리어 `t1`이 2026-01-01 이후면 IS에서 제외 (경계 Purge)

## 파이프라인

```text
Binance aggTrades
  → aggressor-signed ticks
  → day 1: first init_T=20,000 ticks seed E[size], b stays 0.5 (not labeled)
  → later days: load previous ewma_state.json (skip warmup bar)
  → bars capped at max_ticks=50,000
  → D_seed = sliding 365d average daily quote notional ending yesterday / 650
  → dollar imbalance bars; E[θ] EWMA-updates and continues across days
  → CUSUM filter (h = 1σ, reset crossed side only) picks event times
  → primary side = sign(signed dollar flow) on those bars
  → triple-barrier meta labels
  → CPCV
```

## 실행

```bash
pip install -r requirements.txt
pytest
# Vision aggTrades archive from listing (monthly zips + open-month dailies; ~58GB+)
python scripts/download_aggtrades_archive.py --data-dir data/aggtrades
python -m src --symbol BTCUSDT --date 2024-01-15
python -m src --symbol BTCUSDT --date 2024-01-16
python -m src --symbol BTCUSDT --date 2024-01-15 --event-mode every_bar
```

산출물 (`data/`):

- `aggtrades/monthly/{SYMBOL}-aggTrades-YYYY-MM.zip` — 상장 이후 월간 아카이브
- `aggtrades/daily/{SYMBOL}-aggTrades-YYYY-MM-DD.zip` — 미마감 월 일간
- `{SYMBOL}_{day}_bars.csv`
- `{SYMBOL}_{day}_labels.csv`
- `{SYMBOL}_{day}_ewma_state.json` (다음 날 EWMA 이어받기)
- `{SYMBOL}_{day}_summary.json`
