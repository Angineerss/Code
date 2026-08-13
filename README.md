# Binance tick → 불균형 바 → CUSUM → 트리플 베리어

MSFT 분봉/달러바 설계는 폐기했습니다. 학습용 샘플은 **바이낸스 현물 하루치 aggTrades(틱)** 를 받아 **불균형 바**로 샘플링하고, **CUSUM**으로 이벤트를 고른 뒤 트리플 베리어 메타 라벨을 붙입니다.

## 확정 스펙

### 원본 데이터

| 항목 | 결정 |
| --- | --- |
| 거래소 | Binance spot |
| 심볼 | `BTCUSDT` (기본값, CLI로 변경) |
| 기간 | UTC 기준 **하루** (기본: Vision에 올라온 최신 일자) |
| 원본 | `aggTrades` (가격·시각이 같은 체결을 묶은 틱) |
| 가격 | 체결가 그대로. 주식 adjusted/분할 개념 없음 |
| 거래량 | base `qty`, 달러 흐름은 `price * qty` |
| 공격자 | `is_buyer_maker=True` → 적극 매도(`side=-1`), False → 적극 매수(`+1`) |
| 타임스탬프 저장 | UTC |
| 세션 필터 | 없음 (크립토 24/7). 주식 ET RTH 규칙은 적용하지 않음 |
| 결측 | 체결이 없으면 바를 만들지 않음 (forward-fill 없음) |

### 바 / 이벤트 / 라벨

| 단계 | 결정 |
| --- | --- |
| 바 | **달러 불균형 바**. 임계값 = `E[T] * clip(|2P[buy]-1|) * E[size]`. 바가 닫힐 때의 `|θ|/T`는 쓰지 않음 (임계값이 커지는 편향) |
| 바 밀도 | `E[T]=80`로 시작, `|2P-1|`를 `[0.05, 0.15]`로 clip, `E[T]`는 `[0.5, 2]×초기값`으로 제한, `max_ticks=4*E[T]` 강제 종료는 기대치 업데이트에서 제외 |
| Primary | 규칙 기반. CUSUM 방향 = `side ∈ {+1,-1}` |
| Primary 목표 | recall 우선 (precision은 Meta가 회수) |
| 이벤트 | 불균형 바 종가 경로에 대칭 CUSUM, `h = 0.25σ` (기본). `--event-mode every_bar`면 바마다 이벤트 |
| 트리플 베리어 | `pt=sl=1σ`, 수직장벽 `τ=20` 바, 경로는 바 high/low |
| Meta 타깃 | `y=1` 익절 선터치, `y=0` 손절·타임아웃·동시터치 |
| Meta 모델 | Random Forest (이 레포는 라벨까지) |

불균형 바 자체가 order-flow 불균형을 샘플링하므로, 차트 규칙만 쓰는 파이프라인과 입력 공간이 다릅니다. CUSUM은 그 바 시계열에서 **이벤트만** 고릅니다.

이전 설정은 바가 극단 불균형에서 닫힌 `|θ|/T`를 다음 임계값에 넣어서 바가 하루 수십 개로 줄었습니다. 미시구조 분석에는 정보 시계가 너무 성깁니다. 지금은 매수 확률 기반 기대 불균형 + clip + 최대 틱 수로 바를 촘촘히 유지합니다.

### 검증

- Research 구간 안에서 **CPCV**
- Train/test 라벨 구간이 겹치면 **Purge**, 테스트 종료 이후 **Embargo**
- Purge/Embargo **비율은 모델 호라이즌 확정 후**. 코드 기본 길이는 수직장벽 `τ` 바
- 무작위 shuffle 분할 없음

## 파이프라인

```text
Binance aggTrades (1 UTC day)
  → aggressor-signed ticks
  → dollar imbalance bars
  → CUSUM events (primary side)
  → triple-barrier meta labels
  → CPCV paths (purge + embargo)
```

## 실행

```bash
pip install -r requirements.txt
pytest
python -m src --symbol BTCUSDT --date 2024-01-15
python -m src --symbol BTCUSDT --date 2024-01-15 --event-mode every_bar
```

산출물 (`data/`):

- `{SYMBOL}_{day}_imbalance_bars.csv`
- `{SYMBOL}_{day}_labels.csv`
- `{SYMBOL}_{day}_summary.json`

하루치 BTCUSDT aggTrades zip은 수십 MB일 수 있습니다. 테스트는 네트워크 없이 합성 틱만 사용합니다.
