# Binance tick → 달러 바 → 트리플 베리어

학습용 샘플은 **바이낸스 현물 하루치 aggTrades(틱)** 를 받아 **달러 바**로 시간을 자르고, 그 바 안의 달러 불균형으로 방향을 정한 뒤 트리플 베리어 메타 라벨을 붙입니다.

코드 주석 태그: **`[선정]`** = 대화에서 이유를 단 것. **`[임시값]`** = 아직 이유 없음.

**이름:** 문서 `daily_T$` · 말 **어제 조각** · 식 `D = 어제 대금 / 650`. 매일 UTC일마다 다시 계산하는 그날 T$ 기준이다. 파이프라인 전체의 단 한 번 초기값이 아니다. 클립 식은 `[0.5D, 2D]`.

## 선정 vs 임시값

**선정**

| # | 항목 | 이유 |
| --- | --- | --- |
| 6 | `BTCUSDT`, `aggTrades` | 장기간 틱을 받기 유리. 규칙 파악을 위해 aggTrades |
| 7 | 워밍업 컷 | 어제 조각(`daily_T$`)은 어제면 된다. 1년 컷은 IS:OOS 달력(8:2)을 재학습 전까지 유지 |
| 8 | IS · 8:2 | 학습/홀드아웃 8:2가 가장 흔한 방식. CPCV로 학습/CV. 하이퍼·피처는 CV에서만 |
| 9 · 28 | OOS | 백테스트 제외하고 학습에 OOS 안 씀. purge·embargo |
| 10 | `bar_type=dollar` | 시간 바는 거래를 대표하지 않음 (AFML). 대금으로 자름 |
| 11 | divisor 650 | 방법 B. `daily_T$` = 어제 조각 = 어제 대금/650. 더 빨리 사려고 650 |
| 12 | lookback 1일 | 연간 대금 점프가 커서 어제 조각은 어제 하루 |
| 14 | T$ 클립 `[0.5D, 2D]` | 유지. 오늘이 어제보다 세면 바가 더 나오게. 연도 성장은 어제 조각이 이미 반영 |
| 15 · 16 | `max_ticks`, `init_T` | 장치/첫 바. **숫자 자체는 임시값.** 바 만든 뒤 틱이 너무 길거나 동떨어지지 않게 다시 정함 |
| 15* | 대조군 틱 상한 `E[T]×2.5` | `|θ|`가 E[θ]에 안 닿으면 강제 마감. **숫자 2.5는 임시값** |
| 17 | `init_b=0.5` | 대조군. 매수 확률을 반반으로 시작. `|2b-1|`이 0에서 출발 |
| 18 | `every_bar` | 1차 모델 recall을 키운다. 바를 미리 걸러 내지 않음 |
| 19 | `sign(θ)` | 그 바 달러 불균형의 방향 |
| 24 | y 규칙 익절만 1 | **서로 다른 바**에서 익절만 먼저 닿을 때 `y=1`. 같은 바에서 양쪽이면 순서를 몰라 `y=0` |
| 25 | 경로 high/low | 그 바가 벽에 닿았는지. 같은 바 안 고가·저가 시각은 없음 |
| 26 | CPCV 6×2 | AFML 책 예시. 학습 vs CV 칸 수 `C(6,2)=15` |
| 30 | 로그로스 | AFML 권장. 메타가 낸 익절 확률 채점. 입력이 아니라 CPCV 결과값 |
| 31 | 대조군 E[θ] 식 | AFML. `E[θ_T] ≈ E[T] × |2b − 1| × E[size]`. 본실험 닫힘·세기에 안 씀 |
| 32 | `|2b-1|` 클립 `[0.05, 0.15]` | `b=0.5`면 E[θ]=0이라 바로 닫힘. 바닥을 둬서 기대가 사라지지 않게 |

**T$ 클립 `[0.5D, 2D]` 유지 `[선정]`**

- 연도 단위 대금 성장은 어제 조각(`daily_T$`)이 이미 반영한다. 클립이 그걸 막는 게 아니다.
- 달러 바는 거래대금이 많으면 바가 많아야 한다. 윗벽(2D)이 있으면 오늘이 어제보다 셀 때 T$가 따라가지 못해 바가 더 늘어난다.
- 클립을 빼면 T$가 오늘 바 크기를 따라가서 하루 바 개수가 비슷해진다. 원하던 것과 반대다.
- 단기 추세를 T$가 빨리 따라가게 클립을 빼는 선택은, 바를 늘리려는 쪽과 반대라 하지 않는다.

**대조군 E[θ] (`dollar_imbalance`) `[선정]`** — 본실험 닫힘·세기에는 안 쓴다. 원래 샘플러를 AFML 식으로 유지한다. EWMA 폴더는 본실험과 섞지 않는다.

- **17 `init_b=0.5`:** `b`는 적극 매수 체결 비율이다. 0.5는 시작을 매수/매도 반반으로 둔다는 뜻이다. 그러면 `|2b − 1| = 0`에서 출발한다.
- **31 E[θ] 식 (AFML):** 한 바의 불균형 θ는 각 체결의 (매수 +1, 매도 −1) × 그 체결 대금의 합이다. 바가 닫히는 문턱은

  `E[θ_T] ≈ E[T] × |2b − 1| × E[size]`

  틱만 세는 불균형이면 한 체결 크기 `E[size] = 1`이라 `E[θ_T] ≈ E[T] × |2b − 1|`이 된다. 대조군은 달러라 한 체결 평균 대금 `E[size]`를 곱한다.

  - `E[T]`: 한 바에 들어갈 틱 수. 바가 닫힐 때마다 그 바의 틱 수로 EWMA
  - `b`: 매수 비율. 바가 닫힐 때마다 EWMA. 시작은 `init_b`
  - `|2b − 1|`: 반반에서 얼마나 치우쳤는지. 반반이면 0, 전부 매수(또는 매도)면 1
  - `E[size]`: 한 체결의 평균 대금. 바가 닫힐 때마다 EWMA

  닫힘: `|θ| ≥ E[θ]`.
- **32 `|2b-1|` 클립 `[0.05, 0.15]`:** `init_b=0.5`면 `|2b − 1| = 0` → `E[θ] = 0` → 첫 체결에 바로 닫힌다. 바닥 0.05를 둬서 기대 불균형이 사라지지 않게 한다. 천장 0.15는 기대가 너무 커져 `|θ|`가 문턱에 안 닿는 날을 줄인다.
- **15* `E[T] × 2.5` 장치 `[선정]`:** 닫힘의 본규칙은 `|θ| ≥ E[θ]`이다. 사고팔이가 섞이면 `|θ|`가 문턱에 안 닿아 틱이 계속 쌓인다. 그때 “지금 한 바로 보는 틱 수(`E[T]`)의 몇 배”가 되면 강제 마감한다. 전역 상한과도 같이 쓴다: `min(max_ticks, max(E[T]×배수, init_T))`. **배수 2.5는 임시값.** 역할만 있으면 2여도 3이어도 같다. 본실험 달러 바는 이 배수를 쓰지 않고 `max_ticks`만 본다.

**임시값** (이유 미작성): `sigma` (피처·벽 넓이 같은 하나, 길이 50), EWMA 50, `pt`/`sl`/`τ`, RF 크기, 거래소 외 UTC·세션 필터 등. `max_ticks`·`init_T`·대조군 틱 배수(2.5)의 **숫자**도 여기.

**채점 (결과값):** 로그로스 `[선정]` (AFML 권장). 메타가 낸 익절 확률이 실제와 맞았는지 본다. `flow_strength`·`sigma`는 입력이고, 로그로스는 CPCV가 계산하는 점수다. 바를 자르는 규칙이 아님.

## 방법 B · divisor

시계 조각 크기(`daily_T$`, 식 `D = 어제 대금 / divisor`, 말 어제 조각)는 CPCV가 아니라 **바 모양**으로 골랐다. 같은 틱을 하루 한 번만 읽고 후보 `100, 200, 400, 650, 1000`으로 달러 바를 잘랐다. 라벨·CPCV·OOS 없음.

- 구간: IS 안 다섯 달 (2019-01, 2020-03, 2021-05, 2023-06, 2024-03). 채점 154 UTC일. 창마다 앞 7일은 EWMA만.
- 스크립트: `scripts/compare_divisor_bars.py`. 숫자: `results/divisor_bar_screen.json`.
- 본 것: 한 바 길이, `flow_strength`가 0이나 1에만 몰리는지, 같은 달 조용한 날 vs 바쁜 날의 바 개수. T$가 `[0.5D, 2D]` 벽(어제 조각 기준)에 붙는 비율도 같이 봄.

| divisor | 하루 바 중앙 | 한 바 길이 중앙 | 세기 중앙 | 세기 0.05 아래 | T$가 2D 벽 |
| --- | --- | --- | --- | --- | --- |
| 100 | 84 | 11분 | 0.09 | 28% | 10% |
| 200 | 157 | 6분 | 0.11 | 24% | 14% |
| 400 | 275 | 3.6분 | 0.13 | 21% | 24% |
| **650** | **389** | **2.5분** | **0.15** | **18%** | **38%** |
| 1000 | 553 | 1.8분 | 0.17 | 16% | 53% |

4시간짜리 바는 후보 전부 없음. 같은 달 안에서 대금이 약 2.8배일 때 바 개수는 약 2.3배이고, 이 배율은 divisor가 달라도 거의 같다.

가운데는 400과 650. 100은 바가 굵고 세기가 약하다. 1000은 조각이 작고 T$ 중앙값이 이미 윗벽이다. **둘 중 더 빨리 사려고 650** `[선정]`. `[0.5D, 2D]` 클립은 **유지** `[선정]` (오늘이 어제보다 세면 바가 더 나오게).

## 연구 가설

**달러 바를 자른 뒤, 그 바의 달러 불균형이 셀 때 그 방향으로 투자한다.**

시계와 방향 공식을 나누는 이유: 바를 `|θ| ≥ E[θ]`로 자르면, 닫힌 바는 이미 “불균형이 충분하다”는 뜻이라 primary `sign(θ)`가 같은 말을 한 번 더 하게 된다.

| | 본실험 (`--bar-type dollar`, 기본) `[선정]` | 대조군 (`--bar-type dollar_imbalance`). E[θ] 식·`init_b`·`|2b-1|` 클립·틱 상한 장치 `[선정]` (배수 2.5는 임시값) |
| --- | --- | --- |
| 정보 구조 (언제 자를지) | 거래대금이 T$에 닿으면 자른다 | `|θ|`가 E[θ]에 닿으면 자른다 (원래 샘플러) |
| Primary (어느 쪽) | `sign(θ)` = `sign(signed_flow)` `[선정]`. 약한 θ도 남김 `[선정]` | 동일 |
| 세기 | 메타 피처 `flow_strength = |θ|/바대금` `[선정]`. Primary가 자르지 않음 `[선정]` | 시계가 이미 `|θ| ≥ E[θ]`로 닫혀 세기가 바에 묶임 |
| 겹침 | 시계와 방향이 분리됨 | 시계와 방향이 같은 불균형 공식 |

운영 규칙:

- 정보 구조 = **달러 바** `[선정]`. 시드 `daily_T$` (`D = 어제 UTC일 일별 quote / 650`, 어제 조각) (창·650 `[선정]`). T$ EWMA `[임시값]`. `[0.5D, 2D]` 클립 `[선정]` (오늘이 어제보다 세면 바가 더 나오게). 세기 `|θ|/그 바의 대금` `[선정]`
- Primary = `sign(θ)` `[선정]`. `require_strong_imbalance=False` `[선정]` (AFML recall)
- 이벤트 = `every_bar` `[선정]`. 바를 미리 걸러 내지 않아 1차 recall을 키운다. CUSUM은 대조 `[임시값]`
- **Meta 피처**: `flow_strength` `[선정]`, `sigma` `[임시값]`
- **대조군** = AFML `E[θ_T] ≈ E[T] × |2b − 1| × E[size]`로 자른다 `[선정]`. `init_b=0.5`, `|2b-1|` 클립, 틱 상한 장치 `[선정]` (배수 2.5는 임시값). EWMA/바는 본실험과 섞지 말 것. `--out-dir`를 따로 쓴다

## 스펙

아래 숫자는 위 **`[선정]` / `[임시값]`** 을 따른다. 선정 표에 없는 숫자는 임시값이다.

### 원본 데이터


| 항목       | 결정                                                                |
| -------- | ----------------------------------------------------------------- |
| 거래소      | Binance spot                                                      |
| 심볼       | `BTCUSDT` — 장기간 실시간 틱을 받기 유리 (기본값, CLI로 변경)                     |
| 기간       | Vision 아카이브 **2017-08-17 ~ 2026-08-13** UTC. 워밍업 / IS / OOS는 아래 컷 |
| 원본       | `aggTrades` — 가격·시각이 같은 체결을 묶어 두어 공격자 규칙을 읽기 좋음                 |
| 연구 컷     | 아카이브 전체를 쓰되, 어제 조각(`daily_T$`) 시드용 1년 워밍업 후 IS → OOS                              |
| 가격       | 체결가 그대로. 주식 adjusted/분할 개념 없음                                     |
| 거래량      | base `qty`, 달러 흐름은 `price * qty`                                  |
| 공격자      | `is_buyer_maker=True` → 적극 매도(`side=-1`), False → 적극 매수(`+1`)     |
| 타임스탬프 저장 | UTC                                                               |
| 세션 필터    | 없음 (크립토 24/7)                                                     |
| 결측       | 체결이 없으면 바를 만들지 않음 (forward-fill 없음)                               |




### 바 / 이벤트 / 라벨


| 단계         | 결정                                                                                                                                                                                                                                                           |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 바          | **달러 바** (본실험). 시간 바는 시장 거래를 잘 대표하지 않아 거래대금으로 자른다 (AFML). 시드 `daily_T$` (`D = 어제 UTC일 일별 quote / 650`, 어제 조각) `[선정]` (방법 B). T$는 달러로 닫힐 때마다 EWMA `[임시값]`, 그날 `D`의 `[0.5D, 2D]`로만 클립 `[선정]` (오늘이 어제보다 세면 바가 더 나오게. 연도 성장은 어제 조각이 이미 반영). 기록 `threshold` = `|2b-1| × 그 바 대금`. `max_ticks`·`init_T`는 바를 만든 뒤 틱이 너무 길거나 첫 바와 동떨어지지 않게 **다시 정할 값** (지금은 `max_ticks=50,000`, `init_T=20,000`). 첫날만 `init_T` 틱 워밍업(라벨 제외). 대조군은 `--bar-type dollar_imbalance` (닫힘 = `|θ| ≥ E[θ]`, AFML `E[θ_T] ≈ E[T] × |2b-1| × E[size]`, `init_b=0.5`, `|2b-1|` 클립 `[0.05, 0.15]`) `[선정]`. 틱 상한 장치 `min(max_ticks, E[T]×배수)` `[선정]` (배수 2.5는 임시값) |
| Primary    | `sign(signed_flow)` = `sign(θ)` `[선정]`. 바를 자르는 규칙이 아님 |
| Primary 목표 | recall 우선 (precision은 Meta가 회수)                                                                                                                                                                                                                              |
| 이벤트       | 달러 바 종가 `every_bar` `[선정]` (1차 recall). Primary는 `sign(θ)`만. `--event-mode cusum`은 대조                                                                                                                 |
| 트리플 베리어    | `pt=sl=1σ`, 수직장벽 **τ=20** `[임시값]`. 경로는 바 high/low `[선정]` (그 바가 벽에 닿았는지. 같은 바 안 순서는 없음). Purge/Embargo = 1τ (각 20바, 길이는 `[임시값]`) |
| Meta 타깃    | `y=1` `[선정]`: **서로 다른 바**에서 익절만 먼저 닿음. `y=0`: 손절, 시간 초과, **같은 바**에서 익절·손절 둘 다 (바 안 순서는 없음) |
| Meta 피처    | `flow_strength` (`|θ|/바대금`) `[선정]`. `sigma` `[임시값]` |
| Meta 모델    | Random Forest. 그루 수·깊이·잎 크기는 **초기값** (200, 6, 10). 아직 미정, CPCV로 나중에 고름 |
| 채점         | 로그로스 `[선정]` (AFML 권장). 메타가 익절 확률을 맞게 냈는지. CPCV 결과값. 바를 자르는 규칙 아님 |


바이낸스 `BTCUSDT`는 2017년에 상장되어 상장 당일은 어제가 없습니다. 기준일은 틱의 UTC 날짜입니다. 예: `--date 2024-01-16` → D의 창은 `2024-01-15` 하루.

### 검증

- 전체 시계를 시간 순으로만 분할. 무작위 shuffle 없음
- **Warmup** `2017-08-17` ~ `2018-08-16` (365 UTC일). 어제 조각(`daily_T$`, `D = 어제 대금 / 650`)은 어제면 된다. 상장 당일만 어제가 없어 부트스트랩한다. 1년 워밍업 컷은 IS:OOS 달력(~8:2)을 바꾸지 않으려고 재학습 전까지 유지
- **IS** `2018-08-17` ~ `2024-12-31` (2329 UTC일). **OOS** `2025-01-01` ~ `2026-08-13` (590 UTC일). 비율 **2329 : 590 ≈ 8 : 2**. 학습/홀드아웃을 8:2로 나누는 가장 흔한 방식을 따른다. 학습 샘플은 IS만. OOS가 학습에 섞이지 않도록 purge + embargo (정책 A, 각 1τ). 학습 vs 교차검증은 **CPCV**로 나눈다. 하이퍼파라미터·메타 피처는 교차검증 경로에서만 정함
- **IS 안 학습 vs CV**: 별도 연도 holdout이 아니라 **CPCV**. 라벨 이벤트를 시간 순 **6개 contiguous 그룹**으로 나누고, 그중 **2개**를 CV-test로 쓰는 경로 `C(6,2)=15` `[선정]` (AFML 책 예시). 각 경로의 나머지 그룹 = train (겹치면 **Purge**, 테스트 종료 후 **Embargo**). **정책 A:** Purge + Embargo = **1τ** 바씩 (`τ=20` → 각 20바; `purge_bars=embargo_bars=None`이면 `vertical_bars`를 따름) `[임시값]` 길이. OOS는 CV에 넣지 않음
- **OOS** `2025-01-01` ~ `2026-08-13` (590 UTC일). **백테스트를 제외하고 학습에 OOS를 쓰지 않는다.** 모델 확정 후 성적만 `--allow-oos`. 이후 공개분은 OOS 끝에만 붙임
- **OOS 가드:** 학습·구조화·튜닝에서 OOS는 코드가 거부 (`assert_learning_range` / `assert_not_oos_day`)
- **메타 학습 샘플:** `split==is`만. 워밍업은 EWMA·어제 조각(`daily_T$`) 전용(메타 샘플 제외). 선택/MDA/하이퍼/메타 임계값은 **CPCV만** (`selection_method=cpcv_only`)
- **채점:** CPCV에서 메타의 익절 확률을 로그로스로 본다 `[선정]` (AFML 권장). 입력이 아니라 **결과값**. `scripts/compare_vertical_tau.py`, `scripts/compare_pt_sl.py`
- **IS↔OOS 경계 (AFML, 정책 A = 1τ):** Purge = `t1`이 OOS 직전 1τ 창에 들어오면 제거. Embargo = 이벤트 시각이 OOS 직전 1τ 이내면 제거. `filter_meta_learning_samples`



## 파이프라인

```text
Binance aggTrades
  → aggressor-signed ticks
  → day 1: first init_T=20,000 ticks seed E[size], b stays 0.5 (not labeled)
  → later days: load previous ewma_state.json (skip warmup bar)
  → bars capped at max_ticks=50,000
  → daily_T$ (D = yesterday's daily quote / 650, 어제 조각)
  → dollar bars close on T$; strength = |θ| / that bar's quote (no E[T])
  → event = every dollar-bar close (warmup excluded)
  → primary side = sign(θ); weak |θ| kept for recall
  → meta sees strength via flow_strength = |θ| / bar quote
  → control clock: --bar-type dollar_imbalance
      (AFML E[θ_T]≈E[T]×|2b-1|×E[size]; |2b-1| clip; E[T]×mult cap [선정], 2.5 is 임시값; separate out-dir)
  → meta features: flow_strength, sigma
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
python scripts/compare_divisor_bars.py
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

