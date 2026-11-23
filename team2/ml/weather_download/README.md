# 공항 날씨 원본 데이터 (IEM ASOS/AWOS METAR)

> 저장소 안내: 이 폴더에는 다운로드·검증 **코드**와 `station_map.csv`만 있습니다. 아래에 나오는 `asos_raw/`, `download_log.csv`, `coverage_report.csv`, `meta/` 등은 데이터 폴더(`$AIRLINE_DATA_ROOT/weather/`)에 만들어지며 저장소에는 올리지 않습니다.

BTS On-Time Performance 2020~2025 운항 데이터에 붙일 출발·도착 공항의 날씨 원본입니다.
이 폴더에는 **원본 다운로드와 검증 결과만** 들어 있습니다. 가공, 피처 생성, BTS 조인은 하지 않았습니다.

## 출처
- Iowa Environmental Mesonet (IEM), Iowa State University: ASOS/AWOS 공항 관측 METAR 아카이브
  - 데이터셋 안내: https://mesonet.agron.iastate.edu/info/datasets/
  - 다운로드 API: https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py ([도움말](https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?help))
  - 관측소 메타데이터: `https://mesonet.agron.iastate.edu/geojson/network/{NET}.geojson` (캐시: `meta/iem_networks/`)
- 공항 좌표·FAA/ICAO 코드 교차확인: OurAirports `airports.csv` (https://ourairports.com/data/, 캐시: `meta/ourairports_airports.csv`)
- 다운로드 일시: 2026-11-17 (UTC)

## 기간
- UTC 기준 **2019-12-31 00:00 ~ 2026-01-02 00:00** (끝 시각 미포함). BTS가 현지시각 기준이라 앞뒤로 하루씩 여유를 뒀습니다.
- 파일은 공항 × 1년 단위입니다. `{IATA}_2020`은 2019-12-31부터, `{IATA}_2025`는 2026-01-01까지 들어 있고, 나머지는 해당 연도 1/1 ~ 12/31입니다.
- 시각은 모두 UTC입니다(`tz=Etc/UTC`). `valid` 컬럼이 관측 시각입니다.

## 요청 파라미터
```
station={iem_station}
data=tmpf,dwpf,relh,feel,drct,sknt,gust,peak_wind_gust,vsby,skyc1,skyc2,skyc3,skyc4,
     skyl1,skyl2,skyl3,skyl4,p01i,wxcodes,ice_accretion_1hr,ice_accretion_3hr,
     ice_accretion_6hr,snowdepth,alti,mslp,metar
sts=..., ets=..., tz=Etc/UTC, format=onlycomma, latlon=yes, elev=yes,
missing=empty, trace=T, report_type=3,4
```
- 요청한 26개 컬럼은 모두 IEM 도움말에 있는 이름이라 **뺀 항목은 없습니다.**
- `format`은 `onlycomma`와 `tdf`만 있고 `comma`는 없어서 `onlycomma`를 썼습니다.
- `report_type`: 3 = 정시 관측(routine), 4 = 특별 관측(SPECI). 1분 단위 HFMETAR(1)는 넣지 않았습니다.
  - 예: ATL 2024년은 정시 관측(:52분) 8,776건에 특별 관측 등 1,380건이 더 있습니다.
- 출력 컬럼: `station, valid, lon, lat, elevation(m), ` + 위 26개.
- 미량 강수는 `T`, 결측은 빈 문자열입니다.
- IEM에 없어 받을 수 없었던 항목은 없습니다. 쓰지 않은 항목으로 `peak_wind_drct`, `peak_wind_time`이 있습니다.

## 파일
| 경로 | 내용 |
|---|---|
| `station_map.csv` | BTS IATA → IEM 관측소 매핑 (386행) |
| `asos_raw/{IATA}/{IATA}_{YYYY}.csv.gz` | IEM 응답 원본을 그대로 gzip으로 압축 |
| `download_log.csv` | 공항·연도별 다운로드 결과 (rows, status, error) |
| `coverage_report.csv` | 공항·연도별 행 수, 관측이 1건 이상 있는 UTC 시간의 비율(%) |
| `meta/` | IEM 네트워크 geojson, OurAirports, 검증 요약(`validation_summary.json`) |
| `taf_raw/`, `taf_download_log.csv` | (선택 작업) TAF 공항 예보 원본 — 아래 참고 |
| `scripts/` | `build_station_map.py` → `download_asos.py` → `validate_asos.py`, `download_taf.py` |

다시 실행하는 방법(`weather/`에서). 이미 받은 파일은 건너뜁니다.
```
python -I -X utf8 scripts/build_station_map.py
python -I -X utf8 scripts/download_asos.py
python -I -X utf8 scripts/validate_asos.py
```
서버 부담을 줄이기 위해 요청은 한 번에 1개씩 순차로 보내고, 요청 사이에 1.5초 쉽니다. IEM은 IP당 1초 간격 제한이 있습니다. 실패하면 5초부터 간격을 두 배씩 늘려 최대 5번 다시 시도합니다.

## 매핑 결과 (386개)
| match_type | 공항 수 |
|---|---|
| exact (IEM ID = IATA) | 342 |
| faa_code (FAA LID 또는 ICAO가 IATA와 다름) | 44 |
| nearest (근처 관측소로 대체) | 0 |
| unmatched (실패) | 0 |

모든 공항이 **공항 구내 관측소**에 매칭됐습니다. 관측소는 공항 좌표에서 최대 3.3km(DEN) 안에 있습니다.

**IATA와 FAA 코드가 다른 본토 공항**
| IATA | IEM | 비고 |
|---|---|---|
| AZA | IWA | Mesa Gateway |
| USA | JQF | Concord-Padgett |
| SCE | UNV | State College |
| HHH | HXD | Hilton Head |
| MQT | SAW | Marquette Sawyer |
| FCA | GPI | Glacier Park Intl |
| CLD | CRQ | McClellan-Palomar |
| GUF | JKA | Jack Edwards (Gulf Shores) |
| BKG | BBG | Branson |
| YUM | NYL | IEM의 `YUM`은 기록이 2007년에 끝남 → 현재 ID `NYL` 사용 |
| PBI | DJT | **2026년 공항 코드가 PBI → DJT로 변경**되어 IEM ID도 `DJT` |

**IEM이 4글자 ICAO ID를 쓰는 지역**
- 알래스카(AK_ASOS): ANC→PANC, FAI→PAFA, JNU→PAJN, KTN→PAKT, SIT→PASI, ADQ→PADQ, PSG→PAPG, WRG→PAWG, YAK→PAYA, CDV→PACV, BET→PABE, SCC→PASC, OME→PAOM, OTZ→PAOT, BRW→PABR, AKN→PAKN, DLG→PADL, ADK→PADK, GST→PAGS, CDB→PACD
- 하와이(HI_ASOS): HNL→PHNL, OGG→PHOG, KOA→PHKO, LIH→PHLI, ITO→PHTO
- 푸에르토리코·버진아일랜드: SJU→TJSJ, BQN→TJBQ, PSE→TJPS, STT→TIST, STX→TISX
- 괌·사이판·파고파고: GUM→PGUM (GU_ASOS), SPN→PGSN (MP__ASOS), PPG→NSTU (AS_ASOS)

## 커버리지 요약
- 파일 2,314 / 2,316개, 전체 **26,121,449행**, 압축 용량 **약 0.95GiB** (1,014,691,033 bytes)
- 공항·연도별 커버리지(관측이 1건 이상 있는 UTC 시간의 비율) 평균: **98.95%**
- 다운로드 실패 2건: **PSE 2020, PSE 2021** — IEM에 해당 기간 데이터가 없습니다(헤더만 응답).
- 커버리지 80% 미만인 공항·연도 18건:

| 공항 | 연도 (커버리지 %) | 공백 형태 (데이터로 확인) |
|---|---|---|
| PSE | 2020 (0), 2021 (0), 2022 (13.95) | 관측이 2022-11에야 시작 |
| BQN | 2020~2025 (64.6~70.4) | 매일 UTC 05~09시(현지 01~05시)에 관측이 거의 없고, 01~04시도 드묾 → 야간 공백 |
| WYS | 2021~2025 (46.1~53.7) | 5월~11월 초에만 관측 (2023 기준) → 계절 운영 |
| HDN | 2025 (67.71) | 2025년 4월 말~6월 관측 없음 |
| GST | 2024 (66.02) | 1~2월과 11~12월에 관측이 거의 없음 |
| GUF | 2020 (70.72) | 2020-09 중순 이후 관측 없음 |
| IFP | 2025 (74.67) | 2025년 5~8월에 관측 희박 |

## 컬럼별 빈 값 비율 (전체 행 기준)
"현상이 있을 때만 보고되는" 컬럼은 빈 값이 곧 '현상 없음'을 뜻하므로, 결측과 구분해서 다뤄야 합니다.

| 컬럼 | 빈 값 % | 해석 |
|---|---|---|
| ice_accretion_3hr / 6hr / 1hr | 99.87 / 99.81 / 99.21 | 착빙이 있을 때만 보고 |
| snowdepth | 99.85 | 적설 보고가 드묾 |
| skyc4, skyl4 / skyc3, skyl3 / skyc2, skyl2 | 99.03 / 86.05 / 68.33 | 구름층이 그만큼 없음 |
| peak_wind_gust | 93.67 | 최대 순간풍 기록이 있을 때만 |
| gust | 85.84 | 돌풍이 있을 때만 |
| wxcodes | 78.64 | 일기 현상 없음 |
| skyl1 | 36.09 | CLR/SKC(맑음)일 때는 고도가 없음 |
| mslp | 27.99 | SPECI와 일부 AWOS에는 SLP가 없음 |
| drct | 3.34 | 무풍·VRB(변동 풍향) |
| 그 밖의 컬럼 (tmpf, dwpf, relh, feel, sknt, vsby, skyc1, p01i, alti) | < 1 | |
| metar | 0.00 | |

## TAF (선택 작업)
- IEM TAF 아카이브: https://mesonet.agron.iastate.edu/cgi-bin/request/taf.py
  - `sts`/`ets`는 예보 **발표 시각** 범위입니다. 출력은 예보 구간별로 한 행입니다(`valid` = 발표 시각, `fx_valid`~`fx_valid_end` = 예보가 적용되는 시간, `ftype`, `raw` 등).
- 같은 공항·기간·연도 단위로 `taf_raw/{IATA}/{IATA}_{YYYY}.csv.gz`에 저장합니다.
  - TAF를 발표하지 않는 공항·연도는 `taf_download_log.csv`에 `status=no_taf`로 기록합니다.
