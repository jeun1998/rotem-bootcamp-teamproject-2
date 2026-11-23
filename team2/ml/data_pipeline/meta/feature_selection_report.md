# 특징 선정 보고서 (v3)

원본 컬럼 분류 기준과 근거는 v1(`processed/feature_selection_report.md`)과 같고, v2·v3에서 바뀐 부분만 반영했습니다. 결측률은 원본 전체 기준(BTS 2020~2025, IEM 전체)입니다.

## 1. 원본 컬럼 분류

| Dataset | Column | Description | Missing Rate (%) | Category | Selection Reason |
|---|---|---|---|---|---|
| BTS | Year | 연도 | 0 | Excluded | FlightDate와 중복(파생 관계 100% 일치). 시간순 분할에서 연도는 상수 |
| BTS | Quarter | 분기(1-4) | 0 | Excluded | FlightDate에서 파생 가능, 1~3월 데이터셋에서는 상수 |
| BTS | Month | 월 | 0 | Excluded | FlightDate와 중복. 연 단위 분할(v2)에서는 파생값 month·season을 학습 특징으로 사용 |
| BTS | DayofMonth | 일 | 0 | Performance Candidate | day_of_month 로 사용 (월초·월말 운항 패턴 후보). FlightDate와 중복이므로 원본 대신 파생값 사용 |
| BTS | DayOfWeek | 요일(1=월 ... 7=일) | 0 | Essential | 요일별 운항량·지연 패턴. day_of_week, is_weekend 생성 |
| BTS | FlightDate | 운항일(현지, yyyy-mm-dd) | 0 | Essential | 운항일(현지). 시간순 분할, 현지→UTC 변환, 공휴일 거리 계산의 기준 |
| BTS | Reporting_Airline | 항공사 고유 코드(Unique Carrier Code) | 0 | Essential | 항공사 운항 특성(범주형 입력). 운항 전 확정 정보 |
| BTS | Flight_Number_Reporting_Airline | 편명 번호 | 0 | Essential | 항공편 식별 키(입력 특징 아님). 고유값이 많고 의미가 항공사마다 달라 입력에서 제외 |
| BTS | Tail_Number | 기체 등록번호 | 0.66 | Performance Candidate | 원본은 입력 제외. 당일 값은 결항 편에서 결측이 많아 누수(v2) → v3는 과거 참조일(작년 같은 요일·M-3월)의 같은 편 기체 배정으로만 tail_*_hist를 만든다. 당일 기준 값은 *_sameday로 병합본에만 보존 |
| BTS | OriginAirportID | 출발 공항 DOT ID | 0 | Excluded | Origin과 1:1 (공항 코드 변경 추적용). 기간 내 코드 변경이 없어 중복 |
| BTS | Origin | 출발 공항 코드(IATA) | 0 | Essential | 출발 공항: 기상 관측소 매칭·시간대 결정·범주형 입력 |
| BTS | OriginCityName | 출발 도시명 | 0 | Excluded | Origin과 중복 |
| BTS | OriginState | 출발 주 코드 | 0 | Excluded | Origin에 종속(공항→주). 중복 정보 |
| BTS | DestAirportID | 도착 공항 DOT ID | 0 | Excluded | Dest와 1:1, 중복 |
| BTS | Dest | 도착 공항 코드(IATA) | 0 | Essential | 도착 공항: 범주형 입력, 노선(route) 생성, 도착지 시간대 검증 |
| BTS | DestCityName | 도착 도시명 | 0 | Excluded | Dest와 중복 |
| BTS | DestState | 도착 주 코드 | 0 | Excluded | Dest에 종속, 중복 |
| BTS | CRSDepTime | 예정 출발 시각(현지 hhmm) | 0 | Essential | 예정 출발 현지 시각: UTC 변환, 예측 시점(3시간 전) 계산, 출발 시간대 특징 |
| BTS | DepTimeBlk | 예정 출발 시간대 블록(1시간) | 0 | Excluded | CRSDepTime에서 결정되는 1시간 블록(00~05시는 '0001-0559' 하나로 묶임) → dep_hour_local이 더 세밀 |
| BTS | CRSArrTime | 예정 도착 시각(현지 hhmm) | 0 | Performance Candidate | 예정 도착 현지 시각: arr_hour_local, 도착 공항 혼잡도, 시간대 교차검증 |
| BTS | ArrTimeBlk | 예정 도착 시간대 블록(1시간) | 0 | Excluded | CRSArrTime 블록, 중복 |
| BTS | CRSElapsedTime | 예정 비행 소요시간(분) | 0 | Performance Candidate | 예정 소요시간(사전 공개). Distance와 상관이 높지만 스케줄 여유(buffer)를 반영 |
| BTS | Distance | 공항 간 거리(mile) | 0 | Essential | 노선 거리(사전 확정). 장·단거리 운항 특성 |
| BTS | DistanceGroup | 거리 구간(250mile 단위) | 0 | Excluded | Distance의 250mile 구간화(파생 관계 일치), 중복 |
| BTS | DepTime | 실제 출발 시각(현지 hhmm) | 2.173 | Evaluation Only | 실제 출발 시각 = 사후 정보. 누수 |
| BTS | DepDelay | 출발 지연(분, 조기출발 음수) | 2.175 | Evaluation Only | Label 생성용(출발 지연 분). 입력 사용 시 직접 누수 |
| BTS | DepDelayMinutes | 출발 지연(분, 조기출발 0) | 2.175 | Evaluation Only | DepDelay의 음수 0 처리판, Label 검증용 |
| BTS | DepDel15 | 15분 이상 출발 지연 여부(1=예) | 2.175 | Evaluation Only | Label(15분 지연) 교차검증용 |
| BTS | DepartureDelayGroups | 출발 지연 구간(15분 단위) | 2.175 | Evaluation Only | DepDelay 구간화, 사후 정보 |
| BTS | TaxiOut | 택시아웃 시간(분) | 2.214 | Evaluation Only | 출발 후 확정되는 사후 정보 |
| BTS | WheelsOff | 이륙 시각(현지 hhmm) | 2.214 | Evaluation Only | 사후 정보 |
| BTS | WheelsOn | 착륙 시각(현지 hhmm) | 2.246 | Evaluation Only | 사후 정보 |
| BTS | TaxiIn | 택시인 시간(분) | 2.246 | Evaluation Only | 사후 정보 |
| BTS | ArrTime | 실제 도착 시각(현지 hhmm) | 2.246 | Evaluation Only | 사후 정보 |
| BTS | ArrDelay | 도착 지연(분, 조기도착 음수) | 2.461 | Evaluation Only | 사후 정보 (도착 지연 분석용) |
| BTS | ArrDelayMinutes | 도착 지연(분, 조기도착 0) | 2.461 | Evaluation Only | 사후 정보 |
| BTS | ArrDel15 | 15분 이상 도착 지연 여부(1=예) | 2.461 | Evaluation Only | 사후 정보. 도착 기준 레이블 label_arr(평가용) 생성에 사용 |
| BTS | ArrivalDelayGroups | 도착 지연 구간(15분 단위) | 2.461 | Evaluation Only | 사후 정보 |
| BTS | ActualElapsedTime | 실제 소요시간(분) | 2.461 | Evaluation Only | 사후 정보 |
| BTS | AirTime | 비행 시간(분) | 2.461 | Evaluation Only | 사후 정보 |
| BTS | Cancelled | 결항 여부(1=예) | 0 | Evaluation Only | Label 생성(Class 2) 전용. 입력 금지 |
| BTS | CancellationCode | 결항 사유(A=항공사, B=기상, C=NAS, D=보안) | 97.777 | Evaluation Only | 결항 사유(B=기상) 분석·검증용. 결항편에만 존재하는 사후 정보 |
| BTS | Diverted | 회항 여부(1=예) | 0 | Evaluation Only | 회항 여부(사후). 회항편은 출발했으므로 출발지연 기준으로 Label 부여, 분석 플래그로만 보존 |
| BTS | CarrierDelay | 항공사 원인 지연(분) | 81.247 | Evaluation Only | 지연 원인 분해(ArrDel15=1일 때만 존재). 사후 정보, 원인 분석용 |
| BTS | WeatherDelay | 기상 원인 지연(분) | 81.247 | Evaluation Only | 사후 정보, 기상 지연 분석용 |
| BTS | NASDelay | NAS 원인 지연(분) | 81.247 | Evaluation Only | 사후 정보 |
| BTS | SecurityDelay | 보안 원인 지연(분) | 81.247 | Evaluation Only | 사후 정보 |
| BTS | LateAircraftDelay | 선행 항공기 지연(분) | 81.247 | Evaluation Only | 사후 정보 (선행편 지연). 예측 시점에 부분적으로만 알 수 있어 이번 단계 제외 |
| IEM | station | IEM 관측소 ID | 0 | Essential | 관측소 ID: 출발 공항 매칭 키 |
| IEM | valid | 관측 시각(UTC) | 0 | Essential | 관측 시각(UTC): 예측 시점 이전 최신 관측 매칭 키 |
| IEM | lon | 관측소 경도 | 0 | Excluded | 관측소 고정 좌표. 공항 내에서 변하지 않음(공항 코드와 중복) |
| IEM | lat | 관측소 위도 | 0 | Excluded | 관측소 고정 좌표, 공항별 상수 |
| IEM | elevation | 관측소 고도(m) | 0 | Excluded | 관측소 고도, 공항별 상수(공항 범주형과 중복). 밀도고도 계산이 필요하면 재검토 |
| IEM | tmpf | 기온(°F) | 0.264 | Essential | 기온: 결빙·제빙 조건 판단의 핵심 |
| IEM | dwpf | 이슬점(°F) | 0.429 | Performance Candidate | 이슬점: temp_dew_spread(안개 가능성) 계산에 사용 |
| IEM | relh | 상대습도(%) | 0.429 | Performance Candidate | 상대습도: tmpf·dwpf에서 계산 가능하나 비선형 변환값으로 후보 |
| IEM | feel | 체감온도(°F, 열지수/풍속냉각) | 0.545 | Excluded | 체감온도: tmpf·sknt에서 결정되는 파생값, 운항과 직접 관련 낮음 |
| IEM | drct | 풍향(도, 진북 기준) | 3.342 | Performance Candidate | 풍향: sin/cos 주기 변환. 활주로 방향 정보가 없어 측풍은 계산 불가 |
| IEM | sknt | 풍속(knot) | 0.49 | Essential | 풍속: 강풍 운영 제한 |
| IEM | gust | 돌풍(knot) | 85.844 | Essential | 돌풍: 결측 86%이나 '돌풍 미보고'의 의미가 있음 → wind_max_kt, gust_reported로 사용 |
| IEM | peak_wind_gust | 최대 순간풍(knot, 지난 1시간 RMK PK WND) | 93.673 | Excluded | 결측 94%, 정보는 gust·max_wind_3h와 중복 |
| IEM | vsby | 시정(statute mile) | 0.321 | Essential | 시정: 저시정 시 이착륙 간격 증가. 비행 기상 등급 계산 |
| IEM | skyc1 | 1층 운량(CLR/FEW/SCT/BKN/OVC/VV) | 0.412 | Essential | 1층 운량: 실링(BKN/OVC/VV) 판단 |
| IEM | skyc2 | 2층 운량 | 68.329 | Performance Candidate | 2층 운량: ceiling_ft 계산에만 사용(원본 입력 제외) |
| IEM | skyc3 | 3층 운량 | 86.051 | Performance Candidate | 3층 운량: ceiling_ft 계산에만 사용 |
| IEM | skyc4 | 4층 운량 | 99.027 | Excluded | 결측 99%. ceiling 계산에는 포함하되 원본 입력 제외 |
| IEM | skyl1 | 1층 운저 고도(ft AGL) | 36.085 | Essential | 1층 운저 고도: 실링 계산 |
| IEM | skyl2 | 2층 운저 고도(ft) | 68.33 | Performance Candidate | ceiling_ft 계산에만 사용 |
| IEM | skyl3 | 3층 운저 고도(ft) | 86.051 | Performance Candidate | ceiling_ft 계산에만 사용 |
| IEM | skyl4 | 4층 운저 고도(ft) | 99.027 | Excluded | 결측 99%, ceiling 계산에만 포함 |
| IEM | p01i | 1시간 강수량(inch, 정시 관측 기준 누적; T=미량) | 0.261 | Essential | 1시간 강수량 |
| IEM | wxcodes | 현재 일기 코드(METAR) | 78.637 | Essential | 현재 일기 코드: 뇌우·눈·어는비·안개 플래그 생성 (결항 핵심 신호) |
| IEM | ice_accretion_1hr | 1시간 착빙(inch) | 99.212 | Excluded | 결측 99%(착빙 센서·사례 드묾). 어는 강수는 wx_freezing으로 포착 |
| IEM | ice_accretion_3hr | 3시간 착빙(inch) | 99.871 | Excluded | 결측 99.9% |
| IEM | ice_accretion_6hr | 6시간 착빙(inch) | 99.809 | Excluded | 결측 99.8% |
| IEM | snowdepth | 적설 깊이(inch) | 99.853 | Excluded | 결측 99.9%, 보고 관측소가 극히 적음 |
| IEM | alti | 고도계 설정치(inHg) | 0.121 | Performance Candidate | 고도계 기압: 3시간 기압 경향(alti_change_3h)으로 기상 변화 추세 |
| IEM | mslp | 해면기압(hPa) | 27.986 | Excluded | alti와 정보 중복, SPECI에 없어 결측 28% |
| IEM | metar | 원문 METAR | 0 | Evaluation Only | 원문 보관(재파싱·검증용), 입력 아님 |

## 2. 학습 피처 목록 (`feature_lists.json`)

| list | n | 설명 |
|---|---|---|
| base | 21 | 항공편 일정 + 월·계절 + 출발 공항 핵심 기상 |
| base_plus_candidates | 70 | base + 혼잡도 proxy(공개된 BTS 기준)·작년 대비 추세, 과거 실적(M−14~M−3), 출발 공항 기상 후보·3시간 창 |
| no_weather | 32 | base_plus_candidates에서 출발 공항 기상 피처 전부 제외 (일주일 전 예측용) |
| with_dest_weather | 94 | base_plus_candidates + 도착 공항 기상(dest_*) |
| experimental_tail | 99 | with_dest_weather + tail_*_hist 5개 (참조일 기준 기체 일정) |
| alt_congestion_candidates | 4 | base_plus_candidates 의 혼잡도 proxy 대신 넣어 비교할 대안 4개 |

| feature | lists | categorical | missing_% train | definition |
|---|---|---|---|---|
| Reporting_Airline | base,base_plus_candidates,no_weather,with_dest_weather,experimental_tail | Y | 0 | 항공사 운항 특성(범주형 입력). 운항 전 확정 정보 |
| Origin | base,base_plus_candidates,no_weather,with_dest_weather,experimental_tail | Y | 0 | 출발 공항: 기상 관측소 매칭·시간대 결정·범주형 입력 |
| Dest | base,base_plus_candidates,no_weather,with_dest_weather,experimental_tail | Y | 0 | 도착 공항: 범주형 입력, 노선(route) 생성, 도착지 시간대 검증 |
| dep_hour_local | base,base_plus_candidates,no_weather,with_dest_weather,experimental_tail |  | 0 | 예정 출발 현지 시(0~23). CRSDepTime 2400은 0시로 처리 |
| day_of_week | base,base_plus_candidates,no_weather,with_dest_weather,experimental_tail |  | 0 | 요일(1=월~7=일) |
| is_weekend | base,base_plus_candidates,no_weather,with_dest_weather,experimental_tail |  | 0 | 토·일 여부 |
| month | base,base_plus_candidates,no_weather,with_dest_weather,experimental_tail |  | 0 | 월 |
| season | base,base_plus_candidates,no_weather,with_dest_weather,experimental_tail |  | 0 | 계절(0=겨울 … 3=가을) |
| Distance | base,base_plus_candidates,no_weather,with_dest_weather,experimental_tail |  | 0 | 노선 거리(사전 확정). 장·단거리 운항 특성 |
| tmpf | base,base_plus_candidates,with_dest_weather,experimental_tail |  | 0.04 | 기온: 결빙·제빙 조건 판단의 핵심 |
| sknt | base,base_plus_candidates,with_dest_weather,experimental_tail |  | 0.15 | 풍속: 강풍 운영 제한 |
| wind_max_kt | base,base_plus_candidates,with_dest_weather,experimental_tail |  | 0.149 | max(풍속, 돌풍) |
| vsby | base,base_plus_candidates,with_dest_weather,experimental_tail |  | 0.046 | 시정: 저시정 시 이착륙 간격 증가. 비행 기상 등급 계산 |
| ceiling_ft | base,base_plus_candidates,with_dest_weather,experimental_tail |  | 0.04 | 실링 = BKN/OVC/VV 중 최저 운저(ft). 운량은 관측됐으나 실링 층이 없으면 30000(실링 없음) |
| flight_category | base,base_plus_candidates,with_dest_weather,experimental_tail |  | 0.065 | FAA 비행 기상 등급 0=VFR,1=MVFR,2=IFR,3=LIFR (실링·시정 중 나쁜 쪽) |
| p01i | base,base_plus_candidates,with_dest_weather,experimental_tail |  | 0.034 | 1시간 강수량 |
| wx_thunder | base,base_plus_candidates,with_dest_weather,experimental_tail |  | 0 | 뇌우(TS, VCTS 포함) |
| wx_snow | base,base_plus_candidates,with_dest_weather,experimental_tail |  | 0 | 눈(SN/SG, BLSN 포함) |
| wx_rain | base,base_plus_candidates,with_dest_weather,experimental_tail |  | 0 | 비·이슬비(RA/DZ) |
| wx_freezing | base,base_plus_candidates,with_dest_weather,experimental_tail |  | 0 | 어는 비·이슬비(FZRA/FZDZ), 얼음싸라기(PL), 빙정(IC) |
| wx_fog | base,base_plus_candidates,with_dest_weather,experimental_tail |  | 0 | 안개(FG, FZFG/BCFG/MIFG 포함) |
| route | base_plus_candidates,no_weather,with_dest_weather,experimental_tail | Y | 0 | Origin_Dest 노선 코드(범주형) |
| arr_hour_local | base_plus_candidates,no_weather,with_dest_weather,experimental_tail |  | 0 | 예정 도착 현지 시 |
| CRSElapsedTime | base_plus_candidates,no_weather,with_dest_weather,experimental_tail |  | 0 | 예정 소요시간(사전 공개). Distance와 상관이 높지만 스케줄 여유(buffer)를 반영 |
| overnight_arrival | base_plus_candidates,no_weather,with_dest_weather,experimental_tail |  | 0 | 출발 시각+예정 소요시간이 자정을 넘는지 (레드아이 편) |
| day_of_month | base_plus_candidates,no_weather,with_dest_weather,experimental_tail |  | 0 | 일 |
| days_to_federal_holiday | base_plus_candidates,no_weather,with_dest_weather,experimental_tail |  | 0 | 가장 가까운 미국 연방 공휴일까지 일수(최대 30) |
| origin_sched_deps_hour_proxy_recent | base_plus_candidates,no_weather,with_dest_weather,experimental_tail |  | 0 |  |
| origin_sched_deps_day_proxy_recent | base_plus_candidates,no_weather,with_dest_weather,experimental_tail |  | 0 |  |
| dest_sched_arrs_hour_proxy_recent | base_plus_candidates,no_weather,with_dest_weather,experimental_tail |  | 0 | 도착 공항 ·  |
| carrier_sched_deps_origin_day_proxy_recent | base_plus_candidates,no_weather,with_dest_weather,experimental_tail |  | 0 |  |
| origin_yoy_ratio | base_plus_candidates,no_weather,with_dest_weather,experimental_tail |  | 4.224 | 출발 공항의 (M-3월 편수) / (1년 전 M-3월 편수). 작년 대비 운항량 추세 |
| hist_carrier_delay_rate | base_plus_candidates,no_weather,with_dest_weather,experimental_tail |  | 0 | 운항월 M 기준 M-14~M-3월(12개월, BTS 공개 지연 반영) BTS 전체 실적의 그룹별 15분 지연률·결항률, 베이지안 평활(m=50) |
| hist_carrier_cancel_rate | base_plus_candidates,no_weather,with_dest_weather,experimental_tail |  | 0 | 운항월 M 기준 M-14~M-3월(12개월, BTS 공개 지연 반영) BTS 전체 실적의 그룹별 15분 지연률·결항률, 베이지안 평활(m=50) |
| hist_origin_delay_rate | base_plus_candidates,no_weather,with_dest_weather,experimental_tail |  | 0.003 | 운항월 M 기준 M-14~M-3월(12개월, BTS 공개 지연 반영) BTS 전체 실적의 그룹별 15분 지연률·결항률, 베이지안 평활(m=50) |
| hist_origin_cancel_rate | base_plus_candidates,no_weather,with_dest_weather,experimental_tail |  | 0.003 | 운항월 M 기준 M-14~M-3월(12개월, BTS 공개 지연 반영) BTS 전체 실적의 그룹별 15분 지연률·결항률, 베이지안 평활(m=50) |
| hist_dest_delay_rate | base_plus_candidates,no_weather,with_dest_weather,experimental_tail |  | 0.003 | 운항월 M 기준 M-14~M-3월(12개월, BTS 공개 지연 반영) BTS 전체 실적의 그룹별 15분 지연률·결항률, 베이지안 평활(m=50) |
| hist_dest_cancel_rate | base_plus_candidates,no_weather,with_dest_weather,experimental_tail |  | 0.003 | 운항월 M 기준 M-14~M-3월(12개월, BTS 공개 지연 반영) BTS 전체 실적의 그룹별 15분 지연률·결항률, 베이지안 평활(m=50) |
| hist_route_delay_rate | base_plus_candidates,no_weather,with_dest_weather,experimental_tail |  | 0.244 | 운항월 M 기준 M-14~M-3월(12개월, BTS 공개 지연 반영) BTS 전체 실적의 그룹별 15분 지연률·결항률, 베이지안 평활(m=50) |
| hist_route_cancel_rate | base_plus_candidates,no_weather,with_dest_weather,experimental_tail |  | 0.244 | 운항월 M 기준 M-14~M-3월(12개월, BTS 공개 지연 반영) BTS 전체 실적의 그룹별 15분 지연률·결항률, 베이지안 평활(m=50) |
| hist_origin_hour_delay_rate | base_plus_candidates,no_weather,with_dest_weather,experimental_tail |  | 0.185 | 운항월 M 기준 M-14~M-3월(12개월, BTS 공개 지연 반영) BTS 전체 실적의 그룹별 15분 지연률·결항률, 베이지안 평활(m=50) |
| hist_origin_hour_cancel_rate | base_plus_candidates,no_weather,with_dest_weather,experimental_tail |  | 0.185 | 운항월 M 기준 M-14~M-3월(12개월, BTS 공개 지연 반영) BTS 전체 실적의 그룹별 15분 지연률·결항률, 베이지안 평활(m=50) |
| hist_carrier_origin_delay_rate | base_plus_candidates,no_weather,with_dest_weather,experimental_tail |  | 0.139 | 운항월 M 기준 M-14~M-3월(12개월, BTS 공개 지연 반영) BTS 전체 실적의 그룹별 15분 지연률·결항률, 베이지안 평활(m=50) |
| hist_carrier_origin_cancel_rate | base_plus_candidates,no_weather,with_dest_weather,experimental_tail |  | 0.139 | 운항월 M 기준 M-14~M-3월(12개월, BTS 공개 지연 반영) BTS 전체 실적의 그룹별 15분 지연률·결항률, 베이지안 평활(m=50) |
| dwpf | base_plus_candidates,with_dest_weather,experimental_tail |  | 0.069 | 이슬점: temp_dew_spread(안개 가능성) 계산에 사용 |
| relh | base_plus_candidates,with_dest_weather,experimental_tail |  | 0.051 | 상대습도: tmpf·dwpf에서 계산 가능하나 비선형 변환값으로 후보 |
| temp_dew_spread_f | base_plus_candidates,with_dest_weather,experimental_tail |  | 0.075 | 기온-이슬점(°F). 작을수록 안개·저운 가능성 |
| wind_dir_sin | base_plus_candidates,with_dest_weather,experimental_tail |  | 4.243 | 풍향 주기 변환(무풍은 0,0; VRB는 NaN) |
| wind_dir_cos | base_plus_candidates,with_dest_weather,experimental_tail |  | 4.243 | 풍향 주기 변환(무풍은 0,0; VRB는 NaN) |
| gust_reported | base_plus_candidates,with_dest_weather,experimental_tail |  | 0.15 | 돌풍 보고 여부(관측 있음 & gust 값 존재) |
| strong_wind | base_plus_candidates,with_dest_weather,experimental_tail |  | 0.15 | 풍속≥20kt 또는 돌풍≥25kt |
| low_ceiling | base_plus_candidates,with_dest_weather,experimental_tail |  | 0.04 | 실링<1000ft (IFR 이하) |
| low_visibility | base_plus_candidates,with_dest_weather,experimental_tail |  | 0.046 | 시정<3mi |
| wx_hail | base_plus_candidates,with_dest_weather,experimental_tail |  | 0 | 우박(GR/GS) |
| wx_mist_haze | base_plus_candidates,with_dest_weather,experimental_tail |  | 0 | 박무·연무·연기(BR/HZ/FU) |
| wx_heavy | base_plus_candidates,with_dest_weather,experimental_tail |  | 0 | 강한 강도(+) 현상 |
| precip_any | base_plus_candidates,with_dest_weather,experimental_tail |  | 0 | 강수 현상 아무거나(UP 포함) |
| icing_risk | base_plus_candidates,with_dest_weather,experimental_tail |  | 0.04 | 어는 강수 또는 기온≤32°F에서 강수·안개 (제빙 필요 가능 조건) |
| weather_severity | base_plus_candidates,with_dest_weather,experimental_tail |  | 0 | 뇌우+어는강수+눈+시정<1mi+실링<500ft+강풍 개수(0~6) |
| alti | base_plus_candidates,with_dest_weather,experimental_tail |  | 0.015 | 고도계 기압: 3시간 기압 경향(alti_change_3h)으로 기상 변화 추세 |
| thunder_3h | base_plus_candidates,with_dest_weather,experimental_tail |  | 0 | 직전 3시간 관측 중 해당 현상 존재 여부 |
| snow_3h | base_plus_candidates,with_dest_weather,experimental_tail |  | 0 | 직전 3시간 관측 중 해당 현상 존재 여부 |
| freezing_3h | base_plus_candidates,with_dest_weather,experimental_tail |  | 0 | 직전 3시간 관측 중 해당 현상 존재 여부 |
| max_wind_3h_kt | base_plus_candidates,with_dest_weather,experimental_tail |  | 0.018 | 직전 3시간 최대 wind_max_kt |
| min_vsby_3h | base_plus_candidates,with_dest_weather,experimental_tail |  | 0.024 | 직전 3시간 최저 시정 |
| min_ceiling_3h_ft | base_plus_candidates,with_dest_weather,experimental_tail |  | 0.015 | 직전 3시간 최저 실링 |
| precip_3h_in | base_plus_candidates,with_dest_weather,experimental_tail |  | 0.179 | 직전 3시간 정시 관측 1시간 강수 합 |
| n_special_obs_3h | base_plus_candidates,with_dest_weather,experimental_tail |  | 0 | 직전 3시간 특별 관측(SPECI 등) 수 = 기상 급변 지표 |
| alti_change_3h | base_plus_candidates,with_dest_weather,experimental_tail |  | 1.88 | 고도계 기압 3시간 변화(inHg) |
| obs_age_min | base_plus_candidates,with_dest_weather,experimental_tail |  | 0 | 예측 시점 - 매칭된 관측 시각(분) |
| dest_ceiling_ft | with_dest_weather,experimental_tail |  | 0.097 | 도착 공항 · 실링 = BKN/OVC/VV 중 최저 운저(ft). 운량은 관측됐으나 실링 층이 없으면 30000(실링 없음) |
| dest_vsby | with_dest_weather,experimental_tail |  | 0.101 | 도착 공항 · 시정: 저시정 시 이착륙 간격 증가. 비행 기상 등급 계산 |
| dest_flight_category | with_dest_weather,experimental_tail |  | 0.121 | 도착 공항 · FAA 비행 기상 등급 0=VFR,1=MVFR,2=IFR,3=LIFR (실링·시정 중 나쁜 쪽) |
| dest_wind_max_kt | with_dest_weather,experimental_tail |  | 0.212 | 도착 공항 · max(풍속, 돌풍) |
| dest_wx_thunder | with_dest_weather,experimental_tail |  | 0.059 | 도착 공항 · 뇌우(TS, VCTS 포함) |
| dest_wx_snow | with_dest_weather,experimental_tail |  | 0.059 | 도착 공항 · 눈(SN/SG, BLSN 포함) |
| dest_wx_rain | with_dest_weather,experimental_tail |  | 0.059 | 도착 공항 · 비·이슬비(RA/DZ) |
| dest_wx_freezing | with_dest_weather,experimental_tail |  | 0.059 | 도착 공항 · 어는 비·이슬비(FZRA/FZDZ), 얼음싸라기(PL), 빙정(IC) |
| dest_wx_hail | with_dest_weather,experimental_tail |  | 0.059 | 도착 공항 · 우박(GR/GS) |
| dest_wx_fog | with_dest_weather,experimental_tail |  | 0.059 | 도착 공항 · 안개(FG, FZFG/BCFG/MIFG 포함) |
| dest_wx_mist_haze | with_dest_weather,experimental_tail |  | 0.059 | 도착 공항 · 박무·연무·연기(BR/HZ/FU) |
| dest_wx_heavy | with_dest_weather,experimental_tail |  | 0.059 | 도착 공항 · 강한 강도(+) 현상 |
| dest_precip_any | with_dest_weather,experimental_tail |  | 0.059 | 도착 공항 · 강수 현상 아무거나(UP 포함) |
| dest_weather_severity | with_dest_weather,experimental_tail |  | 0.059 | 도착 공항 · 뇌우+어는강수+눈+시정<1mi+실링<500ft+강풍 개수(0~6) |
| dest_thunder_3h | with_dest_weather,experimental_tail |  | 0.059 | 도착 공항 · 직전 3시간 관측 중 해당 현상 존재 여부 |
| dest_snow_3h | with_dest_weather,experimental_tail |  | 0.059 | 도착 공항 · 직전 3시간 관측 중 해당 현상 존재 여부 |
| dest_freezing_3h | with_dest_weather,experimental_tail |  | 0.059 | 도착 공항 · 직전 3시간 관측 중 해당 현상 존재 여부 |
| dest_max_wind_3h_kt | with_dest_weather,experimental_tail |  | 0.075 | 도착 공항 · 직전 3시간 최대 wind_max_kt |
| dest_min_vsby_3h | with_dest_weather,experimental_tail |  | 0.079 | 도착 공항 · 직전 3시간 최저 시정 |
| dest_min_ceiling_3h_ft | with_dest_weather,experimental_tail |  | 0.07 | 도착 공항 · 직전 3시간 최저 실링 |
| dest_precip_3h_in | with_dest_weather,experimental_tail |  | 0.214 | 도착 공항 · 직전 3시간 정시 관측 1시간 강수 합 |
| dest_n_special_obs_3h | with_dest_weather,experimental_tail |  | 0.059 | 도착 공항 · 직전 3시간 특별 관측(SPECI 등) 수 = 기상 급변 지표 |
| dest_alti_change_3h | with_dest_weather,experimental_tail |  | 1.863 | 도착 공항 · 고도계 기압 3시간 변화(inHg) |
| dest_obs_age_min | with_dest_weather,experimental_tail |  | 0.059 | 예측 시점 − 매칭된 도착 공항 관측 시각(분) |
| tail_leg_index_hist | experimental_tail |  | 52.943 | 참조일(작년 같은 요일 ±1주 + M-3월 같은 요일)에 같은 항공사·편명·출발 공항 편에 배정된 기체의 하루 중 순서, 참조일 중앙값 |
| tail_sched_turn_min_hist | experimental_tail |  | 60.526 | 참조일 같은 편 기체의 앞 편 예정 도착~이 편 예정 출발 간격(분) 중앙값 |
| tail_prev_origin_hist | experimental_tail | Y | 60.526 | 참조일 같은 편 기체의 앞 편 출발 공항 최빈값(범주형) |
| tail_prev_sched_elapsed_hist | experimental_tail |  | 60.526 | 참조일 같은 편 기체의 앞 편 예정 소요시간 중앙값 |
| tail_ref_days_found | experimental_tail |  | 0 | 참조일 중 같은 편(기체 배정 있음)을 찾은 날 수 |

## 3. v2·v3에서 바뀐 판단

- **BTS 공개 지연(v3):** 서비스는 BTS 공개 데이터와 기상만 씁니다. BTS는 약 2개월 늦게 공개되므로, 운항월 M의 BTS 기반 피처는 M−3월 말까지만 씁니다.
- **혼잡도(v3):** 당일 편수(`*_oracle`)와 직전 4주(`*_proxy_v2_unavailable`)는 서비스 시점에 알 수 없어 학습에서 뺐습니다. 작년 같은 요일(`*_proxy_ly`)과 공개된 최신 달 같은 요일(`*_proxy_recent`) 중 oracle과 상관이 높은 쪽을 씁니다.
- **과거 실적(v3):** 운항월마다 M−14~M−3 창을 BTS 전체로 다시 계산합니다.
- **month, season:** split마다 12개월이 모두 들어가므로 다시 base에 넣었습니다.
- **도착 공항 기상:** 같은 예측 시점의 도착 공항 최신 관측입니다. 도착지 악천후로 인한 지상 대기(GDP) 같은 영향을 반영하기 위한 후보입니다.
- **기체 일정(v3):** 당일 Tail_Number 기준 `tail_*`는 결측이 곧 결항이라(v2 누수) `*_sameday`로 병합본에만 남겼습니다. 대신 과거 참조일의 같은 편 기체 일정으로 만든 `tail_*_hist`를 `experimental_tail`에 넣었습니다.
- 위 기대 효과는 가설이며, 목록별 비교 실험으로 확인해야 합니다.
