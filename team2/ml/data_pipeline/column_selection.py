"""원본 컬럼 분류와 학습 특징 목록 (feature_selection_report.md 와 feature_lists.json 의 단일 근거).

Category: Essential / Performance Candidate / Evaluation Only / Excluded
- Essential 원본 컬럼은 키·매칭·UTC 변환·Label 생성·기본 입력 특징에 필요한 것
- 학습 입력 특징은 BASE_FEATURES(기본 실험) 와 CANDIDATE_FEATURES(추가 실험)로 나눈다
"""

E, P, V, X = "Essential", "Performance Candidate", "Evaluation Only", "Excluded"

BTS_COLUMNS = {
    "Year": (X, "FlightDate와 중복(파생 관계 100% 일치). 시간순 분할에서 연도는 상수"),
    "Quarter": (X, "FlightDate에서 파생 가능, 1~3월 데이터셋에서는 상수"),
    "Month": (X, "FlightDate와 중복. 연 단위 분할(v2)에서는 파생값 month·season을 학습 특징으로 사용"),
    "DayofMonth": (P, "day_of_month 로 사용 (월초·월말 운항 패턴 후보). FlightDate와 중복이므로 원본 대신 파생값 사용"),
    "DayOfWeek": (E, "요일별 운항량·지연 패턴. day_of_week, is_weekend 생성"),
    "FlightDate": (E, "운항일(현지). 시간순 분할, 현지→UTC 변환, 공휴일 거리 계산의 기준"),
    "Reporting_Airline": (E, "항공사 운항 특성(범주형 입력). 운항 전 확정 정보"),
    "Flight_Number_Reporting_Airline": (E, "항공편 식별 키(입력 특징 아님). 고유값이 많고 의미가 항공사마다 달라 입력에서 제외"),
    "Tail_Number": (P, "원본은 입력 제외. 당일 값은 결항 편에서 결측이 많아 누수(v2) → v3는 과거 참조일(작년 같은 요일·M-3월)의 같은 편 기체 배정으로만 tail_*_hist를 만든다. 당일 기준 값은 *_sameday로 병합본에만 보존"),
    "OriginAirportID": (X, "Origin과 1:1 (공항 코드 변경 추적용). 기간 내 코드 변경이 없어 중복"),
    "Origin": (E, "출발 공항: 기상 관측소 매칭·시간대 결정·범주형 입력"),
    "OriginCityName": (X, "Origin과 중복"),
    "OriginState": (X, "Origin에 종속(공항→주). 중복 정보"),
    "DestAirportID": (X, "Dest와 1:1, 중복"),
    "Dest": (E, "도착 공항: 범주형 입력, 노선(route) 생성, 도착지 시간대 검증"),
    "DestCityName": (X, "Dest와 중복"),
    "DestState": (X, "Dest에 종속, 중복"),
    "CRSDepTime": (E, "예정 출발 현지 시각: UTC 변환, 예측 시점(3시간 전) 계산, 출발 시간대 특징"),
    "DepTimeBlk": (X, "CRSDepTime에서 결정되는 1시간 블록(00~05시는 '0001-0559' 하나로 묶임) → dep_hour_local이 더 세밀"),
    "CRSArrTime": (P, "예정 도착 현지 시각: arr_hour_local, 도착 공항 혼잡도, 시간대 교차검증"),
    "ArrTimeBlk": (X, "CRSArrTime 블록, 중복"),
    "CRSElapsedTime": (P, "예정 소요시간(사전 공개). Distance와 상관이 높지만 스케줄 여유(buffer)를 반영"),
    "Distance": (E, "노선 거리(사전 확정). 장·단거리 운항 특성"),
    "DistanceGroup": (X, "Distance의 250mile 구간화(파생 관계 일치), 중복"),
    "DepTime": (V, "실제 출발 시각 = 사후 정보. 누수"),
    "DepDelay": (V, "Label 생성용(출발 지연 분). 입력 사용 시 직접 누수"),
    "DepDelayMinutes": (V, "DepDelay의 음수 0 처리판, Label 검증용"),
    "DepDel15": (V, "Label(15분 지연) 교차검증용"),
    "DepartureDelayGroups": (V, "DepDelay 구간화, 사후 정보"),
    "TaxiOut": (V, "출발 후 확정되는 사후 정보"),
    "WheelsOff": (V, "사후 정보"),
    "WheelsOn": (V, "사후 정보"),
    "TaxiIn": (V, "사후 정보"),
    "ArrTime": (V, "사후 정보"),
    "ArrDelay": (V, "사후 정보 (도착 지연 분석용)"),
    "ArrDelayMinutes": (V, "사후 정보"),
    "ArrDel15": (V, "사후 정보. 도착 기준 레이블 label_arr(평가용) 생성에 사용"),
    "ArrivalDelayGroups": (V, "사후 정보"),
    "ActualElapsedTime": (V, "사후 정보"),
    "AirTime": (V, "사후 정보"),
    "Cancelled": (V, "Label 생성(Class 2) 전용. 입력 금지"),
    "CancellationCode": (V, "결항 사유(B=기상) 분석·검증용. 결항편에만 존재하는 사후 정보"),
    "Diverted": (V, "회항 여부(사후). 회항편은 출발했으므로 출발지연 기준으로 Label 부여, 분석 플래그로만 보존"),
    "CarrierDelay": (V, "지연 원인 분해(ArrDel15=1일 때만 존재). 사후 정보, 원인 분석용"),
    "WeatherDelay": (V, "사후 정보, 기상 지연 분석용"),
    "NASDelay": (V, "사후 정보"),
    "SecurityDelay": (V, "사후 정보"),
    "LateAircraftDelay": (V, "사후 정보 (선행편 지연). 예측 시점에 부분적으로만 알 수 있어 이번 단계 제외"),
}

IEM_COLUMNS = {
    "station": (E, "관측소 ID: 출발 공항 매칭 키"),
    "valid": (E, "관측 시각(UTC): 예측 시점 이전 최신 관측 매칭 키"),
    "lon": (X, "관측소 고정 좌표. 공항 내에서 변하지 않음(공항 코드와 중복)"),
    "lat": (X, "관측소 고정 좌표, 공항별 상수"),
    "elevation": (X, "관측소 고도, 공항별 상수(공항 범주형과 중복). 밀도고도 계산이 필요하면 재검토"),
    "tmpf": (E, "기온: 결빙·제빙 조건 판단의 핵심"),
    "dwpf": (P, "이슬점: temp_dew_spread(안개 가능성) 계산에 사용"),
    "relh": (P, "상대습도: tmpf·dwpf에서 계산 가능하나 비선형 변환값으로 후보"),
    "feel": (X, "체감온도: tmpf·sknt에서 결정되는 파생값, 운항과 직접 관련 낮음"),
    "drct": (P, "풍향: sin/cos 주기 변환. 활주로 방향 정보가 없어 측풍은 계산 불가"),
    "sknt": (E, "풍속: 강풍 운영 제한"),
    "gust": (E, "돌풍: 결측 86%이나 '돌풍 미보고'의 의미가 있음 → wind_max_kt, gust_reported로 사용"),
    "peak_wind_gust": (X, "결측 94%, 정보는 gust·max_wind_3h와 중복"),
    "vsby": (E, "시정: 저시정 시 이착륙 간격 증가. 비행 기상 등급 계산"),
    "skyc1": (E, "1층 운량: 실링(BKN/OVC/VV) 판단"),
    "skyc2": (P, "2층 운량: ceiling_ft 계산에만 사용(원본 입력 제외)"),
    "skyc3": (P, "3층 운량: ceiling_ft 계산에만 사용"),
    "skyc4": (X, "결측 99%. ceiling 계산에는 포함하되 원본 입력 제외"),
    "skyl1": (E, "1층 운저 고도: 실링 계산"),
    "skyl2": (P, "ceiling_ft 계산에만 사용"),
    "skyl3": (P, "ceiling_ft 계산에만 사용"),
    "skyl4": (X, "결측 99%, ceiling 계산에만 포함"),
    "p01i": (E, "1시간 강수량"),
    "wxcodes": (E, "현재 일기 코드: 뇌우·눈·어는비·안개 플래그 생성 (결항 핵심 신호)"),
    "ice_accretion_1hr": (X, "결측 99%(착빙 센서·사례 드묾). 어는 강수는 wx_freezing으로 포착"),
    "ice_accretion_3hr": (X, "결측 99.9%"),
    "ice_accretion_6hr": (X, "결측 99.8%"),
    "snowdepth": (X, "결측 99.9%, 보고 관측소가 극히 적음"),
    "alti": (P, "고도계 기압: 3시간 기압 경향(alti_change_3h)으로 기상 변화 추세"),
    "mslp": (X, "alti와 정보 중복, SPECI에 없어 결측 28%"),
    "metar": (V, "원문 보관(재파싱·검증용), 입력 아님"),
}

# 파생 특징 정의 (보고서용)
FEATURE_DESC = {
    "dep_hour_local": "예정 출발 현지 시(0~23). CRSDepTime 2400은 0시로 처리",
    "arr_hour_local": "예정 도착 현지 시",
    "dep_minute_of_day": "예정 출발 현지 분(0~1439)",
    "overnight_arrival": "출발 시각+예정 소요시간이 자정을 넘는지 (레드아이 편)",
    "month": "월", "season": "계절(0=겨울 … 3=가을)",
    "day_of_week": "요일(1=월~7=일)", "is_weekend": "토·일 여부", "day_of_month": "일",
    "days_to_federal_holiday": "가장 가까운 미국 연방 공휴일까지 일수(최대 30)",
    "route": "Origin_Dest 노선 코드(범주형)",
    "*_proxy_ly": "작년 같은 요일(D-364)과 앞뒤 1주(D-371, D-357) 같은 그룹의 예정 편수 평균 (편이 없던 날은 0). 항상 공개된 BTS",
    "*_proxy_recent": "공개된 최신 달(M-3) 중 운항일과 같은 요일인 날들의 같은 그룹 평균 예정 편수",
    "origin_yoy_ratio": "출발 공항의 (M-3월 편수) / (1년 전 M-3월 편수). 작년 대비 운항량 추세",
    "hist_*_delay_rate / hist_*_cancel_rate": "운항월 M 기준 M-14~M-3월(12개월, BTS 공개 지연 반영) BTS 전체 실적의 그룹별 15분 지연률·결항률, 베이지안 평활(m=50)",
    "tail_leg_index_hist": "참조일(작년 같은 요일 ±1주 + M-3월 같은 요일)에 같은 항공사·편명·출발 공항 편에 배정된 기체의 하루 중 순서, 참조일 중앙값",
    "tail_sched_turn_min_hist": "참조일 같은 편 기체의 앞 편 예정 도착~이 편 예정 출발 간격(분) 중앙값",
    "tail_prev_origin_hist": "참조일 같은 편 기체의 앞 편 출발 공항 최빈값(범주형)",
    "tail_prev_sched_elapsed_hist": "참조일 같은 편 기체의 앞 편 예정 소요시간 중앙값",
    "tail_ref_days_found": "참조일 중 같은 편(기체 배정 있음)을 찾은 날 수",
    "tail_leg_index_of_day": "같은 날 같은 기체의 몇 번째 예정 운항인지(예정 출발 UTC 순)",
    "tail_sched_turn_min": "앞 편 예정 도착(예정 출발+예정 소요) ~ 이 편 예정 출발 간격(분)",
    "tail_prev_origin": "같은 기체 앞 편의 출발 공항(범주형)",
    "tail_prev_sched_elapsed": "같은 기체 앞 편의 예정 소요시간(분)",
    "dest_*": "도착 공항 관측소의 같은 예측 시점 이전 최신 관측(최대 2시간)으로 만든 같은 정의의 기상 특징",
    "temp_dew_spread_f": "기온-이슬점(°F). 작을수록 안개·저운 가능성",
    "wind_dir_sin / wind_dir_cos": "풍향 주기 변환(무풍은 0,0; VRB는 NaN)",
    "wind_max_kt": "max(풍속, 돌풍)",
    "gust_reported": "돌풍 보고 여부(관측 있음 & gust 값 존재)",
    "strong_wind": "풍속≥20kt 또는 돌풍≥25kt",
    "ceiling_ft": "실링 = BKN/OVC/VV 중 최저 운저(ft). 운량은 관측됐으나 실링 층이 없으면 30000(실링 없음)",
    "low_ceiling": "실링<1000ft (IFR 이하)",
    "low_visibility": "시정<3mi",
    "flight_category": "FAA 비행 기상 등급 0=VFR,1=MVFR,2=IFR,3=LIFR (실링·시정 중 나쁜 쪽)",
    "wx_thunder": "뇌우(TS, VCTS 포함)", "wx_snow": "눈(SN/SG, BLSN 포함)", "wx_rain": "비·이슬비(RA/DZ)",
    "wx_freezing": "어는 비·이슬비(FZRA/FZDZ), 얼음싸라기(PL), 빙정(IC)", "wx_hail": "우박(GR/GS)",
    "wx_fog": "안개(FG, FZFG/BCFG/MIFG 포함)", "wx_mist_haze": "박무·연무·연기(BR/HZ/FU)",
    "wx_heavy": "강한 강도(+) 현상", "precip_any": "강수 현상 아무거나(UP 포함)",
    "icing_risk": "어는 강수 또는 기온≤32°F에서 강수·안개 (제빙 필요 가능 조건)",
    "weather_severity": "뇌우+어는강수+눈+시정<1mi+실링<500ft+강풍 개수(0~6)",
    "thunder_3h / snow_3h / freezing_3h": "직전 3시간 관측 중 해당 현상 존재 여부",
    "max_wind_3h_kt": "직전 3시간 최대 wind_max_kt", "min_vsby_3h": "직전 3시간 최저 시정",
    "min_ceiling_3h_ft": "직전 3시간 최저 실링", "precip_3h_in": "직전 3시간 정시 관측 1시간 강수 합",
    "n_special_obs_3h": "직전 3시간 특별 관측(SPECI 등) 수 = 기상 급변 지표",
    "alti_change_3h": "고도계 기압 3시간 변화(inHg)",
    "obs_age_min": "예측 시점 - 매칭된 관측 시각(분)",
}

CATEGORICAL = ["Reporting_Airline", "Origin", "Dest", "route", "tail_prev_origin_hist"]
CONGESTION_NAMES = ["origin_sched_deps_hour", "origin_sched_deps_day", "dest_sched_arrs_hour",
                    "carrier_sched_deps_origin_day"]

BASE_FEATURES = [
    # 항공편 (사전 확정 일정)
    "Reporting_Airline", "Origin", "Dest", "dep_hour_local", "day_of_week", "is_weekend", "month", "season",
    "Distance",
    # 출발 공항 기상 (예측 시점 이전 최신 관측)
    "tmpf", "sknt", "wind_max_kt", "vsby", "ceiling_ft", "flight_category", "p01i",
    "wx_thunder", "wx_snow", "wx_rain", "wx_freezing", "wx_fog",
]
CANDIDATE_FEATURES = [
    # 항공편
    "route", "arr_hour_local", "CRSElapsedTime", "overnight_arrival", "day_of_month", "days_to_federal_holiday",
    "{CONGESTION}", "origin_yoy_ratio",
    "hist_carrier_delay_rate", "hist_carrier_cancel_rate", "hist_origin_delay_rate", "hist_origin_cancel_rate",
    "hist_dest_delay_rate", "hist_dest_cancel_rate", "hist_route_delay_rate", "hist_route_cancel_rate",
    "hist_origin_hour_delay_rate", "hist_origin_hour_cancel_rate", "hist_carrier_origin_delay_rate",
    "hist_carrier_origin_cancel_rate",
    # 기상 (관측 1건)
    "dwpf", "relh", "temp_dew_spread_f", "wind_dir_sin", "wind_dir_cos", "gust_reported", "strong_wind",
    "low_ceiling", "low_visibility", "wx_hail", "wx_mist_haze", "wx_heavy", "precip_any", "icing_risk",
    "weather_severity", "alti",
    # 기상 (직전 3시간 창)
    "thunder_3h", "snow_3h", "freezing_3h", "max_wind_3h_kt", "min_vsby_3h", "min_ceiling_3h_ft",
    "precip_3h_in", "n_special_obs_3h", "alti_change_3h",
    # 매칭 품질
    "obs_age_min",
]
# 출발 공항 기상에서 온 특징 전부 (no_weather 실험에서 제외)
ORIGIN_WEATHER_FEATURES = [
    "tmpf", "sknt", "wind_max_kt", "vsby", "ceiling_ft", "flight_category", "p01i", "wx_thunder", "wx_snow",
    "wx_rain", "wx_freezing", "wx_fog", "dwpf", "relh", "temp_dew_spread_f", "wind_dir_sin", "wind_dir_cos",
    "gust_reported", "strong_wind", "low_ceiling", "low_visibility", "wx_hail", "wx_mist_haze", "wx_heavy",
    "precip_any", "icing_risk", "weather_severity", "alti", "thunder_3h", "snow_3h", "freezing_3h",
    "max_wind_3h_kt", "min_vsby_3h", "min_ceiling_3h_ft", "precip_3h_in", "n_special_obs_3h", "alti_change_3h",
    "obs_age_min",
]
# 도착 공항 기상 (같은 예측 시점 기준). 원래 이름 -> dest_ 접두어
DEST_SOURCE = ["ceiling_ft", "vsby", "flight_category", "wind_max_kt", "wx_thunder", "wx_snow", "wx_rain",
               "wx_freezing", "wx_hail", "wx_fog", "wx_mist_haze", "wx_heavy", "precip_any", "weather_severity",
               "thunder_3h", "snow_3h", "freezing_3h", "max_wind_3h_kt", "min_vsby_3h", "min_ceiling_3h_ft",
               "precip_3h_in", "n_special_obs_3h", "alti_change_3h"]
DEST_WEATHER_FEATURES = [f"dest_{c}" for c in DEST_SOURCE] + ["dest_obs_age_min"]
TAIL_FEATURES = ["tail_leg_index_hist", "tail_sched_turn_min_hist", "tail_prev_origin_hist",
                 "tail_prev_sched_elapsed_hist", "tail_ref_days_found"]
TAIL_SAMEDAY = ["tail_leg_index_of_day_sameday", "tail_sched_turn_min_sameday", "tail_prev_origin_sameday",
                "tail_prev_sched_elapsed_sameday"]

LABEL = "label"
SAMPLING_COLS = ["sample_rate", "sample_weight"]


def candidate_features(proxy="proxy_ly"):
    out = []
    for f in CANDIDATE_FEATURES:
        out += [f"{n}_{proxy}" for n in CONGESTION_NAMES] if f == "{CONGESTION}" else [f]
    return out


def feature_lists(proxy="proxy_ly"):
    """proxy: base_plus_candidates 에 넣을 혼잡도 proxy ('proxy_ly' 또는 'proxy_recent')."""
    base_plus = BASE_FEATURES + candidate_features(proxy)
    with_dest = base_plus + DEST_WEATHER_FEATURES
    lists = {
        "base": BASE_FEATURES,
        "base_plus_candidates": base_plus,
        "no_weather": [f for f in base_plus if f not in ORIGIN_WEATHER_FEATURES],
        "with_dest_weather": with_dest,
        "experimental_tail": with_dest + TAIL_FEATURES,
    }
    other = "proxy_recent" if proxy == "proxy_ly" else "proxy_ly"
    lists["alt_congestion_candidates"] = [f"{n}_{other}" for n in CONGESTION_NAMES]
    return lists


def forbidden_columns():
    """학습 특징에 들어가면 안 되는 컬럼: 사후·정답, 샘플링, 당일 혼잡도(oracle), 도착 레이블."""
    post = [c for c, (cat, _) in BTS_COLUMNS.items() if cat == V]
    oracle = [f"{n}_oracle" for n in CONGESTION_NAMES]
    unavailable = [f"{n}_proxy_v2_unavailable" for n in CONGESTION_NAMES]
    return (post + SAMPLING_COLS + oracle + unavailable + TAIL_SAMEDAY
            + ["label", "label_arr", "label_status", "Tail_Number", "metar", "wx_obs_metar", "tail_ref_max_date",
               "bts_source_max_date", "bts_source_limit_date"])
