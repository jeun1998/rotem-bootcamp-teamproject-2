"""BTS IATA 공항 코드 -> IEM ASOS 관측소 매핑 (weather/station_map.csv 생성).

근거 자료
- IEM 네트워크별 관측소 메타데이터: https://mesonet.agron.iastate.edu/geojson/network/{NET}.geojson
  (meta/iem_networks/ 에 캐시)
- 공항 좌표·FAA/ICAO 코드 교차확인: OurAirports airports.csv (meta/ourairports_airports.csv)

매칭 규칙
1. IEM 관측소 ID == BTS IATA 이고 공항 좌표와 MAX_ONSITE_KM 이내   -> exact
2. IATA와 다른 FAA LID 또는 ICAO 코드가 IEM ID와 일치하고 MAX_ONSITE_KM 이내 -> faa_code
3. 위가 없거나 관측 기간이 대상 기간을 못 덮으면 공항 좌표에서 가장 가까운 관측소 -> nearest
4. MAX_NEAREST_KM 안에 관측소가 없거나 공항 좌표를 못 찾으면 -> unmatched
"""
import csv
import json
import math
import time
import urllib.request
import os
from pathlib import Path

def _weather_root():
    """날씨 데이터 폴더(station_map.csv, asos_raw/ 가 있는 곳, 저장소에는 없음).

    1) 환경변수 AIRLINE_DATA_ROOT/weather  2) 이 파일의 상위 폴더 중 weather/station_map.csv 가 있는 곳
    3) 없으면 현재 폴더/weather (처음 받을 때)
    """
    if os.environ.get("AIRLINE_DATA_ROOT"):
        return Path(os.environ["AIRLINE_DATA_ROOT"]).resolve() / "weather"
    for d in Path(__file__).resolve().parents:
        if (d / "weather" / "station_map.csv").is_file():
            return d / "weather"
    return Path.cwd() / "weather"


ROOT = _weather_root()
META = ROOT / "meta"
NET_DIR = META / "iem_networks"
UA = "bts-weather-research/1.0 (python-urllib)"

AIRPORTS = """ATL DFW DEN ORD CLT LAX PHX LAS SEA MCO LGA DTW SFO BOS DCA MSP IAH EWR SLC JFK MIA PHL BWI BNA FLL SAN AUS MDW TPA DAL STL PDX HNL HOU IAD RDU SMF SJC MSY MCI IND SNA OAK PIT CMH CLE CVG RSW SAT SJU BUR JAX OGG PBI MKE CHS ONT BOI OMA BDL SDF MEM ABQ OKC ORF RNO BUF ANC GEG SAV RIC TUS GRR ELP TUL BHM SRQ KOA LGB MYR DSM LIH PVD PSP TYS GSP PNS COS LIT FAT SYR XNA ALB MSN ROC PWM HPN BZN GSO VPS AVL SFB ICT MAF PIE ECP EYW SGF HSV LEX CID EUG DAY SBA FSD ILM JAN BTV ASE RDM PGD FAR ITO SBN AZA MHT LBB CAE MFR CHA JAC PSC ATW FWA STT TLH MDT RAP AMA SBP SHV ISP BTR MSO JNU BIL TVC FCA ABE GRB MFE MRY BGR GJT BIS MLI CRP FAI HRL PIA IDA STS DRO AGS SGU GPT GNV EGE LFT CAK CRW DAB ACY MOB MTJ CHO FNT MLB BFL MGM GTF BLI MOT TRI KTN HHH HDN SAF EVV ROA BMI GRK BQN DLH AEX SUN MLU FAY FLG ABI AVP LAN TTN OAJ RST CPR ACV PVU MBS YUM COU PAE BRO LRD XWA CMI HLN AZO ACK RDD SJT LNK CWA ACT CLL MHK FSM SIT GFK STX BLV TYR LCH LSE ELM ORH MQT LAW LCK CSG ROW MVY GRI TXK USA LWS SPS ABY PLN GUC ADQ RHI RFD GTR JMS VLD BPT SCE GGG SLN DEC PRC SUX CIU EWN DHN ABR TWF CMX PSG WRG YAK CDV BJI IMT JLN GCK LBE BQK GUM HYS BET GCC SPI PSE SWO BTM PBG MEI ESC LBF DVL SHR PIH BFF BRD HIB PIB INL LAR LBL TOL JST DDC SCK SWF APN CDC COD RKS CKB RIW VCT ITH SCC EAR OME OTZ MCW FOD EKO HTS CYS CNY EAU IAG DIK BGM VEL PHF PUW ALW HOB BRW PSM ALO OTH PAH SPN SHD DRT WYS PUB MKG LWB CGI AKN ERI OGS ATY DBQ PIR DLG EAT HGR YKM BIH HYA SMX LYH ALS STC MGW CLD TBN ADK HVN GST PPG PQI OWB GUF MMH CDB FMN LAF OGD PGV BKG ILG FLO IPT BFM VRB UIN IFP""".split()

US_STATES = """AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY""".split()
# 미국령: 푸에르토리코, 버진아일랜드, 괌, 아메리칸사모아(PPG), 북마리아나(SPN; IEM은 국가 네트워크 MP__ASOS)
NETWORKS = [f"{s}_ASOS" for s in US_STATES] + ["PR_ASOS", "VI_ASOS", "GU_ASOS", "AS_ASOS", "MP__ASOS"]
US_COUNTRIES = {"US", "PR", "VI", "GU", "AS", "MP"}

PERIOD_BEGIN, PERIOD_END = "2019-12-31", "2026-01-01"
MAX_ONSITE_KM = 8.0
MAX_NEAREST_KM = 75.0


def fetch(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return dest.read_bytes()
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read()
            dest.write_bytes(body)
            time.sleep(1.5)
            return body
        except Exception as e:  # noqa: BLE001
            print(f"  retry {attempt + 1} {url}: {e}")
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"failed: {url}")


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def load_iem():
    NET_DIR.mkdir(parents=True, exist_ok=True)
    stations = {}
    for net in NETWORKS:
        body = fetch(f"https://mesonet.agron.iastate.edu/geojson/network/{net}.geojson", NET_DIR / f"{net}.geojson")
        for f in json.loads(body)["features"]:
            p = f["properties"]
            lon, lat = f["geometry"]["coordinates"]
            stations[p["sid"]] = {
                "sid": p["sid"], "network": p["network"], "name": p.get("sname", ""),
                "lat": lat, "lon": lon, "elevation": p.get("elevation"),
                "tzname": p.get("tzname") or "",
                "begin": p.get("archive_begin") or "", "end": p.get("archive_end") or "",
            }
    return stations


def covers_period(st):
    """archive 기간이 대상 기간을 (대부분) 덮는지."""
    if not st["begin"]:
        return False
    return st["begin"] <= PERIOD_BEGIN and (not st["end"] or st["end"] >= PERIOD_END)


def load_ourairports():
    """IATA -> OurAirports 행. 코드가 바뀐 공항(예: PBI->DJT)은 keywords의 옛 코드로도 찾는다."""
    rank = {"large_airport": 0, "medium_airport": 1, "small_airport": 2}
    by_iata, by_old = {}, {}
    with open(META / "ourairports_airports.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["iso_country"] not in US_COUNTRIES:
                continue
            key = (r["type"] == "closed", rank.get(r["type"], 3))
            code = r["iata_code"].strip()
            if code and (code not in by_iata or key < by_iata[code][0]):
                by_iata[code] = (key, r)
            if r["scheduled_service"] == "yes":
                for kw in r["keywords"].replace(",", " ").split():
                    if len(kw) == 3 and kw.isupper() and kw != code:
                        by_old.setdefault(kw, (key, r))
    out = {k: v[1] for k, v in by_iata.items()}
    renamed = {k: v[1] for k, v in by_old.items() if k not in out}
    return out, renamed


def main():
    iem = load_iem()
    oa, oa_renamed = load_ourairports()
    print(f"IEM stations: {len(iem)}  OurAirports US IATA: {len(oa)}")
    candidates_nearest = [s for s in iem.values() if covers_period(s)]

    out = []
    for iata in AIRPORTS:
        ap = oa.get(iata)
        row = dict(bts_iata=iata, iem_station="", iem_network="", station_name="", lat="", lon="",
                   elevation="", timezone="", match_type="unmatched", distance_km="", note="")
        notes = []
        if ap is None and iata in oa_renamed:
            ap = oa_renamed[iata]
            notes.append(f"IATA {iata}->{ap['iata_code']} 코드 변경 (OurAirports keywords)")
        if ap is None:
            notes.append("OurAirports에 IATA 없음")
            if iata in iem:  # 좌표 검증은 못 하지만 ID 일치
                st = iem[iata]
                row.update(match_type="exact")
                notes.append("좌표 교차검증 불가")
                _fill(row, st, None)
            row["note"] = "; ".join(notes)
            out.append(row)
            continue

        alat, alon = float(ap["latitude_deg"]), float(ap["longitude_deg"])
        ids = []
        for c in (iata, ap["local_code"], ap["icao_code"], ap["gps_code"], ap["ident"]):
            c = (c or "").strip()
            if c and c not in ids:
                ids.append(c)

        chosen = None
        for c in ids:
            st = iem.get(c)
            if st is None:
                continue
            d = haversine(alat, alon, st["lat"], st["lon"])
            if d > MAX_ONSITE_KM:
                notes.append(f"ID {c} 일치하나 {d:.1f}km 떨어져 제외")
                continue
            if not covers_period(st):
                notes.append(f"ID {c} 일치하나 archive {st['begin']}~{st['end'] or 'now'}로 기간 미충족")
                continue
            chosen = (st, d, "exact" if c == iata else "faa_code")
            if c != iata:
                src = "FAA LID" if c == ap["local_code"] else "ICAO"
                notes.append(f"{src} {c} ({ap['name']})")
            break

        if chosen is None and candidates_nearest:
            st, d = min(((s, haversine(alat, alon, s["lat"], s["lon"])) for s in candidates_nearest),
                        key=lambda t: t[1])
            if d <= MAX_NEAREST_KM:
                chosen = (st, d, "nearest")
                notes.append(f"공항 {ap['name']} 대신 최근접 관측소")
            else:
                notes.append(f"최근접 관측소 {st['sid']} {d:.0f}km > {MAX_NEAREST_KM:.0f}km")

        if chosen:
            st, d, mt = chosen
            row["match_type"] = mt
            _fill(row, st, d)
            if st["end"]:
                notes.append(f"archive_end {st['end']}")
        row["note"] = "; ".join(notes)
        out.append(row)

    cols = ["bts_iata", "iem_station", "iem_network", "station_name", "lat", "lon", "elevation",
            "timezone", "match_type", "distance_km", "note"]
    with open(ROOT / "station_map.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    from collections import Counter
    print(Counter(r["match_type"] for r in out))


def _fill(row, st, d):
    row.update(iem_station=st["sid"], iem_network=st["network"], station_name=st["name"],
               lat=st["lat"], lon=st["lon"], elevation=st["elevation"], timezone=st["tzname"],
               distance_km="" if d is None else round(d, 2))


if __name__ == "__main__":
    main()
