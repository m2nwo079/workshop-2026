"""
00_ingest.py — Empire Flippers 공식 공개 API 수집 스크립트

목적:
    EF 공개 매물 데이터를 공식 API로 받아, 수집일자를 붙여 원본 스냅샷으로 저장한다.
    (스크레이핑 아님. 인증 불필요. 초당 1요청 준수.)

수집 대상:
    listing_status = "For Sale"  (현재 판매중)
    listing_status = "Sold"      (거래 완료 — 호가 배수 타깃이므로 학습에 활용 가능)

저장 위치:
    data/raw/listings_forsale_YYYY-MM-DD.json
    data/raw/listings_sold_YYYY-MM-DD.json

실행:
    python 00_ingest.py

주의:
    - 이 단계는 "받아서 원본 그대로 저장"만 한다. 필드 분석/정제는 이후 단계에서.
    - data/raw 는 .gitignore 로 깃 제외되어 있어야 한다 (원문/용량/저작권).
"""

import json
import time
import datetime
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
BASE_URL = "https://api.empireflippers.com/api/v1/listings/list"

# 문서 스펙: 초당 1요청을 넘기지 말 것.
# 안전 마진을 두어 요청 사이에 1.1초 대기한다.
REQUEST_INTERVAL_SEC = 1.1

# 한 페이지당 최대 개수 (문서 스펙상 최대 100)
PAGE_LIMIT = 100

# 네트워크 오류 시 재시도 횟수
MAX_RETRIES = 3

# 저장 폴더. 이 스크립트가 프로젝트 루트에서 실행된다고 가정.
# (예: workshop-2026/ 안에서 python src/00_ingest.py 또는 python 00_ingest.py)
RAW_DIR = Path("data/raw")

# 오늘 날짜 (UTC 기준 — 문서: 모든 시간은 UTC)
TODAY = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# 한 페이지 요청
# ---------------------------------------------------------------------------
def fetch_page(listing_status: str, page: int) -> dict:
    """지정한 status/page 한 페이지를 받아 JSON(dict)으로 반환."""
    params = {
        "page": page,
        "limit": PAGE_LIMIT,
        "listing_status": listing_status,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            print(f"    [경고] 요청 실패 (시도 {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                # 재시도 전에는 조금 더 길게 대기
                time.sleep(REQUEST_INTERVAL_SEC * 2)
            else:
                raise  # 마지막 시도도 실패하면 예외를 올림

    return {}  # 도달하지 않지만 형식상


# ---------------------------------------------------------------------------
# 한 status 전체 수집 (페이지네이션)
# ---------------------------------------------------------------------------
def fetch_all(listing_status: str) -> list:
    """
    지정한 listing_status 의 모든 매물을 페이지를 넘겨가며 수집.
    각 매물(dict)의 리스트를 반환.
    """
    print(f"\n=== '{listing_status}' 수집 시작 ===")
    all_listings = []
    page = 1

    while True:
        print(f"  page {page} 요청 중...")
        data = fetch_page(listing_status, page)

        # 응답 구조: {"data": {"listings": [...]}}
        listings = data.get("data", {}).get("listings", [])

        if not listings:
            # 더 이상 매물이 없으면 종료
            print(f"  page {page}: 매물 없음 → 수집 종료")
            break

        all_listings.extend(listings)
        print(f"  page {page}: {len(listings)}개 수집 (누적 {len(all_listings)}개)")

        # 마지막 페이지 판단: 받은 개수가 limit 보다 작으면 끝
        if len(listings) < PAGE_LIMIT:
            print("  마지막 페이지 도달 → 수집 종료")
            break

        page += 1
        # 초당 1요청 준수: 다음 요청 전 대기
        time.sleep(REQUEST_INTERVAL_SEC)

    print(f"=== '{listing_status}' 총 {len(all_listings)}개 수집 완료 ===")
    return all_listings


# ---------------------------------------------------------------------------
# 저장
# ---------------------------------------------------------------------------
def save_snapshot(listings: list, label: str) -> Path:
    """수집한 매물 리스트를 수집일자를 붙여 JSON으로 저장."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"listings_{label}_{TODAY}.json"

    snapshot = {
        "collected_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "listing_status": label,
        "count": len(listings),
        "listings": listings,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    print(f"  저장 완료: {out_path}  ({len(listings)}개)")
    return out_path


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main():
    print(f"수집일자(UTC): {TODAY}")
    print(f"저장 경로: {RAW_DIR.resolve()}")

    # status 이름과 저장 파일 라벨을 함께 정의
    targets = [
        ("For Sale", "forsale"),
        ("Sold", "sold"),
    ]

    for status, label in targets:
        listings = fetch_all(status)
        save_snapshot(listings, label)
        # 두 status 사이에도 대기 (예의상 + 레이트리밋 여유)
        time.sleep(REQUEST_INTERVAL_SEC)

    print("\n전체 수집이 끝났습니다.")


if __name__ == "__main__":
    main()
