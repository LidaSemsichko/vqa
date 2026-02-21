import argparse
import base64
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image
from tqdm import tqdm
from openai import OpenAI

DEFAULT_QUERIES = [

    # Nature / Landscapes
    "mountain landscape", "forest trail", "rainforest", "jungle",
    "beach sunset", "rocky beach", "river", "waterfall",
    "desert dunes", "snowy field", "lake reflection",
    "volcano", "canyon", "glacier", "island coast",
    "savanna", "autumn forest", "spring meadow",
    "storm clouds", "sunrise horizon", "foggy valley",

    # Weather
    "rainy street", "snowstorm", "lightning storm",
    "rainbow sky", "cloudy sky", "sunny day park",

    # Cities / Urban
    "city skyline", "night city", "street market",
    "downtown traffic", "busy intersection",
    "suburban house", "apartment building",
    "urban alley", "metro station",
    "train station platform", "airport terminal",
    "shopping mall interior", "sidewalk cafe",

    # Architecture
    "bridge", "castle", "museum interior",
    "church interior", "cathedral",
    "modern skyscraper", "wooden cabin",
    "lighthouse", "stadium", "library interior",
    "ancient ruins",

    # Animals
    "dog", "cat", "horse", "bird flying",
    "butterfly on flower", "fish underwater",
    "cow in field", "sheep flock",
    "elephant", "lion", "tiger",
    "giraffe", "zebra", "monkey",
    "panda", "penguin", "chicken",
    "duck on pond", "owl", "eagle",

    # Transport
    "car on road", "bus stop", "train",
    "airplane in sky", "bicycle rider",
    "motorcycle", "ship at sea",
    "helicopter", "tractor",
    "fire truck", "ambulance",

    # Indoor scenes
    "kitchen", "modern kitchen",
    "living room", "bedroom interior",
    "bathroom", "office desk",
    "conference room", "classroom",
    "laboratory", "hospital room",
    "restaurant interior", "coffee shop",

    # People / Activities
    "family dinner", "birthday party",
    "sports game", "football match",
    "basketball game", "concert stage",
    "wedding ceremony", "cooking in kitchen",
    "students studying", "person jogging",
    "hiking group", "person reading book",

    # Professions
    "doctor in hospital", "nurse",
    "teacher in classroom", "construction worker",
    "chef cooking", "police officer",
    "firefighter", "farmer in field",
    "mechanic workshop", "scientist laboratory",

    # Food
    "pizza", "salad bowl", "fruit basket",
    "bread loaf", "chocolate cake",
    "hamburger", "pasta dish",
    "sushi plate", "breakfast table",
    "grilled vegetables", "ice cream cone",

    # Everyday Objects
    "laptop", "phone", "tablet",
    "book on table", "wall clock",
    "chair", "wooden table",
    "lamp", "cup of coffee",
    "water bottle", "backpack",
    "television", "keyboard",
    "camera", "headphones",

    # Industry / Tech
    "factory interior", "warehouse shelves",
    "construction site", "solar panels",
    "wind turbines", "server room",
    "data center", "robot",
    "drone flying", "3D printer",

    # Art / Culture
    "sculpture", "painting gallery",
    "street art mural", "museum exhibition",
    "festival parade", "traditional dance",

    # Sports
    "tennis match", "swimming pool",
    "skiing", "cycling race",
    "boxing match", "yoga class",
    "surfing", "rock climbing",

    # Science
    "microscope", "chemical experiment",
    "astronomy telescope", "rocket launch",

    # Close-ups / macro
    "flower close up", "leaf texture",
    "raindrops on window", "insect macro",
    "coffee beans close up",

    # Abstract / textures
    "pattern texture", "colorful background",
    "geometric shapes", "wood texture",
    "metal surface", "brick wall texture"
]

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "VQACollector/1.0 (contact: you@example.com) PythonRequests",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
})

def get_with_retry(url: str, *, params=None, headers=None, stream=False, timeout=30,
                   retries: int = 6, base_sleep: float = 0.8) -> requests.Response:
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, headers=headers, stream=stream, timeout=timeout)
            if r.status_code in (403, 429, 500, 502, 503, 504):
                time.sleep(base_sleep * (2 ** attempt) + random.uniform(0, 0.4))
                continue
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(base_sleep * (2 ** attempt) + random.uniform(0, 0.4))
    raise RuntimeError("get_with_retry failed")

IMG_EXTS = {".jpg", ".jpeg", ".png"}

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def safe_name_from_url(url: str, ext_hint: str = ".jpg") -> str:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    ext = ext_hint if ext_hint.startswith(".") else f".{ext_hint}"
    return f"{h}{ext}"

def validate_min_size(path: Path, min_size: int) -> bool:
    try:
        with Image.open(path) as im:
            w, h = im.size
        return w >= min_size and h >= min_size
    except Exception:
        return False

def download_image(url: str, out_path: Path, *, min_size: int, timeout: int = 60, retries: int = 5) -> Optional[Path]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            r = get_with_retry(url, stream=True, timeout=timeout, retries=4, base_sleep=0.6)
            content = r.content
            if len(content) < 10_000:
                raise ValueError("too small download")

            out_path.write_bytes(content)
            with Image.open(out_path) as im:
                im.load()

            if not validate_min_size(out_path, min_size):
                out_path.unlink(missing_ok=True)
                return None

            return out_path
        except Exception:
            out_path.unlink(missing_ok=True)
            time.sleep((2 ** attempt) * 0.4 + random.uniform(0, 0.3))
    return None


def pexels_search(query: str, api_key: str, per_page: int = 80, page: int = 1) -> List[Dict[str, Any]]:
    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": api_key}
    params = {"query": query, "per_page": per_page, "page": page}
    r = get_with_retry(url, headers=headers, params=params, timeout=30, retries=5, base_sleep=0.8)
    data = r.json()
    return data.get("photos", [])

def pexels_pick_url(photo: Dict[str, Any]) -> Optional[str]:
    src = photo.get("src", {})
    return src.get("large") or src.get("medium") or src.get("original")

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"

@dataclass
class CommonsItem:
    url: str
    uniq: str 
    ext: str

def commons_search_titles(query: str, limit: int = 80) -> List[str]:
    titles: List[str] = []
    sroffset = 0
    while len(titles) < limit:
        batch = min(50, limit - len(titles))
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srnamespace": "6",
            "srlimit": batch,
            "sroffset": sroffset,
            "srsearch": f'{query} (jpg OR jpeg OR png) -svg -tiff -webm -ogg',
        }
        data = get_with_retry(WIKIMEDIA_API, params=params).json()
        results = data.get("query", {}).get("search", [])
        if not results:
            break
        for it in results:
            t = it.get("title")
            if t and t.startswith("File:"):
                titles.append(t)
        sroffset += len(results)
        if "continue" not in data:
            break
        time.sleep(0.25)
    seen, out = set(), []
    for t in titles:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def commons_fetch_items(titles: List[str], thumb_width: int = 640) -> List[CommonsItem]:
    items: List[CommonsItem] = []
    for i in range(0, len(titles), 50):
        chunk = titles[i:i+50]
        params = {
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "iiprop": "url|sha1|mime",
            "iiurlwidth": str(thumb_width),
            "titles": "|".join(chunk),
        }
        data = get_with_retry(WIKIMEDIA_API, params=params).json()
        pages = data.get("query", {}).get("pages", {})
        for _, page in pages.items():
            infos = page.get("imageinfo", [])
            if not infos:
                continue
            info = infos[0]
            url = info.get("thumburl") or info.get("url")
            sha1 = info.get("sha1")
            mime = info.get("mime", "")
            if not url or not sha1:
                continue
            if "png" in mime:
                ext = ".png"
            elif "jpeg" in mime or "jpg" in mime:
                ext = ".jpg"
            else:
                continue
            items.append(CommonsItem(url=url, uniq=sha1, ext=ext))
        time.sleep(0.2)
    uniq_map = {}
    for it in items:
        uniq_map[it.uniq] = it
    return list(uniq_map.values())


def wiki_pageimages_search(query: str, lang: str = "en", limit: int = 20) -> List[str]:
    api = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srlimit": str(limit),
        "srsearch": query,
    }
    data = get_with_retry(api, params=params).json()
    results = data.get("query", {}).get("search", [])
    titles = [r["title"] for r in results if "title" in r]
    return titles

def wiki_summary_thumb(title: str, lang: str = "en") -> Optional[str]:
    from urllib.parse import quote
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
    try:
        data = get_with_retry(url, timeout=30, retries=4, base_sleep=0.8).json()
        thumb = data.get("thumbnail", {})
        return thumb.get("source")
    except Exception:
        return None


GEN_SYSTEM = (
    "You create Visual Question Answering (VQA) training data.\n"
    "Rules:\n"
    "- Output MUST be valid JSON only.\n"
    "- Questions/answers in English.\n"
    "- Do NOT invent details. If uncertain, answer \"unknown\".\n"
    "- Keep answers short (1-8 words).\n"
    "- Prefer diverse question types.\n"
)
GEN_USER = (
    "Generate between 4 and 6 QA pairs for this image.\n"
    "Mix types when possible:\n"
    "1) what_in_photo: What is in the photo?\n"
    "2) attributes_color: What color is the X?\n"
    "3) spatial: Is X to the left of Y? / Is X above Y?\n"
    "4) existence: Is there a X in the image?\n"
    "5) counting: How many X are in the image?\n"
    "6) comparison: Which is larger, X or Y?\n"
    "If unreliable, skip.\n"
    "Return JSON: {\"qa\":[{\"type\":\"...\",\"question\":\"...\",\"answer\":\"...\"}, ...]}\n"
)

VERIFY_SYSTEM = (
    "You verify VQA QA pairs against an image.\n"
    "Be strict: if not clearly supported, mark invalid.\n"
    "Output JSON only."
)
VERIFY_USER = (
    "Check each QA pair. Return JSON:\n"
    "{\"results\":[{\"type\":\"...\",\"question\":\"...\",\"answer\":\"...\",\"valid\":true/false,\"reason\":\"short\"}]}\n"
    "If unsure -> valid=false."
)

def img_to_data_url(path: Path) -> str:
    ext = path.suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def openai_generate_qa(client: OpenAI, model: str, image_path: Path, detail: str, retries: int = 4) -> List[Dict[str, str]]:
    data_url = img_to_data_url(image_path)
    for attempt in range(retries):
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": GEN_SYSTEM}]},
                    {"role": "user", "content": [
                        {"type": "input_text", "text": GEN_USER},
                        {"type": "input_image", "image_url": data_url, "detail": detail},
                    ]},
                ],
                text={"format": {"type": "json_object"}},
            )
            data = json.loads(resp.output_text)
            qa = data.get("qa", [])
            out = []
            for x in qa:
                if isinstance(x, dict) and all(k in x for k in ("type", "question", "answer")):
                    t = str(x["type"]).strip()
                    q = str(x["question"]).strip()
                    a = str(x["answer"]).strip()
                    if t and q and a:
                        out.append({"type": t, "question": q, "answer": a})
            return out
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.3 * (attempt + 1))
    return []


def openai_verify_qa(client: OpenAI, model: str, image_path: Path, qa: List[Dict[str, str]], detail: str, retries: int = 4) -> List[Dict[str, str]]:
    if not qa:
        return []
    data_url = img_to_data_url(image_path)

    for attempt in range(retries):
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": VERIFY_SYSTEM}]},
                    {"role": "user", "content": [
                        {"type": "input_text", "text": VERIFY_USER},
                        {"type": "input_image", "image_url": data_url, "detail": detail},
                        {"type": "input_text", "text": json.dumps({"qa": qa}, ensure_ascii=False)},
                    ]},
                ],
                text={"format": {"type": "json_object"}},
            )
            data = json.loads(resp.output_text)
            results = data.get("results", [])
            kept = []
            for r in results:
                if isinstance(r, dict) and r.get("valid") is True:
                    t = str(r.get("type", "")).strip()
                    q = str(r.get("question", "")).strip()
                    a = str(r.get("answer", "")).strip()
                    if t and q and a:
                        kept.append({"type": t, "question": q, "answer": a})
            return kept
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.3 * (attempt + 1))
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=3000)
    ap.add_argument("--out_dir", default="images_3k")
    ap.add_argument("--out_json", default="vqa_3k_en.json")

    # speed / quality knobs
    ap.add_argument("--min_size", type=int, default=256)
    ap.add_argument("--download_workers", type=int, default=16)
    ap.add_argument("--thumb_width", type=int, default=640)

    # image sourcing
    ap.add_argument("--use_pexels", action="store_true", help="Use Pexels if PEXELS_API_KEY is set (recommended)")
    ap.add_argument("--pexels_per_page", type=int, default=80)
    ap.add_argument("--pexels_pages", type=int, default=3)

    ap.add_argument("--queries_per_topic", type=int, default=90)
    ap.add_argument("--max_attempts", type=int, default=20000)

    # QA (OpenAI paid)
    ap.add_argument("--qa_mode", default="openai", choices=["openai"], help="QA generation mode")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--workers", type=int, default=8, help="Parallel OpenAI requests")
    ap.add_argument("--low_ratio", type=float, default=0.75, help="Fraction detail=low, rest high")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--verify_ratio_high", type=float, default=0.30)

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect image URLs from sources
    print("[1/3] Collecting image URLs...")

    pexels_key = os.getenv("PEXELS_API_KEY", "").strip()
    use_pexels = args.use_pexels and bool(pexels_key)

    url_queue: List[Tuple[str, str]] = [] 
    used_uniq = set()

    queries = DEFAULT_QUERIES[:]
    rng.shuffle(queries)

    attempts = 0
    qi = 0

    while len(url_queue) < args.target * 2 and attempts < args.max_attempts:
        q = queries[qi % len(queries)]
        qi += 1
        attempts += 1
        if use_pexels:
            try:
                for page in range(1, args.pexels_pages + 1):
                    photos = pexels_search(q, pexels_key, per_page=args.pexels_per_page, page=page)
                    for ph in photos:
                        u = pexels_pick_url(ph)
                        if not u:
                            continue
                        uniq = f"pexels:{u}"
                        if uniq in used_uniq:
                            continue
                        used_uniq.add(uniq)
                        url_queue.append((u, uniq))
                        if len(url_queue) >= args.target * 2:
                            break
                    if len(url_queue) >= args.target * 2:
                        break
            except Exception:
                pass

        try:
            titles = commons_search_titles(q, limit=args.queries_per_topic)
            if titles:
                items = commons_fetch_items(titles, thumb_width=args.thumb_width)
                rng.shuffle(items)
                for it in items:
                    uniq = f"commons:{it.uniq}"
                    if uniq in used_uniq:
                        continue
                    used_uniq.add(uniq)
                    url_queue.append((it.url, uniq))
                    if len(url_queue) >= args.target * 2:
                        break
        except Exception:
            pass

        try:
            titles = wiki_pageimages_search(q, lang="en", limit=15)
            for t in titles:
                u = wiki_summary_thumb(t, lang="en")
                if not u:
                    continue
                uniq = f"wikipedia:{u}"
                if uniq in used_uniq:
                    continue
                used_uniq.add(uniq)
                url_queue.append((u, uniq))
                if len(url_queue) >= args.target * 2:
                    break
        except Exception:
            pass
        if len(url_queue) and len(url_queue) % 500 == 0:
            print(f"  collected URLs: {len(url_queue)}")
        if len(url_queue) >= args.target * 2:
            break

    if not url_queue:
        print("No image URLs collected. Check network / try --use_pexels with PEXELS_API_KEY.")
        return

    rng.shuffle(url_queue)

    # Download images in parallel
    print("[2/3] Downloading images...")
    downloaded: List[Path] = []
    seen_content_hash = set()

    def dl_job(idx: int, url: str) -> Optional[Path]:
        ext = ".jpg"
        if ".png" in url.lower():
            ext = ".png"
        name = safe_name_from_url(url, ext_hint=ext)
        path = out_dir / name

        if path.exists():
            if validate_min_size(path, args.min_size):
                return path
            path.unlink(missing_ok=True)

        tmp = download_image(url, path, min_size=args.min_size)
        if not tmp:
            return None
        h = sha256_bytes(tmp.read_bytes())
        if h in seen_content_hash:
            tmp.unlink(missing_ok=True)
            return None
        seen_content_hash.add(h)
        return tmp

    with ThreadPoolExecutor(max_workers=args.download_workers) as ex:
        futs = []
        for i, (u, _) in enumerate(url_queue):
            if len(downloaded) >= args.target:
                break
            futs.append(ex.submit(dl_job, i, u))

        for fut in tqdm(as_completed(futs), total=len(futs)):
            p = fut.result()
            if p:
                downloaded.append(p)
                if len(downloaded) >= args.target:
                    break

    print(f"Downloaded: {len(downloaded)} images -> {out_dir.resolve()}")
    if len(downloaded) == 0:
        print("Downloaded 0 images. Try: --thumb_width 320, lower --download_workers, or enable --use_pexels.")
        return

    # Generate QA via OpenAI
    print("[3/3] Generating VQA via OpenAI...")
    client = OpenAI()

    details = ["low" if rng.random() < args.low_ratio else "high" for _ in range(len(downloaded))]

    def gen_one(i: int) -> Dict[str, Any]:
        img_path = downloaded[i]
        detail = details[i]
        qa = openai_generate_qa(client, args.model, img_path, detail=detail)
        item = {"id": i, "image": str(img_path.as_posix()), "detail": detail, "qa": qa}
        return item

    dataset: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(gen_one, i): i for i in range(len(downloaded))}
        for fut in tqdm(as_completed(futs), total=len(futs)):
            i = futs[fut]
            try:
                dataset.append(fut.result())
            except Exception as e:
                dataset.append({"id": i, "image": str(downloaded[i].as_posix()), "qa": [], "error": str(e)})

    dataset.sort(key=lambda x: x["id"])

    if args.verify:
        print("Verifying QA pairs (strict)...")
        verify_details = ["high" if rng.random() < args.verify_ratio_high else "low" for _ in range(len(dataset))]

        def verify_one(item: Dict[str, Any]) -> Dict[str, Any]:
            i = item["id"]
            qa = item.get("qa", [])
            vdetail = verify_details[i]
            item["verify_detail"] = vdetail
            if not qa:
                return item
            kept = openai_verify_qa(client, args.model, Path(item["image"]), qa, detail=vdetail)
            item["qa"] = kept
            return item

        verified: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(2, args.workers // 2)) as ex:
            futs = {ex.submit(verify_one, it): it["id"] for it in dataset}
            for fut in tqdm(as_completed(futs), total=len(futs)):
                i = futs[fut]
                try:
                    verified.append(fut.result())
                except Exception as e:
                    fallback = next(x for x in dataset if x["id"] == i)
                    fallback["verify_error"] = str(e)
                    verified.append(fallback)
        verified.sort(key=lambda x: x["id"])
        dataset = verified

    Path(args.out_json).write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done JSON: {Path(args.out_json).resolve()}")
    print(f"Images folder: {out_dir.resolve()}")

    total_qa = sum(len(x.get("qa", [])) for x in dataset)
    print(f"QA pairs kept: {total_qa} (avg {total_qa / max(1, len(dataset)):.2f} per image)")


if __name__ == "__main__":
    main()