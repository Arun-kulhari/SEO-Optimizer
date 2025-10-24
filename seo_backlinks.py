import requests, gzip, io, json, pandas as pd
from warcio.archiveiterator import ArchiveIterator
from tqdm.notebook import tqdm

# 💡 Change this to your desired domain
TARGET_DOMAIN = "amazon.com"

# Choose a Common Crawl crawl (update as needed)
CRAWL = "CC-MAIN-2025-26"

# Number of WAT files to scan (increase for deeper results)
NUM_WAT_FILES = 2

def get_wat_paths(crawl, num_paths=NUM_WAT_FILES):
    url = f"https://data.commoncrawl.org/crawl-data/{crawl}/wat.paths.gz"
    resp = requests.get(url)
    resp.raise_for_status()
    paths = gzip.decompress(resp.content).decode().splitlines()
    return paths[:num_paths]

# Step 1: Get first few WAT paths
wat_paths = get_wat_paths(CRAWL)
print("✅ Retrieved WAT file paths:", wat_paths)

# Step 2: Scan WATs for backlinks
backlinks = set()

for path in tqdm(wat_paths, desc="Scanning WAT files"):
    wat_url = f"https://data.commoncrawl.org/{path}"
    print("📥 Fetching:", wat_url)
    resp = requests.get(wat_url, timeout=30)
    if resp.status_code != 200:
        print("❌ Download failed. Status:", resp.status_code)
        continue

    stream = io.BytesIO(resp.content)
    try:
        for record in ArchiveIterator(stream):
            if record.rec_type != "metadata":
                continue
            try:
                data = json.loads(record.content_stream().read())
                envelope = data.get("Envelope", {})
                src = envelope.get("WARC-Header-Metadata", {}).get("WARC-Target-URI", "")
                links = (envelope
                         .get("Payload-Metadata", {})
                         .get("HTTP-Response-Metadata", {})
                         .get("HTML-Metadata", {})
                         .get("Links", []))
                for l in links:
                    href = l.get("url") or l.get("href") or ""
                    if TARGET_DOMAIN in href:
                        backlinks.add((src, href))
            except Exception:
                continue
    except Exception as e:
        print("⚠ Archive read failed:", e)

# Step 3: Display results
df = pd.DataFrame(list(backlinks), columns=["Source Page", "Backlink URL"])
if df.empty:
    print(f"❌ No backlinks found for {TARGET_DOMAIN}")
else:
    print(f"✅ Found {len(df)} unique backlinks to {TARGET_DOMAIN}")
    display(df.head(20))