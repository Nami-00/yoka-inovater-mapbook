import json

in_files = [
    "building_centroid_all.geojson",
    "building_centroid_all2.geojson",
]

out_file = "building_centroid_merged.geojson"

features = []
crs_block = None
name_block = None

for path in in_files:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if crs_block is None:
        crs_block = data.get("crs")
        name_block = data.get("name")

    features.extend(data["features"])

merged = {
    "type": "FeatureCollection",
    "name": name_block,
    "crs": crs_block,
    "features": features,
}

with open(out_file, "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False)

print(f"結合完了: {out_file}")
print(f"総件数: {len(features)}")
