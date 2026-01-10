"""
福岡市・北九州市 250mメッシュ分析
建物用途と飲食店データを250mメッシュに集計
"""
import geopandas as gpd
import pandas as pd
from pathlib import Path
from datetime import datetime

# ==================== 設定 ====================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'output'
REPORT_DIR = OUTPUT_DIR

# 入力ファイル
INPUT_MESH_DIR = DATA_DIR / 'mesh_shapefiles'  # mesh ZIPファイルを解凍したディレクトリ
INPUT_BUILDING_FILE = DATA_DIR / 'building_centroid_all.geojson'
INPUT_FOOD_FILE = DATA_DIR / 'fukuoka_40100_food_business_all.csv'

# 出力ファイル
OUTPUT_MESH_RESULT_GEOJSON = OUTPUT_DIR / 'mesh_analysis_result.geojson'
OUTPUT_MESH_RESULT_CSV = OUTPUT_DIR / 'mesh_analysis_result.csv'

# 分析対象の建物用途（追加用途を含む）
TARGET_USAGES = {
    '421': '官公庁施設',
    '412': '共同住宅',
    '411': '住宅',
    '402': '商業施設',
    '422': '文教厚生施設',
    '401': '業務施設',
    '404': '商業系複合施設',
    '413': '店舗等併用住宅',
    '414': '店舗等併用共同住宅',
    '403': '宿泊施設'
}


# 飲食店データの緯度経度範囲（福岡市・北九州市）
FOOD_LAT_MIN = 33.0
FOOD_LAT_MAX = 34.0
FOOD_LON_MIN = 130.0
FOOD_LON_MAX = 131.0

# ==================== 出力ディレクトリ作成 ====================
OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("🏗️  福岡市・北九州市 250mメッシュ分析")
print("=" * 70)
print()

# ==================== 1. メッシュデータ読み込み ====================
print("📂 [1/6] メッシュデータ読み込み")

mesh_files = list(INPUT_MESH_DIR.rglob('*.shp'))
if not mesh_files:
    raise FileNotFoundError(f"❌ {INPUT_MESH_DIR} に .shp ファイルが見つかりません")

print(f"   検出ファイル数: {len(mesh_files)}")

mesh_gdfs = []
for file in mesh_files:
    print(f"   読み込み中: {file.name}")
    gdf = gpd.read_file(file, encoding='shift-jis')
    mesh_gdfs.append(gdf)

mesh = pd.concat(mesh_gdfs, ignore_index=True)
print(f"   総メッシュ数: {len(mesh):,}")

# CRS変換（WGS84）
if mesh.crs != 'EPSG:4326':
    print("   CRS変換: EPSG:4326 (WGS84)")
    mesh = mesh.to_crs('EPSG:4326')

# mesh_code 作成
if 'KEY_CODE' in mesh.columns:
    mesh['mesh_code'] = mesh['KEY_CODE']
elif 'MESH_CODE' in mesh.columns:
    mesh['mesh_code'] = mesh['MESH_CODE']
else:
    mesh['mesh_code'] = mesh.index.astype(str)

print(f"✅ メッシュ準備完了\n")

# ==================== 2. 建物データ読み込み ====================
print("🏢 [2/6] 建物データ読み込み")

if not INPUT_BUILDING_FILE.exists():
    raise FileNotFoundError(f"❌ {INPUT_BUILDING_FILE} が見つかりません")

building_gdf = gpd.read_file(INPUT_BUILDING_FILE)
print(f"総建物数: {len(building_gdf):,}")

# usage_ja フィールドがない場合は usage コードから生成
if 'usage_ja' not in building_gdf.columns:
    print("usage_ja フィールドが存在しないため、usage コードから生成します")
    building_gdf['usage_ja'] = building_gdf['usage'].map(TARGET_USAGES)
    print(f"usage_ja フィールドを追加しました")
else:
    # usage_ja が空の場合も usage コードから補完
    mask = building_gdf['usage_ja'].isna() | (building_gdf['usage_ja'] == '')
    if mask.any():
        print(f"usage_ja が空のレコード {mask.sum():,} 件を補完します")
        building_gdf.loc[mask, 'usage_ja'] = building_gdf.loc[mask, 'usage'].map(TARGET_USAGES)

# 対象用途のみ抽出
building_gdf = building_gdf[building_gdf['usage'].isin(TARGET_USAGES.keys())].copy()
print(f"対象建物数: {len(building_gdf):,}")

# 座標取得
if 'cx' in building_gdf.columns and 'cy' in building_gdf.columns:
    building_gdf['geometry'] = gpd.points_from_xy(building_gdf['cx'], building_gdf['cy'])
    building_gdf = gpd.GeoDataFrame(building_gdf, geometry='geometry', crs='EPSG:4326')

print(f"✅ 建物データ準備完了\n")

# ==================== 3. 飲食店データ読み込み ====================
print("🍽️  [3/6] 飲食店データ読み込み")

if not INPUT_FOOD_FILE.exists():
    raise FileNotFoundError(f"❌ {INPUT_FOOD_FILE} が見つかりません")

food = pd.read_csv(INPUT_FOOD_FILE, encoding='utf-8-sig')
print(f"   総飲食店数: {len(food):,}")

# 緯度経度の有効性確認
food = food.dropna(subset=['緯度', '経度'])
food = food[
    (food['緯度'] >= FOOD_LAT_MIN) & (food['緯度'] <= FOOD_LAT_MAX) &
    (food['経度'] >= FOOD_LON_MIN) & (food['経度'] <= FOOD_LON_MAX)
]
print(f"   有効飲食店数: {len(food):,}")

# GeoDataFrame化
food_gdf = gpd.GeoDataFrame(
    food,
    geometry=gpd.points_from_xy(food['経度'], food['緯度']),
    crs='EPSG:4326'
)

print(f"✅ 飲食店データ準備完了\n")

# ==================== 4. 空間結合: 建物 → メッシュ ====================
print("🔗 [4/6] 空間結合: 建物 → メッシュ")

building_gdf_in_mesh = gpd.sjoin(building_gdf, mesh, how='inner', predicate='within')
print(f"   結合レコード数: {len(building_gdf_in_mesh):,}")

# 用途別集計
building_counts = building_gdf_in_mesh.groupby(['mesh_code', 'usage_ja']).size().reset_index(name='count')
building_pivot = building_counts.pivot(index='mesh_code', columns='usage_ja', values='count').fillna(0)

# カラム名に接頭辞を追加
building_pivot.columns = ['建物_' + col for col in building_pivot.columns]
building_pivot = building_pivot.reset_index()

print(f"✅ 建物集計完了: {len(building_pivot):,} メッシュ\n")

# ==================== 5. 空間結合: 飲食店 → メッシュ ====================
print("🔗 [5/6] 空間結合: 飲食店 → メッシュ")

food_in_mesh = gpd.sjoin(food_gdf, mesh, how='inner', predicate='within')
print(f"   結合レコード数: {len(food_in_mesh):,}")

food_counts = food_in_mesh.groupby('mesh_code').size().reset_index(name='飲食店数')

print(f"✅ 飲食店集計完了: {len(food_counts):,} メッシュ\n")

# ==================== 6. 結果統合 ====================
print("📊 [6/6] 結果統合")

# メッシュに集計結果を結合
result = mesh.copy()
result = result.merge(building_pivot, on='mesh_code', how='left')
result = result.merge(food_counts, on='mesh_code', how='left')

# 欠損値を0埋め
for col in result.columns:
    if col.startswith('建物_') or col == '飲食店数':
        result[col] = result[col].fillna(0).astype(int)

# 建物総数計算
building_cols = [col for col in result.columns if col.startswith('建物_')]
result['建物総数'] = result[building_cols].sum(axis=1)

# 建物または飲食店があるメッシュのみ保存
result_filtered = result[(result['建物総数'] > 0) | (result['飲食店数'] > 0)].copy()

# 中心座標追加
result_filtered['中心_経度'] = result_filtered.geometry.centroid.x
result_filtered['中心_緯度'] = result_filtered.geometry.centroid.y

print(f"   有効メッシュ数: {len(result_filtered):,}")
print(f"   総建物数: {result_filtered['建物総数'].sum():,}")
print(f"   総飲食店数: {result_filtered['飲食店数'].sum():,}")

# ==================== 保存 ====================
print("\n💾 結果保存中...")

# GeoJSON保存
result_filtered.to_file(OUTPUT_MESH_RESULT_GEOJSON, driver='GeoJSON', encoding='utf-8')
print(f"   ✅ {OUTPUT_MESH_RESULT_GEOJSON}")

# CSV保存（geometry除外）
result_csv = result_filtered.drop(columns=['geometry'])
result_csv.to_csv(OUTPUT_MESH_RESULT_CSV, index=False, encoding='utf-8')
print(f"   ✅ {OUTPUT_MESH_RESULT_CSV}")

# ==================== サマリーレポート ====================
print("\n📝 サマリーレポート作成中...")

summary_report = f"""# 福岡市・北九州市 250mメッシュ分析サマリー

## 基本情報
- **分析日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **総メッシュ数**: {len(result_filtered):,}
- **総建物数**: {result_filtered['建物総数'].sum():,}
- **総飲食店数**: {result_filtered['飲食店数'].sum():,}

## 建物用途別集計
"""

for col in sorted(building_cols):
    usage_name = col.replace('建物_', '')
    count = result_filtered[col].sum()
    summary_report += f"- **{usage_name}**: {count:,}棟\n"

summary_report += f"""
## トップ10メッシュ（建物総数）
"""

top10 = result_filtered.nlargest(10, '建物総数')[['mesh_code', '建物総数', '飲食店数', '中心_経度', '中心_緯度']]
for idx, row in top10.iterrows():
    summary_report += f"- メッシュ {row['mesh_code']}: 建物{row['建物総数']:,}棟, 飲食店{row['飲食店数']:,}件 ({row['中心_緯度']:.5f}, {row['中心_経度']:.5f})\n"

with open(REPORT_DIR / 'analysis_summary.md', 'w', encoding='utf-8') as f:
    f.write(summary_report)

print(f"   ✅ {REPORT_DIR / 'analysis_summary.md'}")

print("\n" + "=" * 70)
print("✅ 分析完了！")
print("=" * 70)
print(f"\n📁 出力先: {OUTPUT_DIR}")
print(f"   - {OUTPUT_MESH_RESULT_GEOJSON.name}")
print(f"   - {OUTPUT_MESH_RESULT_CSV.name}")
print(f"   - analysis_summary.md")
print("\n次のステップ: python 2_cluster_analysis_multi.py")
