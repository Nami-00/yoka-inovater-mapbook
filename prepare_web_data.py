"""
Mapbox可視化用データ準備スクリプト
k=4〜9 の各クラスター結果を web_data/ へ変換
"""

import json
import geopandas as gpd
import pandas as pd
from pathlib import Path

# ディレクトリ設定
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / 'output'
WEB_DATA_DIR = BASE_DIR / 'web_data'

WEB_DATA_DIR.mkdir(exist_ok=True)

# k=4〜9
K_LIST = range(4, 10)

# 解析対象用途（列名の前提：建物_{用途}）
USAGE_NAMES = [
    '官公庁施設', '共同住宅', '住宅', '商業施設', '文教厚生施設',
    '業務施設', '商業系複合施設', '店舗等併用住宅', '店舗等併用共同住宅', '宿泊施設'
]


def simplify_geometry(input_geojson: Path, output_geojson: Path, tolerance=0.0001):
    """
    GeoJSON の幾何形状を簡略化してファイルサイズを削減
    Mapbox向けに EPSG:4326 に揃えてから書き出す
    """
    print(f"   簡略化処理: {input_geojson.name} → {output_geojson.name}")

    gdf = gpd.read_file(input_geojson)

    # Mapboxは基本EPSG:4326（経度緯度）
    if gdf.crs is None:
        print("   警告: 入力GeoJSONにCRSがありません。必要なら set_crs() を追加してください。")
    else:
        gdf = gdf.to_crs(epsg=4326)

    # geometry簡略化（ポリゴン想定）
    gdf["geometry"] = gdf["geometry"].simplify(tolerance, preserve_topology=True)

    gdf.to_file(output_geojson, driver="GeoJSON")

    original_size = input_geojson.stat().st_size / 1024 / 1024
    simplified_size = output_geojson.stat().st_size / 1024 / 1024
    if original_size > 0:
        print(f"   元ファイル: {original_size:.2f} MB → 簡略化後: {simplified_size:.2f} MB "
              f"({(1 - simplified_size/original_size)*100:.1f}% 削減)")
    else:
        print(f"   元ファイル: {original_size:.2f} MB → 簡略化後: {simplified_size:.2f} MB")


def _calc_usage_ratios_from_counts(cluster_data: pd.DataFrame) -> dict:
    """
    cluster_data（同一clusterのメッシュ行集合）から、
    建物用途の「棟数合計ベース」の比率を計算する。
    """
    total_buildings = float(cluster_data['建物総数'].sum())
    ratios = {}

    if total_buildings <= 0:
        for u in USAGE_NAMES:
            ratios[u] = 0.0
        return ratios

    for u in USAGE_NAMES:
        col = f'建物_{u}'
        ratios[u] = float(cluster_data[col].sum()) / total_buildings if col in cluster_data.columns else 0.0

    return ratios


def get_cluster_name_and_color(cluster_data: pd.DataFrame):
    """
    改良版命名:
    - 宿泊施設地域を追加
    - 住宅（住宅+共同住宅）の合算判定を追加
    - 平均建物数が極小かつ飲食突出は「要確認」を追加
    """
    # 例外：建物がほぼ無い（or整合性が怪しい）
    avg_buildings = float(cluster_data['建物総数'].mean()) if len(cluster_data) else 0.0
    avg_restaurants = float(cluster_data['飲食店数'].mean()) if len(cluster_data) else 0.0

    if avg_buildings < 1:
        if avg_restaurants > 50:
            return "要確認（建物0/飲食突出）", "#c0392b"  # 濃い赤
        return "低密度地域", "#95a5a6"  # グレー

    ratios = _calc_usage_ratios_from_counts(cluster_data)

    商業比率 = ratios.get('商業施設', 0)
    共同住宅比率 = ratios.get('共同住宅', 0)
    住宅比率 = ratios.get('住宅', 0)
    店舗等併用共同住宅比率 = ratios.get('店舗等併用共同住宅', 0)
    店舗等併用住宅比率 = ratios.get('店舗等併用住宅', 0)
    商業系複合施設比率 = ratios.get('商業系複合施設', 0)
    業務施設比率 = ratios.get('業務施設', 0)
    文教厚生施設比率 = ratios.get('文教厚生施設', 0)
    官公庁施設比率 = ratios.get('官公庁施設', 0)
    宿泊施設比率 = ratios.get('宿泊施設', 0)

    # 1) 単独支配（用途が明確なものを優先）
    if 商業系複合施設比率 >= 0.6:
        return "複合商業地域", "#ffd54f"
    if 宿泊施設比率 >= 0.5:
        return "宿泊施設地域", "#e91e63"
    if 文教厚生施設比率 >= 0.5:
        return "文教施設地域", "#3498db"
    if 官公庁施設比率 >= 0.5:
        return "官公庁施設地域", "#9b59b6"
    if 業務施設比率 >= 0.4:
        return "業務地域", "#e74c3c"

    # 2) 商業（飲食の強さで分ける）
    if 商業比率 > 0.5 and avg_restaurants > 100:
        return "超高密度商業地域", "#f39c12"
    if 商業比率 > 0.5:
        return "商業集積地域", "#f1c40f"
    if 商業比率 > 0.1:
        return "商業混在地域", "#f4d03f"

    # 3) 住宅系：住宅+共同住宅の合算で拾う（クラスタ5対策）
    res_sum = 住宅比率 + 共同住宅比率
    if res_sum >= 0.6:
        if 住宅比率 >= 0.6:
            return "戸建住宅地域", "#27ae60"
        if 共同住宅比率 >= 0.4:
            return "集合住宅地域", "#7dcea0"
        return "住宅混合地域（戸建＋集合）", "#58d68d"  # 中間の緑

    # 4) 併用住宅
    if 店舗等併用共同住宅比率 >= 0.2:
        return "店舗併用集合住宅地域", "#aed581"
    if 店舗等併用住宅比率 >= 0.2:
        return "店舗併用住宅地域", "#c5e1a5"

    # 5) 低密度
    if float(cluster_data['建物総数'].sum()) / max(len(cluster_data), 1) < 50:
        return "低密度地域", "#95a5a6"

    return "混合地域", "#e67e22"


def extract_cluster_config(csv_file: Path, output_json: Path, k: int):
    """CSV からクラスター統計情報を抽出して JSON 化"""
    print(f"   統計情報抽出: {csv_file.name} → {output_json.name}")

    df = pd.read_csv(csv_file)
    cluster_col = 'cluster'

    if cluster_col not in df.columns:
        print(f"   警告: '{cluster_col}' 列が見つかりません")
        print(f"   利用可能な列: {list(df.columns)}")
        return

    cluster_stats = []
    for cluster_id in sorted(df[cluster_col].unique()):
        cluster_data = df[df[cluster_col] == cluster_id]

        cluster_name, cluster_color = get_cluster_name_and_color(cluster_data)

        building_types = {}
        for u in USAGE_NAMES:
            col = f'建物_{u}'
            if col in cluster_data.columns:
                building_types[u] = int(cluster_data[col].sum())
            else:
                building_types[u] = 0

        stats = {
            'id': int(cluster_id),
            'name': cluster_name,
            'color': cluster_color,
            'count': int(len(cluster_data)),
            'avg_buildings': float(cluster_data['建物総数'].mean()) if len(cluster_data) else 0.0,
            'avg_restaurants': float(cluster_data['飲食店数'].mean()) if len(cluster_data) else 0.0,
            'sum_buildings': int(cluster_data['建物総数'].sum()) if len(cluster_data) else 0,
            'sum_restaurants': int(cluster_data['飲食店数'].sum()) if len(cluster_data) else 0,
            'building_types': building_types,
            # Mapboxの凡例/ツールチップ用に主要比率も入れておく（任意）
            'ratios': _calc_usage_ratios_from_counts(cluster_data)
        }
        cluster_stats.append(stats)

    config_json = {
        'cluster_count': k,
        'total_meshes': int(len(df)),
        'clusters': cluster_stats
    }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(config_json, f, ensure_ascii=False, indent=2)

    print(f"   完了: {k}個のクラスター統計を出力")
    for stats in cluster_stats:
        print(f"      クラスター {stats['id']}: {stats['name']} ({stats['color']})")
    print()


def main():
    """メイン処理: k=4〜9 の全データを変換"""
    print("=" * 60)
    print("Mapbox可視化用データ準備")
    print("=" * 60)
    print()

    for k in K_LIST:
        print(f"--- クラスター数: {k} ---")

        input_geojson = OUTPUT_DIR / f'k{k:02d}' / 'mesh_with_clusters.geojson'
        input_csv = OUTPUT_DIR / f'k{k:02d}' / 'mesh_with_clusters.csv'

        output_geojson = WEB_DATA_DIR / f'mesh_clusters_k{k}.geojson'
        output_json = WEB_DATA_DIR / f'cluster_config_k{k}.json'

        if not input_geojson.exists():
            print(f"   スキップ: {input_geojson} が見つかりません\n")
            continue

        simplify_geometry(input_geojson, output_geojson)

        if input_csv.exists():
            extract_cluster_config(input_csv, output_json, k)
        else:
            print(f"   統計情報スキップ: {input_csv} が見つかりません\n")

    print("=" * 60)
    print("データ準備完了")
    print("=" * 60)
    print(f"\n出力先: {WEB_DATA_DIR}")
    print("\n次のステップ:")
    print("1) index.html の mapbox.accessToken を設定")
    print("2) ローカルで動作確認: python -m http.server 8000")
    print("3) ブラウザで http://localhost:8000 を開く\n")


if __name__ == '__main__':
    main()