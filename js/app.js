// Mapbox GL JS を用いたクラスター分析および駅乗降客数モードのビューア
//
// このスクリプトでは、もともとのクラスター分析機能に加えて、タブで
// 「商圏メッシュ」モードと「乗降客数」モードを切り替えられるようにし、
// 乗降客数モードでは駅のシンボルサイズを常に 4pt（4px 相当）に固定して
// 表示します。また、駅の乗降客数に応じて 300m/500m/1000m のバッファー
// （出店候補エリア）を自動生成して表示する機能も備えています。駅が
// 150m 未満の距離で隣接している場合は統合し、合計乗降客数でバッファー
// 半径を決定します。最も乗降客数の多い駅をバッファーの中心とします。

// Mapbox アクセストークン（公開されているためそのまま利用）
mapboxgl.accessToken = 'pk.eyJ1IjoibmFtaTAwIiwiYSI6ImNtazZnc2RnbzBvdnEzZXI1ZHhlN2Yyc3gifQ.ikUmsp1TSWrZNMteouL9aQ';

// グローバル変数の定義
let map;
let currentClusterCount = 6;        // 初期クラスター数
let currentDisplayMode = 'cluster';  // 表示モード（cluster/density/buildings/用途別）
let meshData = null;                 // メッシュ GeoJSON
let clusterConfig = null;            // クラスター設定情報
let visibleClusters = new Set();     // 可視クラスター ID 集合
let stationData = null;              // 駅データ GeoJSON

// モード切り替え用の状態
let currentMode = 'passengers';       // 'mesh' または 'passengers'（初期は乗降客数モード）
let passengerShowStations = true;    // 駅表示フラグ
let passengerShowStoreAreas = true;  // 出店エリア表示フラグ

// バッファー生成用の閾値
const MERGE_DISTANCE_METERS = 150;   // 駅間距離がこの値未満なら統合（m）
const RADIUS_RULES = [               // 乗降客数に応じたバッファー半径（km）
  { max: 30000, radius: 0.3 },       // 3万未満 → 0.3km (300m)
  { max: 50000, radius: 0.5 },       // 3万〜5万未満 → 0.5km (500m)
  { max: Infinity, radius: 1.0 }     // 5万以上 → 1.0km (1000m)
];

// DOM 準備完了時に初期化処理を実行
document.addEventListener('DOMContentLoaded', () => {
  initMap();
  setupControls();
});

// 地図の初期化とデータ読み込み
function initMap() {
  // 地図を初期化（国土地理院パレット地図を背景に使用）
  map = new mapboxgl.Map({
    container: 'map',
    style: {
      version: 8,
      sources: {
        'gsi-pale': {
          type: 'raster',
          tiles: ['https://cyberjapandata.gsi.go.jp/xyz/pale/{z}/{x}/{y}.png'],
          tileSize: 256,
          attribution: '<a href="https://maps.gsi.go.jp/development/ichiran.html">国土地理院</a>'
        }
      },
      layers: [
        {
          id: 'gsi-pale-layer',
          type: 'raster',
          source: 'gsi-pale',
          minzoom: 0,
          maxzoom: 18
        }
      ]
    },
    center: [130.4017, 33.5904], // 福岡市付近
    zoom: 10,
    pitch: 0,
    bearing: 0
  });

  // ナビゲーションコントロール
  map.addControl(new mapboxgl.NavigationControl(), 'top-right');

  // 地図が読み込まれたらデータを読み込む
  map.on('load', () => {
    loadClusterData(currentClusterCount);
    loadStationData();
  });
}

// 各種 UI のイベントリスナー設定
function setupControls() {
  // クラスター数の変更
  const clusterCountSelect = document.getElementById('cluster-count');
  if (clusterCountSelect) {
    clusterCountSelect.addEventListener('change', (e) => {
      currentClusterCount = parseInt(e.target.value);
      // 既存レイヤーを削除して再読み込み
      if (map.getLayer('mesh-fill')) map.removeLayer('mesh-fill');
      if (map.getLayer('mesh-outline')) map.removeLayer('mesh-outline');
      if (map.getSource('mesh-data')) map.removeSource('mesh-data');
      loadClusterData(currentClusterCount);
    });
  }

  // 表示モード変更（密度・建物総数・用途別）
  const displayModeSelect = document.getElementById('display-mode');
  if (displayModeSelect) {
    displayModeSelect.addEventListener('change', (e) => {
      currentDisplayMode = e.target.value;
      updateMapStyle();
      updateLegend();
    });
  }

  // 透明度スライダー
  const opacitySlider = document.getElementById('opacity-slider');
  if (opacitySlider) {
    opacitySlider.addEventListener('input', (e) => {
      const value = parseInt(e.target.value);
      document.getElementById('opacity-value').textContent = `${value}%`;
      const opacity = value / 100;
      if (map.getLayer('mesh-fill')) {
        map.setPaintProperty('mesh-fill', 'fill-opacity', opacity);
      }
    });
  }

  // リセットボタン
  const resetButton = document.getElementById('reset-view');
  if (resetButton) {
    resetButton.addEventListener('click', () => {
      map.flyTo({
        center: [130.4017, 33.5904],
        zoom: 10,
        pitch: 0,
        bearing: 0,
        duration: 1500
      });
      if (clusterConfig) {
        visibleClusters = new Set(clusterConfig.clusters.map(c => c.id));
        updateClusterFilters();
        updateMapStyle();
        updateStatistics();
      }
    });
  }

  // パッセンジャーモードのチェックボックス
  const chkStations = document.getElementById('passenger-show-stations');
  if (chkStations) {
    passengerShowStations = chkStations.checked;
    chkStations.addEventListener('change', (e) => {
      passengerShowStations = e.target.checked;
      updateStationDisplay();
    });
  }
  const chkStoreAreas = document.getElementById('passenger-show-store-areas');
  if (chkStoreAreas) {
    passengerShowStoreAreas = chkStoreAreas.checked;
    chkStoreAreas.addEventListener('change', (e) => {
      passengerShowStoreAreas = e.target.checked;
      updateStoreAreas();
    });
  }

  // タブの切り替えイベント
  const tabPassengers = document.getElementById('tab-passengers');
  const tabMesh = document.getElementById('tab-mesh');
  if (tabPassengers) {
    tabPassengers.addEventListener('click', () => switchMode('passengers'));
  }
  if (tabMesh) {
    tabMesh.addEventListener('click', () => switchMode('mesh'));
  }
}

// 駅データを読み込む
async function loadStationData() {
  try {
    const response = await fetch('web_data/stations.geojson');
    stationData = await response.json();
    console.log(`駅データ読み込み完了: ${stationData.features.length} 駅`);
    // 駅データを読み込んだら、現在のモードに応じて表示を更新
    updateStationDisplay();
    updateStoreAreas();
  } catch (err) {
    console.error('駅データ読み込みエラー:', err);
  }
}

// クラスター（メッシュ）データを読み込む
async function loadClusterData(k) {
  try {
    const [geojsonResponse, configResponse] = await Promise.all([
      fetch(`web_data/mesh_clusters_k${k}.geojson`),
      fetch(`web_data/cluster_config_k${k}.json`)
    ]);
    meshData = await geojsonResponse.json();
    clusterConfig = await configResponse.json();
    console.log(`メッシュデータ読み込み完了: ${meshData.features.length} メッシュ`);
    visibleClusters = new Set(clusterConfig.clusters.map(c => c.id));
    // 地図が既にスタイル読み込み済みかを確認
    if (map.isStyleLoaded()) {
      updateMap();
      updateUI();
    } else {
      map.once('idle', () => {
        updateMap();
        updateUI();
      });
    }
  } catch (err) {
    console.error('メッシュデータ読み込みエラー:', err);
    alert('メッシュデータの読み込みに失敗しました。');
  }
}

// メッシュレイヤーを更新（ロード後に呼び出し）
function updateMap() {
  if (!map.isStyleLoaded() || !meshData) {
    console.log('地図またはデータが未準備');
    return;
  }
  // 既存レイヤーやソースを削除
  if (map.getLayer('mesh-fill')) map.removeLayer('mesh-fill');
  if (map.getLayer('mesh-outline')) map.removeLayer('mesh-outline');
  if (map.getSource('mesh-data')) map.removeSource('mesh-data');
  // 新しいソースを追加
  map.addSource('mesh-data', { type: 'geojson', data: meshData });
  // メッシュ塗りつぶしレイヤー
  map.addLayer({
    id: 'mesh-fill',
    type: 'fill',
    source: 'mesh-data',
    paint: {
      'fill-opacity': 0.7
    },
    layout: {
      // mesh モード時のみ表示、passengers モードでは非表示にする
      'visibility': currentMode === 'mesh' ? 'visible' : 'none'
    }
  });
  // メッシュ境界線レイヤー
  map.addLayer({
    id: 'mesh-outline',
    type: 'line',
    source: 'mesh-data',
    paint: {
      'line-color': '#666',
      'line-width': 0.5,
      'line-opacity': 0.3
    },
    layout: {
      'visibility': currentMode === 'mesh' ? 'visible' : 'none'
    }
  });
  // メッシュレイヤーのポップアップイベント（クラスター情報表示）
  if (!map._clusterEventListenersAdded) {
    map.on('click', 'mesh-fill', (e) => {
      const properties = e.features[0].properties;
      let html = '<div style="max-width: 300px;">';
      html += `<h3>メッシュ情報</h3>`;
      html += `<p><strong>クラスター:</strong> ${properties['cluster']}</p>`;
      if (clusterConfig) {
        const cluster = clusterConfig.clusters.find(c => c.id == properties['cluster']);
        if (cluster) {
          html += `<p><strong>クラスター名:</strong> ${cluster.name}</p>`;
        }
      }
      html += `<p><strong>建物総数:</strong> ${properties['建物総数']}</p>`;
      html += `<p><strong>飲食店数:</strong> ${properties['飲食店数']}</p>`;
      // 用途別建物数を割合付きで表示
      const usages = ['官公庁施設', '共同住宅', '住宅', '商業施設', '文教厚生施設',
                      '業務施設', '商業系複合施設', '店舗等併用住宅', '店舗等併用共同住宅', '宿泊施設'];
      const totalBuildings = properties['建物総数'];
      usages.forEach((usage) => {
        const field = '建物_' + usage;
        if (properties[field] && properties[field] > 0) {
          const count = properties[field];
          const ratio = totalBuildings > 0 ? (count / totalBuildings * 100).toFixed(1) : 0;
          html += `<p><strong>${usage}:</strong> ${count} (${ratio}%)</p>`;
        }
      });
      html += '</div>';
      new mapboxgl.Popup()
        .setLngLat(e.lngLat)
        .setHTML(html)
        .addTo(map);
    });
    map.on('mouseenter', 'mesh-fill', () => {
      map.getCanvas().style.cursor = 'pointer';
    });
    map.on('mouseleave', 'mesh-fill', () => {
      map.getCanvas().style.cursor = '';
    });
    map._clusterEventListenersAdded = true;
  }
  // カラースタイル更新
  updateMapStyle();
}

// 駅レイヤーの表示を更新
function updateStationDisplay() {
  // 地図とデータが読み込まれているか確認
  if (!map || !map.isStyleLoaded() || !stationData) return;
  // 既存レイヤー/ソースの削除
  if (map.getLayer('stations')) map.removeLayer('stations');
  if (map.getLayer('station-labels')) map.removeLayer('station-labels');
  if (map.getSource('stations')) map.removeSource('stations');
  // passengers モードかつチェックボックスがオンでなければ表示しない
  if (currentMode !== 'passengers' || !passengerShowStations) {
    updateLegend();
    return;
  }
  // ポリゴンジオメトリをポイントに変換
  const pointFeatures = stationData.features.map((feature) => {
    // GeoJSON では駅はポリゴンとして定義されている（0番目の座標を使用）
    const coords = feature.geometry.coordinates[0][0];
    return {
      type: 'Feature',
      properties: feature.properties,
      geometry: {
        type: 'Point',
        coordinates: coords
      }
    };
  });
  const sourceData = { type: 'FeatureCollection', features: pointFeatures };
  // ソースを追加
  map.addSource('stations', { type: 'geojson', data: sourceData });
  // サークルレイヤー追加（半径を固定 4px、色は乗降客数に応じて設定）
  map.addLayer({
    id: 'stations',
    type: 'circle',
    source: 'stations',
    paint: {
      'circle-radius': 4,
      // 駅の色分けを5000人刻みに変更
      'circle-color': [
        'step', ['get', '乗降客数2023'],
        '#ffffcc',        // 0-4,999人
        5000, '#ffeda0',   // 5,000-9,999人
        10000, '#fed976',  // 10,000-14,999人
        15000, '#feb24c',  // 15,000-19,999人
        20000, '#fd8d3c',  // 20,000-24,999人
        25000, '#fc4e2a',  // 25,000-29,999人
        30000, '#e31a1c',  // 30,000-34,999人
        35000, '#bd0026',  // 35,000-39,999人
        40000, '#800026',  // 40,000-44,999人
        45000, '#67001f',  // 45,000-49,999人
        50000, '#4d0018'   // 50,000人以上
      ],
      'circle-opacity': 0.8,
      'circle-stroke-width': 1,
      'circle-stroke-color': '#ffffff'
    },
    layout: {
      'visibility': 'visible'
    }
  });
  // 駅名ラベルレイヤー
  map.addLayer({
    id: 'station-labels',
    type: 'symbol',
    source: 'stations',
    layout: {
      'text-field': ['get', '駅名'],
      'text-font': ['Open Sans Regular'],
      'text-size': 12,
      'text-anchor': 'top',
      'text-offset': [0, 1]
    },
    paint: {
      'text-color': '#000000',
      'text-halo-color': '#ffffff',
      'text-halo-width': 2
    }
  });
  // 駅クリックイベント（ポップアップ）
  if (!map._stationEventListenersAdded) {
    map.on('click', 'stations', (e) => {
      const props = e.features[0].properties;
      const html = `
        <div style="max-width: 250px;">
          <h3>${props['駅名']}</h3>
          <p><strong>運営会社:</strong> ${props['運営会社']}</p>
          <p><strong>路線名:</strong> ${props['路線名']}</p>
          <p><strong>乗降客数(2023):</strong> ${props['乗降客数2023']?.toLocaleString()}人</p>
        </div>
      `;
      new mapboxgl.Popup()
        .setLngLat(e.lngLat)
        .setHTML(html)
        .addTo(map);
    });
    map.on('mouseenter', 'stations', () => {
      map.getCanvas().style.cursor = 'pointer';
    });
    map.on('mouseleave', 'stations', () => {
      map.getCanvas().style.cursor = '';
    });
    map._stationEventListenersAdded = true;
  }
  updateLegend();
}

// 出店エリア（バッファー）を生成・表示
function updateStoreAreas() {
  if (!map || !map.isStyleLoaded() || !stationData) return;
  // 既存レイヤー/ソースの削除
  if (map.getLayer('store-area-fills')) map.removeLayer('store-area-fills');
  if (map.getSource('store-areas')) map.removeSource('store-areas');
  // passengers モードかつチェックボックスがオンでなければ描画しない
  if (currentMode !== 'passengers' || !passengerShowStoreAreas) {
    updateLegend();
    return;
  }
  // ポイントを抽出
  const pointFeatures = stationData.features.map((feature) => {
    const coords = feature.geometry.coordinates[0][0];
    return {
      type: 'Feature',
      properties: feature.properties,
      geometry: {
        type: 'Point',
        coordinates: coords
      }
    };
  });
  // グループ化処理
  const visited = new Array(pointFeatures.length).fill(false);
  const bufferFeatures = [];
  for (let i = 0; i < pointFeatures.length; i++) {
    if (visited[i]) continue;
    const group = [pointFeatures[i]];
    visited[i] = true;
    // 隣接駅を探してグループ化
    for (let j = i + 1; j < pointFeatures.length; j++) {
      if (visited[j]) continue;
      const d = turf.distance(pointFeatures[i].geometry.coordinates, pointFeatures[j].geometry.coordinates, { units: 'kilometers' });
      if (d * 1000 < MERGE_DISTANCE_METERS) {
        group.push(pointFeatures[j]);
        visited[j] = true;
      }
    }
    // 合計乗降客数と最大乗降客数の駅を求める
    let totalRidership = 0;
    let maxRidership = -Infinity;
    let anchorFeature = group[0];
    group.forEach((f) => {
      const ridership = Number(f.properties['乗降客数2023']) || 0;
      totalRidership += ridership;
      if (ridership > maxRidership) {
        maxRidership = ridership;
        anchorFeature = f;
      }
    });
    // 乗降客数からバッファー半径を決定
    let bufferRadiusKm = 0;
    for (const rule of RADIUS_RULES) {
      if (totalRidership < rule.max) {
        bufferRadiusKm = rule.radius;
        break;
      }
    }
    // バッファー生成
    const bufferGeom = turf.buffer(anchorFeature, bufferRadiusKm, { units: 'kilometers' });
    bufferGeom.properties = {
      name: anchorFeature.properties['駅名'],
      ridership: totalRidership,
      radius_km: bufferRadiusKm
    };
    bufferFeatures.push(bufferGeom);
  }
  // ソース・レイヤー追加
  map.addSource('store-areas', {
    type: 'geojson',
    data: {
      type: 'FeatureCollection',
      features: bufferFeatures
    }
  });
  map.addLayer({
    id: 'store-area-fills',
    type: 'fill',
    source: 'store-areas',
    paint: {
      'fill-color': '#f28cb1',
      'fill-opacity': 0.4
    },
    layout: {
      'visibility': 'visible'
    }
  });
  updateLegend();
}

// モードを切り替える（タブから呼び出し）
function switchMode(mode) {
  if (mode === currentMode) return;
  currentMode = mode;
  // タブの ARIA 状態とクラスの更新
  const tabPassengers = document.getElementById('tab-passengers');
  const tabMesh = document.getElementById('tab-mesh');
  if (mode === 'passengers') {
    if (tabPassengers) {
      tabPassengers.classList.add('active');
      tabPassengers.setAttribute('aria-selected', 'true');
    }
    if (tabMesh) {
      tabMesh.classList.remove('active');
      tabMesh.setAttribute('aria-selected', 'false');
    }
    // コントロール表示切り替え
    const meshControls = document.getElementById('mesh-controls');
    const passengerControls = document.getElementById('passenger-controls');
    if (meshControls) meshControls.style.display = 'none';
    if (passengerControls) passengerControls.style.display = 'block';
    // メッシュレイヤーを非表示
    if (map.getLayer('mesh-fill')) map.setLayoutProperty('mesh-fill', 'visibility', 'none');
    if (map.getLayer('mesh-outline')) map.setLayoutProperty('mesh-outline', 'visibility', 'none');
    // 駅・バッファーを更新
    updateStationDisplay();
    updateStoreAreas();
  } else if (mode === 'mesh') {
    if (tabPassengers) {
      tabPassengers.classList.remove('active');
      tabPassengers.setAttribute('aria-selected', 'false');
    }
    if (tabMesh) {
      tabMesh.classList.add('active');
      tabMesh.setAttribute('aria-selected', 'true');
    }
    // コントロール表示切り替え
    const meshControls = document.getElementById('mesh-controls');
    const passengerControls = document.getElementById('passenger-controls');
    if (meshControls) meshControls.style.display = 'block';
    if (passengerControls) passengerControls.style.display = 'none';
    // メッシュレイヤーを表示
    if (map.getLayer('mesh-fill')) map.setLayoutProperty('mesh-fill', 'visibility', 'visible');
    if (map.getLayer('mesh-outline')) map.setLayoutProperty('mesh-outline', 'visibility', 'visible');
    // 駅とバッファーを非表示
    if (map.getLayer('stations')) map.setLayoutProperty('stations', 'visibility', 'none');
    if (map.getLayer('station-labels')) map.setLayoutProperty('station-labels', 'visibility', 'none');
    if (map.getLayer('store-area-fills')) map.setLayoutProperty('store-area-fills', 'visibility', 'none');
  }
  // 凡例やスタイルを更新
  updateLegend();
  updateMapStyle();
}

// メッシュの色やフィルターを更新（cluster/density/buildings/用途別）
function updateMapStyle() {
  // メッシュレイヤーが存在する場合のみ処理
  if (!map.getLayer('mesh-fill') || !clusterConfig) return;
  // フィルター：可視クラスターのみ表示
  const filter = ['in', ['get', 'cluster'], ['literal', Array.from(visibleClusters)]];
  map.setFilter('mesh-fill', filter);
  map.setFilter('mesh-outline', filter);
  // 色設定
  let colorExpression;
  if (currentDisplayMode === 'cluster') {
    colorExpression = ['match', ['get', 'cluster']];
    clusterConfig.clusters.forEach((cluster) => {
      colorExpression.push(cluster.id, cluster.color);
    });
    colorExpression.push('#cccccc'); // デフォルト
  } else if (currentDisplayMode === 'density') {
    colorExpression = [
      'interpolate', ['linear'], ['get', '飲食店数'],
      0, '#ffffcc',
      10, '#ffeda0',
      20, '#fed976',
      50, '#feb24c',
      100, '#fd8d3c',
      200, '#fc4e2a',
      500, '#e31a1c',
      1000, '#bd0026'
    ];
  } else if (currentDisplayMode === 'buildings') {
    colorExpression = [
      'interpolate', ['linear'], ['get', '建物総数'],
      0, '#f7fbff',
      50, '#deebf7',
      100, '#c6dbef',
      200, '#9ecae1',
      400, '#6baed6',
      800, '#4292c6',
      1600, '#2171b5',
      3200, '#08519c',
      6400, '#08306b'
    ];
  } else {
    // 用途別建物割合
    const usageField = '建物_' + currentDisplayMode;
    colorExpression = [
      'case',
      ['==', ['get', '建物総数'], 0], '#f0f0f0',
      [
        'interpolate', ['linear'],
        ['/', ['get', usageField], ['get', '建物総数']],
        0, '#ffffcc',
        0.05, '#ffeda0',
        0.1, '#fed976',
        0.2, '#feb24c',
        0.3, '#fd8d3c',
        0.5, '#fc4e2a',
        0.7, '#e31a1c',
        1.0, '#bd0026'
      ]
    ];
  }
  map.setPaintProperty('mesh-fill', 'fill-color', colorExpression);
}

// クラスター選択 UI を更新
function updateClusterFilters() {
  const container = document.getElementById('cluster-filters');
  if (!container || !clusterConfig) return;
  container.innerHTML = '';
  clusterConfig.clusters.forEach((cluster) => {
    const div = document.createElement('div');
    div.className = 'cluster-filter-item';
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.id = `cluster-${cluster.id}`;
    checkbox.checked = visibleClusters.has(cluster.id);
    checkbox.addEventListener('change', (e) => {
      if (e.target.checked) {
        visibleClusters.add(cluster.id);
      } else {
        visibleClusters.delete(cluster.id);
      }
      updateMapStyle();
      updateStatistics();
    });
    const label = document.createElement('label');
    label.htmlFor = `cluster-${cluster.id}`;
    const colorBox = document.createElement('span');
    colorBox.className = 'cluster-color';
    colorBox.style.backgroundColor = cluster.color;
    const text = document.createElement('span');
    text.textContent = `${cluster.name} (${cluster.count})`;
    label.appendChild(colorBox);
    label.appendChild(text);
    div.appendChild(checkbox);
    div.appendChild(label);
    container.appendChild(div);
  });
}

// 統計情報の更新
function updateStatistics() {
  if (!clusterConfig) return;
  const totalMeshesElem = document.getElementById('total-meshes');
  const totalBuildingsElem = document.getElementById('total-buildings');
  const avgBuildingsElem = document.getElementById('avg-buildings');
  const avgRestaurantsElem = document.getElementById('avg-restaurants');
  if (totalMeshesElem) {
    totalMeshesElem.textContent = clusterConfig.total_meshes.toLocaleString();
  }
  if (totalBuildingsElem) {
    const totalBuildings = clusterConfig.clusters.reduce(
      (sum, c) => sum + c.avg_buildings * c.count, 0
    );
    totalBuildingsElem.textContent = Math.round(totalBuildings).toLocaleString();
  }
  if (avgBuildingsElem) {
    const avgBuildings = clusterConfig.clusters.reduce(
      (sum, c) => sum + c.avg_buildings, 0
    ) / clusterConfig.clusters.length;
    avgBuildingsElem.textContent = avgBuildings.toFixed(1);
  }
  if (avgRestaurantsElem) {
    const avgRestaurants = clusterConfig.clusters.reduce(
      (sum, c) => sum + c.avg_restaurants, 0
    ) / clusterConfig.clusters.length;
    avgRestaurantsElem.textContent = avgRestaurants.toFixed(1);
  }
}

// 凡例の更新
function updateLegend() {
  const container = document.getElementById('legend-content');
  if (!container) return;
  container.innerHTML = '';
  // 商圏メッシュモードの場合
  if (currentMode === 'mesh') {
    if (currentDisplayMode === 'cluster' && clusterConfig) {
      container.innerHTML = '<h4>クラスター凡例</h4>';
      clusterConfig.clusters.forEach((cluster) => {
        const item = document.createElement('div');
        item.className = 'legend-item';
        const colorBox = document.createElement('span');
        colorBox.className = 'legend-color';
        colorBox.style.backgroundColor = cluster.color;
        const label = document.createElement('span');
        label.className = 'legend-label';
        label.textContent = cluster.name;
        item.appendChild(colorBox);
        item.appendChild(label);
        container.appendChild(item);
      });
    } else if (currentDisplayMode === 'density') {
      container.innerHTML = '<h4>飲食店密度</h4>';
      const colors = [
        { color: '#ffffcc', label: '0' },
        { color: '#ffeda0', label: '10' },
        { color: '#fed976', label: '20' },
        { color: '#feb24c', label: '50' },
        { color: '#fd8d3c', label: '100' },
        { color: '#fc4e2a', label: '200' },
        { color: '#e31a1c', label: '500' },
        { color: '#bd0026', label: '1000+' }
      ];
      colors.forEach((item) => {
        const div = document.createElement('div');
        div.className = 'legend-item';
        const colorBox = document.createElement('span');
        colorBox.className = 'legend-color';
        colorBox.style.backgroundColor = item.color;
        const label = document.createElement('span');
        label.className = 'legend-label';
        label.textContent = item.label;
        div.appendChild(colorBox);
        div.appendChild(label);
        container.appendChild(div);
      });
    } else if (currentDisplayMode === 'buildings') {
      container.innerHTML = '<h4>建物総数</h4>';
      const colors = [
        { color: '#f7fbff', label: '0' },
        { color: '#deebf7', label: '50' },
        { color: '#c6dbef', label: '100' },
        { color: '#9ecae1', label: '200' },
        { color: '#6baed6', label: '400' },
        { color: '#4292c6', label: '800' },
        { color: '#2171b5', label: '1600' },
        { color: '#08519c', label: '3200' },
        { color: '#08306b', label: '6400+' }
      ];
      colors.forEach((item) => {
        const div = document.createElement('div');
        div.className = 'legend-item';
        const colorBox = document.createElement('span');
        colorBox.className = 'legend-color';
        colorBox.style.backgroundColor = item.color;
        const label = document.createElement('span');
        label.className = 'legend-label';
        label.textContent = item.label;
        div.appendChild(colorBox);
        div.appendChild(label);
        container.appendChild(div);
      });
    } else {
      // 用途別建物割合
      container.innerHTML = `<h4>${currentDisplayMode}（割合）</h4>`;
      const colors = [
        { color: '#ffffcc', label: '0%' },
        { color: '#ffeda0', label: '5%' },
        { color: '#fed976', label: '10%' },
        { color: '#feb24c', label: '20%' },
        { color: '#fd8d3c', label: '30%' },
        { color: '#fc4e2a', label: '50%' },
        { color: '#e31a1c', label: '70%' },
        { color: '#bd0026', label: '100%' }
      ];
      colors.forEach((item) => {
        const div = document.createElement('div');
        div.className = 'legend-item';
        const colorBox = document.createElement('span');
        colorBox.className = 'legend-color';
        colorBox.style.backgroundColor = item.color;
        const label = document.createElement('span');
        label.className = 'legend-label';
        label.textContent = item.label;
        div.appendChild(colorBox);
        div.appendChild(label);
        container.appendChild(div);
      });
    }
  } else {
    // 乗降客数モードの凡例
    container.innerHTML = '<h4>駅 乗降客数</h4>';
    // 駅のカラー凡例
    const stationColors = [
      { color: '#ffffcc', label: '0-4,999人' },
      { color: '#ffeda0', label: '5,000-9,999人' },
      { color: '#fed976', label: '10,000-14,999人' },
      { color: '#feb24c', label: '15,000-19,999人' },
      { color: '#fd8d3c', label: '20,000-24,999人' },
      { color: '#fc4e2a', label: '25,000-29,999人' },
      { color: '#e31a1c', label: '30,000-34,999人' },
      { color: '#bd0026', label: '35,000-39,999人' },
      { color: '#800026', label: '40,000-44,999人' },
      { color: '#67001f', label: '45,000-49,999人' },
      { color: '#4d0018', label: '50,000人以上' }
    ];
    stationColors.forEach((item) => {
      const div = document.createElement('div');
      div.className = 'legend-item';
      const colorBox = document.createElement('span');
      colorBox.className = 'legend-color';
      colorBox.style.backgroundColor = item.color;
      colorBox.style.borderRadius = '50%';
      const label = document.createElement('span');
      label.className = 'legend-label';
      label.textContent = item.label;
      div.appendChild(colorBox);
      div.appendChild(label);
      container.appendChild(div);
    });
    // バッファー半径の凡例
    const bufferLegend = document.createElement('div');
    bufferLegend.style.marginTop = '10px';
    bufferLegend.innerHTML = '<h4>出店エリア半径</h4>';
    const bufferItems = [
      { color: '#f28cb1', label: '3万未満 → 300m' },
      { color: '#f28cb1', label: '3万〜5万 → 500m' },
      { color: '#f28cb1', label: '5万以上 → 1000m' }
    ];
    bufferItems.forEach((item) => {
      const row = document.createElement('div');
      row.className = 'legend-item';
      const colorBox = document.createElement('span');
      colorBox.className = 'legend-color';
      colorBox.style.backgroundColor = item.color;
      const label = document.createElement('span');
      label.className = 'legend-label';
      label.textContent = item.label;
      row.appendChild(colorBox);
      row.appendChild(label);
      bufferLegend.appendChild(row);
    });
    container.appendChild(bufferLegend);
  }
}

// UI 全体を更新（統計・フィルター・凡例）
function updateUI() {
  updateStatistics();
  updateClusterFilters();
  updateLegend();
}