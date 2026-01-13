// Mapbox アクセストークン
mapboxgl.accessToken = 'pk.eyJ1IjoibmFtaTAwIiwiYSI6ImNta2FlNGFkeTFsbzkzZnNjY3kyY3h3a2QifQ._Xgc5PBf9qCnhgtcpe_mjw';

let map;
let currentClusterCount = 6;
let currentDisplayMode = 'cluster';
let meshData = null;
let clusterConfig = null;
let visibleClusters = new Set();
let stationData = null;
let showStations = false;
let stationScaleFilters = { small: true, medium: true, large: true };
let bufferEnabled = false;
let bufferDistance = 500;

document.addEventListener('DOMContentLoaded', () => {
    initMap();
    setupControls();
});

function initMap() {
    map = new mapboxgl.Map({
        container: 'map',
        style: {
            version: 8,
            glyphs: 'mapbox://fonts/mapbox/{fontstack}/{range}.pbf',  // ← これを追加
            sources: {
                'gsi-pale': {
                    type: 'raster',
                    tiles: ['https://cyberjapandata.gsi.go.jp/xyz/pale/{z}/{x}/{y}.png'],
                    tileSize: 256,
                    attribution: '<a href="https://maps.gsi.go.jp/development/ichiran.html">国土地理院</a>'
                }
            },
            layers: [{
                id: 'gsi-pale-layer',
                type: 'raster',
                source: 'gsi-pale',
                minzoom: 0,
                maxzoom: 18
            }]
        },
        center: [130.4017, 33.5904],
        zoom: 10,
        pitch: 0,
        bearing: 0
    });

    map.addControl(new mapboxgl.NavigationControl(), 'top-right');

    map.on('load', () => {
        console.log('地図読み込み完了');
        loadClusterData(currentClusterCount);
        loadStationData();
    });
}

function setupControls() {
    // クラスター数変更
    document.getElementById('cluster-count').addEventListener('change', (e) => {
        currentClusterCount = parseInt(e.target.value);
        
        if (map.getLayer('mesh-fill')) map.removeLayer('mesh-fill');
        if (map.getLayer('mesh-outline')) map.removeLayer('mesh-outline');
        if (map.getSource('mesh-data')) map.removeSource('mesh-data');
        
        loadClusterData(currentClusterCount);
    });

    // 表示モード変更
    document.getElementById('display-mode').addEventListener('change', (e) => {
        currentDisplayMode = e.target.value;
        updateMapStyle();
        updateLegend();
    });

    // 透明度変更
    document.getElementById('opacity-slider').addEventListener('input', (e) => {
        const opacity = parseInt(e.target.value) / 100;
        document.getElementById('opacity-value').textContent = `${e.target.value}%`;
        
        if (map.getLayer('mesh-fill')) {
            map.setPaintProperty('mesh-fill', 'fill-opacity', opacity);
        }
    });

    // 駅表示切替
    document.getElementById('show-stations').addEventListener('change', (e) => {
        showStations = e.target.checked;
        document.getElementById('station-filters-group').style.display = showStations ? 'block' : 'none';
        document.getElementById('station-buffer-group').style.display = showStations ? 'block' : 'none';
        updateStationDisplay();
    });

    // 駅規模フィルター
    ['small', 'medium', 'large'].forEach(scale => {
        document.getElementById(`scale-${scale}`).addEventListener('change', (e) => {
            stationScaleFilters[scale] = e.target.checked;
            updateStationDisplay();
        });
    });

    // バッファー表示切替
    document.getElementById('buffer-enable').addEventListener('change', (e) => {
        bufferEnabled = e.target.checked;
        document.getElementById('buffer-distance-control').style.display = bufferEnabled ? 'block' : 'none';
        updateStationBuffers();
    });

    // バッファー距離変更
    document.getElementById('buffer-distance').addEventListener('input', (e) => {
        bufferDistance = parseInt(e.target.value);
        document.getElementById('buffer-distance-value').textContent = `${bufferDistance}m`;
        updateStationBuffers();
    });

    // リセットボタン
    document.getElementById('reset-view').addEventListener('click', () => {
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
        }
    });
}

async function loadStationData() {
    try {
        console.log('駅データ読み込み中...');
        const response = await fetch('web_data/stations.geojson');
        stationData = await response.json();
        console.log(`駅データ読み込み完了: ${stationData.features.length} 駅`);
    } catch (error) {
        console.error('駅データ読み込みエラー:', error);
    }
}

async function loadClusterData(k) {
    try {
        console.log(`データ読み込み中 k=${k}...`);
        
        const [geojsonResponse, configResponse] = await Promise.all([
            fetch(`web_data/mesh_clusters_k${k}.geojson`),
            fetch(`web_data/cluster_config_k${k}.json`)
        ]);

        meshData = await geojsonResponse.json();
        clusterConfig = await configResponse.json();

        console.log(`データ読み込み完了: ${meshData.features.length} メッシュ`);
        
        visibleClusters = new Set(clusterConfig.clusters.map(c => c.id));
        
        if (map.isStyleLoaded()) {
            updateMap();
            updateUI();
        } else {
            map.once('idle', () => {
                updateMap();
                updateUI();
            });
        }
    } catch (error) {
        console.error('データ読み込みエラー:', error);
        alert('データの読み込みに失敗しました。');
    }
}

function updateMap() {
    if (!map.isStyleLoaded()) {
        console.log('地図準備待機中...');
        map.once('idle', updateMap);
        return;
    }

    if (!meshData) {
        console.log('データ未準備');
        return;
    }

    // 既存レイヤー削除
    if (map.getLayer('mesh-fill')) map.removeLayer('mesh-fill');
    if (map.getLayer('mesh-outline')) map.removeLayer('mesh-outline');
    if (map.getSource('mesh-data')) map.removeSource('mesh-data');

    // データソース追加
    map.addSource('mesh-data', {
        type: 'geojson',
        data: meshData
    });

    // メッシュ塗りつぶしレイヤー
    map.addLayer({
        id: 'mesh-fill',
        type: 'fill',
        source: 'mesh-data',
        paint: {
            'fill-opacity': 0.7
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
        }
    });

    updateMapStyle();

    // ポップアップイベント
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
            
            const usages = ['官公庁施設', '共同住宅', '住宅', '商業施設', '文教厚生施設', 
                           '業務施設', '商業系複合施設', '店舗等併用住宅', '店舗等併用共同住宅', '宿泊施設'];
            const totalBuildings = properties['建物総数'];
            usages.forEach(usage => {
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
}

function updateStationDisplay() {
    // 地図とデータの準備確認
    if (!map || !map.isStyleLoaded()) {
        console.log('地図準備待機中...');
        return;
    }
    
    if (!stationData) {
        console.log('駅データ未読み込み');
        return;
    }

    // 既存の駅レイヤーを削除
    if (map.getLayer('stations')) map.removeLayer('stations');
    if (map.getLayer('station-labels')) map.removeLayer('station-labels');
    if (map.getSource('stations')) map.removeSource('stations');

    if (!showStations) {
        updateLegend();
        return;
    }

    // 駅データをフィルタリング
    const filteredStations = {
        type: 'FeatureCollection',
        features: stationData.features.filter(feature => {
            const passengers = feature.properties['乗降客数2023'] || 0;
            
            if (passengers < 2000 && stationScaleFilters.small) return true;
            if (passengers >= 2000 && passengers < 10000 && stationScaleFilters.medium) return true;
            if (passengers >= 10000 && stationScaleFilters.large) return true;
            
            return false;
        }).map(feature => {
            const coords = feature.geometry.coordinates[0][0];
            return {
                type: 'Feature',
                properties: feature.properties,
                geometry: {
                    type: 'Point',
                    coordinates: coords
                }
            };
        })
    };

    // 駅データソース追加
    map.addSource('stations', {
        type: 'geojson',
        data: filteredStations
    });

    // 駅マーカーレイヤー（1000人刻みで色分け）
    map.addLayer({
        id: 'stations',
        type: 'circle',
        source: 'stations',
        paint: {
            'circle-radius': [
                'interpolate', ['linear'], ['get', '乗降客数2023'],
                0, 4,
                1000, 5,
                5000, 8,
                10000, 11,
                20000, 14,
                50000, 18
            ],
            'circle-color': [
                'step', ['get', '乗降客数2023'],
                '#ffffcc',      // 0-999人
                1000, '#ffeda0',   // 1,000-1,999人
                2000, '#fed976',   // 2,000-2,999人
                3000, '#feb24c',   // 3,000-3,999人
                4000, '#fd8d3c',   // 4,000-4,999人
                5000, '#fc4e2a',   // 5,000-5,999人
                6000, '#e31a1c',   // 6,000-6,999人
                7000, '#bd0026',   // 7,000-7,999人
                8000, '#800026',   // 8,000-8,999人
                9000, '#67001f',   // 9,000-9,999人
                10000, '#4d0018'   // 10,000人以上
            ],
            'circle-opacity': 0.8,
            'circle-stroke-width': 2,
            'circle-stroke-color': '#ffffff'
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

    // 駅クリックイベント（一度だけ登録）
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

    updateStationBuffers();
    updateLegend();
}


function updateStationBuffers() {

    // 既存のバッファーレイヤーを削除
    if (map.getLayer('station-buffers')) map.removeLayer('station-buffers');
    if (map.getSource('station-buffers')) map.removeSource('station-buffers');

    if (!bufferEnabled || !showStations || !stationData) return;

    // バッファーを生成
    const bufferedFeatures = [];
    const filteredStations = stationData.features.filter(feature => {
        const passengers = feature.properties['乗降客数2023'] || 0;
        
        if (passengers < 2000 && stationScaleFilters.small) return true;
        if (passengers >= 2000 && passengers < 10000 && stationScaleFilters.medium) return true;
        if (passengers >= 10000 && stationScaleFilters.large) return true;
        
        return false;
    });

    filteredStations.forEach(feature => {
        const coords = feature.geometry.coordinates[0][0];
        const point = turf.point(coords);
        const buffered = turf.buffer(point, bufferDistance / 1000, { units: 'kilometers' });
        bufferedFeatures.push(buffered);
    });

    if (bufferedFeatures.length === 0) return;

    const bufferCollection = {
        type: 'FeatureCollection',
        features: bufferedFeatures
    };

    map.addSource('station-buffers', {
        type: 'geojson',
        data: bufferCollection
    });

    map.addLayer({
        id: 'station-buffers',
        type: 'fill',
        source: 'station-buffers',
        paint: {
            'fill-color': '#0080ff',
            'fill-opacity': 0.2
        }
    }, 'mesh-fill');
}

function updateMapStyle() {
    if (!map.getLayer('mesh-fill') || !clusterConfig) return;

    // フィルター設定
    const filter = ['in', ['get', 'cluster'], ['literal', Array.from(visibleClusters)]];
    map.setFilter('mesh-fill', filter);
    map.setFilter('mesh-outline', filter);

    // 色設定
    let colorExpression;

    if (currentDisplayMode === 'cluster') {
        // クラスター別の色
        colorExpression = ['match', ['get', 'cluster']];
        
        clusterConfig.clusters.forEach(cluster => {
            colorExpression.push(cluster.id, cluster.color);
        });
        
        colorExpression.push('#cccccc');  // デフォルト
    } else if (currentDisplayMode === 'density') {
        // 飲食店密度ヒートマップ
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
        // 建物総数ヒートマップ
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
        // 用途別建物割合（%表示）
        const usageField = '建物_' + currentDisplayMode;
        colorExpression = [
            'case',
            ['==', ['get', '建物総数'], 0], '#f0f0f0',  // 建物なし = グレー
            [
                'interpolate', ['linear'], 
                ['/', ['get', usageField], ['get', '建物総数']],  // 割合を計算
                0, '#ffffcc',      // 0%
                0.05, '#ffeda0',   // 5%
                0.1, '#fed976',    // 10%
                0.2, '#feb24c',    // 20%
                0.3, '#fd8d3c',    // 30%
                0.5, '#fc4e2a',    // 50%
                0.7, '#e31a1c',    // 70%
                1.0, '#bd0026'     // 100%
            ]
        ];
    }

    map.setPaintProperty('mesh-fill', 'fill-color', colorExpression);
}

function updateUI() {
    if (!clusterConfig) return;

    // 統計情報更新
    updateStatistics();
    
    // クラスターフィルター更新
    updateClusterFilters();
    
    // レジェンド更新
    updateLegend();
}

function updateStatistics() {
    document.getElementById('total-meshes').textContent = 
        clusterConfig.total_meshes.toLocaleString();
    
    const totalBuildings = clusterConfig.clusters.reduce(
        (sum, c) => sum + c.avg_buildings * c.count, 0
    );
    document.getElementById('total-buildings').textContent = 
        Math.round(totalBuildings).toLocaleString();
    
    const avgBuildings = clusterConfig.clusters.reduce(
        (sum, c) => sum + c.avg_buildings, 0
    ) / clusterConfig.clusters.length;
    document.getElementById('avg-buildings').textContent = 
        avgBuildings.toFixed(1);
    
    const avgRestaurants = clusterConfig.clusters.reduce(
        (sum, c) => sum + c.avg_restaurants, 0
    ) / clusterConfig.clusters.length;
    document.getElementById('avg-restaurants').textContent = 
        avgRestaurants.toFixed(1);
}

function updateClusterFilters() {
    const container = document.getElementById('cluster-filters');
    container.innerHTML = '';

    clusterConfig.clusters.forEach(cluster => {
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

function updateLegend() {
    const container = document.getElementById('legend-content');
    container.innerHTML = ''; // まずクリア

    // --- メッシュ側の凡例 ---
    if (currentDisplayMode === 'cluster' && clusterConfig) {
        container.innerHTML = '<h4>クラスター凡例</h4>';

        clusterConfig.clusters.forEach(cluster => {
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

        colors.forEach(item => {
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

        colors.forEach(item => {
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

        colors.forEach(item => {
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

    // --- 駅側の凡例（追記） ---
    if (showStations) {
        container.innerHTML += '<h4 style="margin-top: 14px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.3);">駅（乗降客数2023）</h4>';

        const stationColors = [
            { color: '#ffffcc', label: '0-999人' },
            { color: '#ffeda0', label: '1,000-1,999人' },
            { color: '#fed976', label: '2,000-2,999人' },
            { color: '#feb24c', label: '3,000-3,999人' },
            { color: '#fd8d3c', label: '4,000-4,999人' },
            { color: '#fc4e2a', label: '5,000-5,999人' },
            { color: '#e31a1c', label: '6,000-6,999人' },
            { color: '#bd0026', label: '7,000-7,999人' },
            { color: '#800026', label: '8,000-8,999人' },
            { color: '#67001f', label: '9,000-9,999人' },
            { color: '#4d0018', label: '10,000人以上' }
        ];

        stationColors.forEach(item => {
            const div = document.createElement('div');
            div.className = 'legend-item';

            const colorBox = document.createElement('span');
            colorBox.className = 'legend-color';
            colorBox.style.backgroundColor = item.color;
            colorBox.style.borderRadius = '50%'; // 駅は丸表示

            const label = document.createElement('span');
            label.className = 'legend-label';
            label.textContent = item.label;

            div.appendChild(colorBox);
            div.appendChild(label);
            container.appendChild(div);
        });
    }
}