// Mapbox アクセストークン（必ず自分のトークンに置き換えてください）
mapboxgl.accessToken = 'pk.eyJ1IjoibmFtaTAwIiwiYSI6ImNtazZnc2RnbzBvdnEzZXI1ZHhlN2Yyc3gifQ.ikUmsp1TSWrZNMteouL9aQ';

let map;
let currentClusterCount = 6;
let currentDisplayMode = 'cluster';
let meshData = null;
let clusterConfig = null;
let visibleClusters = new Set();

document.addEventListener('DOMContentLoaded', () => {
    initMap();
    setupControls();
});

function initMap() {
    map = new mapboxgl.Map({
        container: 'map',
        style: 'mapbox://styles/mapbox/light-v11',
        center: [130.4017, 33.5904],
        zoom: 10,
        pitch: 0,
        bearing: 0
    });

    map.addControl(new mapboxgl.NavigationControl(), 'top-right');

    map.on('load', () => {
        console.log('地図読み込み完了');
        loadClusterData(currentClusterCount);
    });
}

function setupControls() {
    // クラスター数変更
    document.getElementById('cluster-count').addEventListener('change', (e) => {
        currentClusterCount = parseInt(e.target.value);
        
        // 既存レイヤーを削除
        if (map.getLayer('mesh-fill')) map.removeLayer('mesh-fill');
        if (map.getLayer('mesh-outline')) map.removeLayer('mesh-outline');
        if (map.getSource('mesh-data')) map.removeSource('mesh-data');
        
        loadClusterData(currentClusterCount);
    });

    // 表示モード変更
    document.getElementById('display-mode').addEventListener('change', (e) => {
        currentDisplayMode = e.target.value;
        updateMapStyle();
    });

    // 透明度変更
    document.getElementById('opacity-slider').addEventListener('input', (e) => {
        const opacity = parseInt(e.target.value) / 100;
        document.getElementById('opacity-value').textContent = `${e.target.value}%`;
        
        if (map.getLayer('mesh-fill')) {
            map.setPaintProperty('mesh-fill', 'fill-opacity', opacity);
        }
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
        
        // 地図が準備できるまで待機
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
    // 地図が準備できているか確認
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

    // ポップアップイベント（一度だけ登録）
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
    }
}
