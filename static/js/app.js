const videoInput = document.getElementById('video-input');
const dropZone = document.getElementById('drop-zone');
const progressContainer = document.getElementById('progress-container');
const progressFill = document.getElementById('progress-fill');
const statusText = document.getElementById('status-text');
const resultsSection = document.getElementById('results-section');
const originalVideo = document.getElementById('original-video');
const summaryVideo = document.getElementById('summary-video');
const captionsList = document.getElementById('captions-list');

let attentionChart = null;

// Only trigger input if not clicking directly on it (avoids double bubble)
// dropZone.onclick is removed to prevent conflict with label

videoInput.onchange = (e) => {
    const file = e.target.files[0];
    if (file) handleUpload(file);
};

// Drag & Drop
dropZone.ondragover = (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-active');
};

dropZone.ondragleave = () => dropZone.classList.remove('drag-active');

dropZone.ondrop = (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-active');
    const file = e.dataTransfer.files[0];
    if (file) handleUpload(file);
};

async function handleUpload(file) {
    const formData = new FormData();
    formData.append('video', file);

    progressContainer.classList.remove('hidden');
    progressFill.style.width = '10%';
    statusText.innerText = 'Uploading to HSTA Engine...';

    // Show local preview immediately
    originalVideo.src = URL.createObjectURL(file);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (data.task_id) {
            pollStatus(data.task_id);
        } else {
            alert("Upload failed: " + data.error);
        }
    } catch (err) {
        alert("Upload error: " + err);
    }
}

async function pollStatus(taskId) {
    const interval = setInterval(async () => {
        const response = await fetch(`/status/${taskId}`);
        const data = await response.json();

        if (data.status === 'completed') {
            clearInterval(interval);
            progressFill.style.width = '100%';
            showResults(data);
        } else if (data.status === 'error') {
            clearInterval(interval);
            statusText.innerText = 'Error: ' + data.message;
            statusText.style.color = '#f43f5e';
            progressFill.style.backgroundColor = '#f43f5e';
        } else {
            // Use real status text from backend if available
            if (data.status_text) {
                statusText.innerText = data.status_text;
            }

            // Advance progress bar smoothly
            let currentWidth = parseFloat(progressFill.style.width) || 0;
            if (currentWidth < 98) {
                // Slower increment as it gets higher
                const increment = (100 - currentWidth) / 20;
                progressFill.style.width = (currentWidth + increment) + '%';
            }
        }
    }, 2000);
}

function showResults(data) {
    progressContainer.classList.add('hidden');
    dropZone.classList.add('hidden');
    resultsSection.classList.remove('hidden');

    // Update Videos
    summaryVideo.src = data.summary_url;
    document.getElementById('download-summary').href = data.summary_url;

    // Render Chart
    renderChart(data.scores);

    // Render Captions
    captionsList.innerHTML = data.captions.map(c => `
        <div class="caption-item">
            <img src="${c.image_url}" alt="Frame">
            <div class="caption-content">
                <span class="caption-frame">Segment ${c.frame_idx}</span>
                <p class="caption-text">${c.caption}</p>
            </div>
        </div>
    `).join('');
}

function renderChart(scores) {
    const ctx = document.getElementById('attentionChart').getContext('2d');

    if (attentionChart) attentionChart.destroy();

    attentionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: scores.map((_, i) => i),
            datasets: [{
                label: 'HSTA Importance Score',
                data: scores,
                borderColor: '#2dd4bf',
                backgroundColor: 'rgba(45, 212, 191, 0.1)',
                borderWidth: 2,
                pointRadius: 0,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { display: false, min: 0, max: 1 },
                x: { display: false }
            },
            plugins: {
                legend: { display: false }
            },
            interaction: {
                intersect: false,
                mode: 'index'
            }
        }
    });
}
