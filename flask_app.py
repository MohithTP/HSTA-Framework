import os
import uuid
import threading
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from utils.summarizer import VideoSummarizer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 # 100MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['STATIC_FOLDER'], 'temp_frames'), exist_ok=True)

summarizer = VideoSummarizer()

# In-memory store for task status
tasks = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    task_id = str(uuid.uuid4())
    filename = f"{task_id}_{secure_filename(file.filename)}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    tasks[task_id] = {"status": "processing", "progress": 10}
    
    # Start processing in background
    def run_summarization(tid, path):
        def update_status(text):
            tasks[tid]["status_text"] = text

        try:
            summary_path, scores, top_indices = summarizer.summarize(path, status_callback=update_status)
            captions = summarizer.generate_captions(path, top_indices, status_callback=update_status)
            
            tasks[tid].update({
                "status": "completed",
                "summary_url": f"/uploads/{os.path.basename(summary_path)}",
                "scores": scores.tolist(),
                "captions": captions
            })
        except Exception as e:
            tasks[tid] = {"status": "error", "message": str(e)}

    threading.Thread(target=run_summarization, args=(task_id, filepath)).start()
    
    return jsonify({"task_id": task_id}), 202

@app.route('/status/<task_id>')
def get_status(task_id):
    return jsonify(tasks.get(task_id, {"status": "not_found"}))

@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
