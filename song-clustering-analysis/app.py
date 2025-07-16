from flask import Flask, render_template, request, send_file, redirect, url_for
import pandas as pd
import joblib
import os
import tempfile
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Get current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained components using absolute paths
scaler = joblib.load(os.path.join(BASE_DIR, 'model', 'scaler.pkl'))
kmeans = joblib.load(os.path.join(BASE_DIR, 'model', 'kmeans_model.pkl'))

# Configure paths
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, 'results')
SAMPLE_FILE = os.path.join(BASE_DIR, 'data', 'sample.csv')  # Corrected path

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit

# Required columns for prediction
REQUIRED_COLS = [
    'danceability', 'energy', 'valence', 'tempo',
    'acousticness', 'instrumentalness', 'speechiness'
]

@app.route("/", methods=["GET", "POST"])
def index():
    cluster = None
    if request.method == "POST" and 'danceability' in request.form:
        try:
            features = [float(request.form.get(f)) for f in REQUIRED_COLS]
            X = scaler.transform([features])
            cluster = int(kmeans.predict(X)[0])
        except Exception as e:
            return render_template("index.html", error=str(e))
            
    return render_template("index.html", cluster=cluster)

@app.route("/upload", methods=["GET", "POST"])
def upload():
    message = None
    result_filename = None
    result_table = None  # <-- Add this line

    
    if request.method == "POST":
        # Handle CSV upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                message = "No file selected"
            else:
                result = process_uploaded_file(file)
                if 'error' in result:
                    message = result['error']
                    result_table = None
                else:
                    message = result['message']['text']
                    result_filename = result['message']['filename']
                    result_df = pd.read_csv(os.path.join(app.config['RESULT_FOLDER'], result_filename))
                    result_table = result_df.to_html(classes='table table-bordered', index=False)

        
        # Handle single song prediction via CSV
        elif 'csv_danceability' in request.form:
            try:
                features = {
                    'danceability': float(request.form['csv_danceability']),
                    'energy': float(request.form['csv_energy']),
                    'valence': float(request.form['csv_valence']),
                    'tempo': float(request.form['csv_tempo']),
                    'acousticness': float(request.form['csv_acousticness']),
                    'instrumentalness': float(request.form['csv_instrumentalness']),
                    'speechiness': float(request.form['csv_speechiness'])
                }
                
                # Create a temporary CSV
                df = pd.DataFrame([features])
                result_filename = "single_prediction.csv"
                result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
                df.to_csv(result_path, index=False)
                
                # Add cluster prediction
                df['cluster'] = predict_clusters(df)
                df.to_csv(result_path, index=False)
                
                message = {
                            'text': "Single song prediction complete! Download your result below.",
                            'filename': result_filename
                                                                        }
                result_table = df.to_html(classes='table table-bordered', index=False)

            except Exception as e:
                message = f"Error in prediction: {str(e)}"
    
    return render_template("upload.html", message=message, result_filename=result_filename, result_table=result_table)


def process_uploaded_file(file):
    if not allowed_file(file.filename):
        return {'error': "Invalid file type. Only CSV files are allowed."}
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        df = pd.read_csv(filepath)
        
        # Check for required columns
        missing = [col for col in REQUIRED_COLS if col not in df.columns]
        if missing:
            return {'error': f"Missing columns: {', '.join(missing)}"}
        
        # Predict clusters
        df['cluster'] = predict_clusters(df)
        
        # Save results
        result_filename = f"result_{filename}"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        df.to_csv(result_path, index=False)
        
        return {
            'message': {
                'text': "Prediction complete! Download your results below.",
                'filename': result_filename
            }
        }
    except Exception as e:
        return {'error': f"Error processing file: {str(e)}"}

def predict_clusters(df):
    """Predict clusters for a dataframe of songs"""
    X = scaler.transform(df[REQUIRED_COLS])
    return kmeans.predict(X)

def allowed_file(filename):
    return '.' in filename and filename.lower().endswith('.csv')

@app.route("/download/<filename>")
def download(filename):
    result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    return send_file(result_path, as_attachment=True, download_name=filename)

@app.route("/sample")
def download_sample():
    if not os.path.exists(SAMPLE_FILE):
        return "Sample file not found", 404
    return send_file(SAMPLE_FILE, as_attachment=True, download_name="sample_songs.csv")

@app.route("/results")
def results_list():
    """List available result files"""
    try:
        files = os.listdir(app.config['RESULT_FOLDER'])
        return render_template("results.html", files=files)
    except Exception as e:
        return f"Error accessing results: {str(e)}", 500

if __name__ == '__main__':
    # Verify paths on startup
    print(f"Sample file path: {SAMPLE_FILE}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Result folder: {RESULT_FOLDER}")
    app.run(debug=True)