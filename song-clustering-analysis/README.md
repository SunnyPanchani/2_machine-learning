# 🎧 Song Clustering Web App (Flask + KMeans)

This project is a machine learning web app that clusters songs based on audio features using KMeans. It allows:

- 🔘 **Single song cluster prediction**
- 📤 **Batch CSV upload for predictions**
- 📥 **Sample CSV download**
- 📈 **Results visualization (PCA)**

---

## 🚀 Features

- Web interface built with Flask
- KMeans clustering on features like danceability, energy, valence, etc.
- Batch prediction via CSV file
- Download predicted results
- Automatically shows how to prepare input data
- Uses `StandardScaler` and optional `PCA` preprocessing

---

## 📁 Project Structure

📄 app.py # Flask app
📄 requirements.txt # Dependencies
📄 README.md # Project instructions
📄 project_summary.md # Summary & pipeline
📁 model/ # Trained model and scaler
├── kmeans_model.pkl
├── pca.pkl
└── scaler.pkl
📁 templates/ # HTML pages
├── index.html
├── upload.html
└── results.html
└── data.html
📁 static/
└── style.css
📁 results/
└── *.csv # Prediction outputs
📁 uploads/
└── sample.csv # Template for user input
📁 notebook/
└── song_clustering_analysisipynb.ipynb
📄 clusters.png # PCA visualization




---

## 🧪 Example Features Used

- `danceability` (0.0 - 1.0)
- `energy` (0.0 - 1.0)
- `valence` (0.0 - 1.0)
- `tempo` (50 - 250)
- `acousticness` (0.0 - 1.0)
- `instrumentalness` (0.0 - 1.0)
- `speechiness` (0.0 - 1.0)

---

## 🛠 Setup Instructions

pip install -r requirements.txt

python app.py
