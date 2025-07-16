
---

### ✅ `project_summary.md`

```markdown
# 📊 Project Summary – Song Clustering

This project uses machine learning to group similar songs using unsupervised learning (KMeans clustering).

---

## 🎯 Objective
Cluster songs based on audio features to discover patterns and groupings in music characteristics.

---

## 🔧 Technologies Used

- Python (Pandas, scikit-learn, Flask)
- HTML/CSS for UI
- KMeans Clustering
- PCA for 2D visualization
- Joblib for saving models

---

## 🧠 Workflow

1. **Data Loading**
   - Source: Spotify-style CSV with audio features

2. **Preprocessing**
   - StandardScaler applied
   - Selected audio features:
     - danceability, energy, valence, tempo
     - acousticness, instrumentalness, speechiness

3. **Clustering**
   - Optimal `k` determined using elbow method
   - KMeans clustering performed with `k=4`

4. **Dimensionality Reduction**
   - PCA (2 components) used for visualization only

5. **Model Export**
   - Saved `scaler.pkl`, `kmeans_model.pkl`, and `pca.pkl`

6. **Flask Deployment**
   - Predict clusters from user input or CSV
   - Visual and downloadable results

---

## 📁 Inputs & Outputs

### Inputs:
- 7 audio features per song
- CSV file (no `cluster` column)

### Outputs:
- Cluster label `0`, `1`, `2`, or `3`
- Result CSV with `cluster` column added
- Downloadable from UI

---

## 🔍 Improvements (Future Work)

- Use DBSCAN or HDBSCAN
- Cluster genre + audio together (via embeddings)
- Add visualization dashboard (e.g., Plotly)
- Host on Render / Hugging Face Spaces

