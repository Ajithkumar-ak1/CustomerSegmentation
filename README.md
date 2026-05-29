# Customer Segmentation ML - Streamlit App

A simple and elegant Streamlit app for customer segmentation using K-means clustering.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the App
```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

## 📊 Features

- **📁 CSV Upload**: Upload your customer data
- **📊 Data Overview**: Explore data statistics
- **🔍 Elbow Method**: Find optimal number of clusters
- **🎯 Clustering**: Visualize customer segments
- **📈 Insights**: View cluster statistics
- **📥 Download**: Export segmented data as CSV

## 🎮 How to Use

1. **Upload CSV File**
   - Click "Upload CSV file" in the sidebar
   - Select your customer data file

2. **Explore Data**
   - View dataset statistics
   - Check data types and missing values

3. **Find Optimal Clusters**
   - View WCSS values for different k values
   - Look for the "elbow" point

4. **Analyze Segments**
   - Use the slider to select number of clusters
   - View scatter plot with cluster centers
   - See cluster distribution

5. **View Insights**
   - Expand each cluster to see details
   - Check average income and spending scores

6. **Export Results**
   - Download segmented data with cluster labels

## 📋 Expected CSV Format

Your CSV file should contain at least these columns:
- **Column 3**: Annual Income (k$)
- **Column 4**: Spending Score (1-100)

Example format:
```
CustomerID,Gender,Age,Annual Income,Spending Score
1,Male,19,15,39
2,Male,21,15,81
3,Female,20,16,6
...
```

## 🌐 Deploy to Streamlit Cloud

1. **Push code to GitHub**
```bash
git init
git add .
git commit -m "Customer segmentation app"
git push
```

2. **Deploy**
   - Go to https://streamlit.io/cloud
   - Click "New app"
   - Select your repository
   - Select path: `streamlit_app.py`
   - Click "Deploy"

3. **Share**
   - Get your public URL
   - Share with team/users

## 📦 Files Structure

```
.
├── streamlit_app.py      # Main Streamlit app
├── data_processor.py     # Data processing module
├── kmeans_model.py       # K-means model
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## 🔧 Customization

Edit `streamlit_app.py` to:
- Change colors and styles
- Modify cluster range (default 2-10)
- Add/remove tabs
- Change feature columns

## 📞 Support

For issues:
1. Check CSV format
2. Verify dependencies: `pip install -r requirements.txt`
3. Restart the app: `streamlit run streamlit_app.py`

## 📝 License

Open source - Use freely

---

**Happy Analyzing! 🎉**
