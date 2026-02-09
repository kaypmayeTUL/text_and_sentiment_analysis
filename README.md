# Python for Text Analysis Workshop

**Howard-Tilton Memorial Library**  
**Tulane University**

## Overview

This workshop introduces two powerful Python techniques for analyzing text data in social science research:

1. **Topic Modeling with LDA (Latent Dirichlet Allocation)** - Discover hidden themes in large document collections using unsupervised machine learning
2. **Sentiment Analysis with VADER** - Quantify opinions and emotions in text, then compare sentiment across different groups

## Target Audience

Undergraduate and graduate students in:
- Political Science
- Sociology
- Communications
- Linguistics
- Digital Humanities

**Prerequisites:** Basic Python familiarity helpful but not required.

---

## 📂 Workshop Materials

### Presentation

- **`python_text_analysis_detailed.html`** - Interactive slide-based presentation
  - Navigate using arrow keys, spacebar, or navigation controls
  - 24 comprehensive slides covering both techniques
  - Detailed code explanations with inline comments
  - Mathematical foundations of text vectorization
  - Examples using real political and social commentary data

### Jupyter Notebooks

#### 1. Topic Modeling with LDA
- **File:** `topic_modeling_lda_advanced.ipynb`
- **Download:** [https://drive.google.com/file/d/1d1ez1GemRi3d5BVAnYmJNIUh2DE_qzw-/view?usp=sharing](https://drive.google.com/file/d/1d1ez1GemRi3d5BVAnYmJNIUh2DE_qzw-/view?usp=sharing)
- **Purpose:** Complete walkthrough of discovering topics in political speeches
- **Includes:**
  - Data loading and preprocessing
  - Text vectorization (Bag-of-Words model)
  - Stemming and n-grams
  - LDA model training and interpretation
  - Experimenting with different numbers of topics
  - Document-topic distribution analysis

#### 2. Sentiment Analysis with VADER
- **File:** `sentiment_analysis_vader.ipynb`
- **Download:** [https://drive.google.com/file/d/1J0KfWGcS5lqi2NVEfORIPc7g4JgWgz56/view?usp=sharing](https://drive.google.com/file/d/1J0KfWGcS5lqi2NVEfORIPc7g4JgWgz56/view?usp=sharing)
- **Purpose:** Complete walkthrough of sentiment analysis on social commentary
- **Includes:**
  - VADER sentiment scoring
  - Categorical sentiment classification
  - Comparing sentiment across sources
  - Multiple visualization techniques
  - Statistical summaries and export functionality

### Datasets

#### Political Speeches Dataset
- **File:** `data.csv` (or `data.xlsx`)
- **Download:** [https://tulane.box.com/s/re0651df54nk6inc4px41ev8lvox84at](https://tulane.box.com/s/re0651df54nk6inc4px41ev8lvox84at)
- **Structure:**
  - `Speech_ID` - Unique identifier
  - `Speaker` - Political leader name
  - `Text` - Full speech text
- **Size:** 50 political speeches
- **Use:** Topic modeling to discover policy themes (economy, climate, healthcare, etc.)

#### Social Commentary Dataset
- **File:** `data.csv`
- **Download:** [https://tulane.box.com/s/0d0ntev9dqzkw53n51n6ggtphdsw59f7](https://tulane.box.com/s/0d0ntev9dqzkw53n51n6ggtphdsw59f7)
- **Structure:**
  - `Comment` - Text of commentary
  - `Source` - Media source category
- **Size:** Social media comments from various sources
- **Use:** Sentiment analysis to compare opinions across media types

**⚠️ Important:** Both datasets are named `data.csv`. After downloading, rename one to avoid overwriting (e.g., `speeches_data.csv` and `sentiment_data.csv`).

---

## 🚀 Getting Started

### Option 1: View the Presentation

1. Open `python_text_analysis_detailed.html` in any web browser
2. Navigate through slides using:
   - Arrow keys (← →)
   - Spacebar (forward)
   - Navigation dropdown (jump to any slide)
   - Previous/Next buttons

### Option 2: Run the Code in Google Colab

**No installation required!**

1. Download the notebooks:
   - [Topic Modeling Notebook](https://drive.google.com/file/d/1d1ez1GemRi3d5BVAnYmJNIUh2DE_qzw-/view?usp=sharing)
   - [Sentiment Analysis Notebook](https://drive.google.com/file/d/1J0KfWGcS5lqi2NVEfORIPc7g4JgWgz56/view?usp=sharing)

2. Download the datasets:
   - [Political Speeches Data](https://tulane.box.com/s/re0651df54nk6inc4px41ev8lvox84at)
   - [Social Commentary Data](https://tulane.box.com/s/0d0ntev9dqzkw53n51n6ggtphdsw59f7)

3. Go to [Google Colab](https://colab.research.google.com/)

4. Upload the notebook:
   - Click `File` → `Upload notebook`
   - Choose the downloaded `.ipynb` file

5. Upload the corresponding dataset:
   - Click the folder icon in the left sidebar
   - Click the upload button
   - Select the CSV file

6. Run the cells in order (Shift + Enter)

**⚠️ Important:** Both data files are named `data.csv`. Make sure you upload the correct dataset for each notebook!

### Option 3: Run Locally with Jupyter

**Requirements:**
- Python 3.7 or higher
- Jupyter Notebook or JupyterLab

**Installation:**

```bash
# Create a virtual environment (recommended)
python -m venv text-analysis-env
source text-analysis-env/bin/activate  # On Windows: text-analysis-env\Scripts\activate

# Install required packages
pip install pandas scikit-learn nltk seaborn matplotlib jupyter

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('vader_lexicon')"

# Launch Jupyter
jupyter notebook
```

**Running the notebooks:**

1. Download the datasets from the links above
2. Place the CSV files in the same directory as the notebooks
3. Open the notebook in Jupyter
4. Run cells sequentially

---

## 📚 Required Python Libraries

### For Topic Modeling:
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning (CountVectorizer, LatentDirichletAllocation)
- `nltk` - Natural language toolkit (stopwords)

### For Sentiment Analysis:
- `pandas` - Data manipulation
- `nltk` - VADER sentiment analyzer
- `seaborn` - Statistical visualizations
- `matplotlib` - Plotting

### Installation Command:
```bash
pip install pandas scikit-learn nltk seaborn matplotlib
```

---

## 📖 Workshop Content Summary

### Part 1: Topic Modeling (LDA)

**Slides 4-16 | Notebook: topic_modeling_lda.ipynb**

**Key Concepts:**
- Unsupervised machine learning
- Bag-of-Words vectorization
- Document-term matrices
- Latent Dirichlet Allocation algorithm
- Topic interpretation

**What You'll Learn:**
1. How to preprocess text data (stop word removal)
2. Mathematical foundations of text vectorization
3. How LDA discovers hidden topics probabilistically
4. Interpreting topic-word distributions
5. Choosing the optimal number of topics

### Part 2: Sentiment Analysis

**Slides 17-22 | Notebook: sentiment_analysis_vader.ipynb**

**Key Concepts:**
- Lexicon-based sentiment analysis
- VADER (Valence Aware Dictionary and sEntiment Reasoner)
- Compound sentiment scores
- Categorical comparison
- Data visualization

**What You'll Learn:**
1. How VADER handles social media text (punctuation, capitalization, emoticons)
2. Calculating compound sentiment scores (-1 to +1)
3. Classifying sentiment into discrete categories
4. Comparing sentiment across groups
5. Creating effective visualizations

---

## 🎯 Learning Outcomes

By the end of this workshop, you will be able to:

✅ Load and preprocess text data for analysis  
✅ Convert text into numerical representations using Bag-of-Words  
✅ Apply unsupervised learning to discover topics in document collections  
✅ Measure sentiment in text using VADER  
✅ Compare sentiment across different categories  
✅ Create visualizations to communicate findings  
✅ Interpret and evaluate results from both techniques  

---

## 💡 Next Steps

### Experiment with Your Own Data

Both notebooks are designed to work with any text data in CSV format:

**For Topic Modeling:**
- Prepare a CSV with a text column containing your documents
- Examples: research abstracts, news articles, social media posts, survey responses

**For Sentiment Analysis:**
- Prepare a CSV with a text column and a categorical column
- Examples: customer reviews by product, tweets by political party, comments by time period

### Advanced Techniques

- **Topic coherence metrics** - Objectively evaluate topic quality
- **Time series sentiment** - Track how sentiment changes over time
- **Transformer models** - Explore state-of-the-art sentiment analysis with BERT
- **Topic visualization** - Use tools like pyLDAvis for interactive exploration

---

## 📧 Contact & Support

**Need help with your research project?**

**Kay P Maye**  
Librarian II & Scholarly Engagement Librarian  
Howard-Tilton Memorial Library  
Tulane University

**Email:** [kmaye@tulane.edu](mailto:kmaye@tulane.edu)

Available for consultations on:
- Python programming and data analysis
- Text mining and natural language processing
- Research data management
- Digital scholarship tools and methods

**Library Website:** [library.tulane.edu](https://library.tulane.edu)  
**Research Guides:** [libguides.tulane.edu](https://libguides.tulane.edu)

---

## 📄 File Inventory

```
workshop-materials/
│
├── python_text_analysis_detailed.html    # Interactive presentation (30 slides)
├── topic_modeling_lda_advanced.ipynb     # Advanced topic modeling notebook
├── sentiment_analysis_vader.ipynb        # Sentiment analysis notebook
├── README.md                             # This file
│
└── data/
    ├── data.csv (Topic Modeling)         # Political speeches dataset (50 speeches)
    ├── data.xlsx (Topic Modeling)        # Same dataset in Excel format
    └── data.csv (Sentiment Analysis)     # Social commentary dataset
```

**⚠️ Note:** Both data files are named `data.csv`. Download them separately and rename one to avoid confusion (e.g., `speeches_data.csv` and `sentiment_data.csv`).

---

## 🔗 Additional Resources

### Documentation
- [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html) - LDA API reference
- [NLTK Documentation](https://www.nltk.org/) - Natural Language Toolkit
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment) - Original VADER repository

### Tutorials
- [Text Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) - CountVectorizer guide
- [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html) - Visualization examples

### Books
- *Natural Language Processing with Python* by Bird, Klein, and Loper
- *Applied Text Analysis with Python* by Bengfort, Bilbro, and Ojeda

---

## ⚖️ License & Citation

These materials are provided for educational purposes.

**When using these materials, please cite:**

```
Python for Text Analysis Workshop
Howard-Tilton Memorial Library, Tulane University
[Year]
```

---

## 🆘 Troubleshooting

### Common Issues

**"Module not found" error:**
```bash
pip install [missing-module-name]
```

**NLTK data not found:**
```python
import nltk
nltk.download('stopwords')
nltk.download('vader_lexicon')
```

**CSV file not found:**
- Ensure the CSV file is in the same directory as your notebook
- Or provide the full path: `pd.read_csv('/path/to/file.csv')`

**Presentation slides not changing:**
- Try refreshing the browser
- Ensure JavaScript is enabled
- Use a modern browser (Chrome, Firefox, Safari, Edge)

---

**Last Updated:** February 2026  
**Version:** 1.0

For questions or feedback, please contact Howard-Tilton Memorial Library Research Services.
