# Public Risk Perception of Microplastics in Groundwater  
*Survey Data Analysis with Python*  

## 📖 Project Overview  
This repository contains Python scripts and notebooks for analyzing survey data on public risk perception of **microplastics in groundwater**. The project aims to:  
- Explore how people perceive risks associated with microplastics.  
- Identify demographic patterns in survey responses.  
- Visualize results through clear and accessible figures.  
- Provide reproducible analysis for researchers and policymakers.  

## 📊 Data  
- **Source**: The survey was developed as part of my PhD research and counted on other members of the Plastic Underground Project to validate and disseminate. The survey was available online for 50 days for respondents living in the UK, Switzerland, Italy, Spain, France, Germany and Cyprus, and an additional 30 days in Italy. The survey was conducted in 2024 and had 1385 participants. The analysis presented here concern the Italian respondents of the survey.  
- **Format**: CSV/Excel.  
- **Variables**: demographics, water quality, concern levels, trust in institutions, awareness of MPs sources and impacts, etc.  

⚠️ *Note: Raw data may not be included for privacy reasons. A sample dataset or anonymized data can be provided instead.*  

## 🛠️ Requirements  
Install dependencies using:  
```bash
pip install -r requirements.txt
```  

Key Python packages:  
- `Pandas` / `NumPy` / `os` / `intertools` / `math` – data manipulation  and numerical analysis
- `SciPly` / `scikit-posthocs` / `Factor Analyzer` / `Pingouin` / `Statsmodel` – statistics and psycometrics 
- `matplotlib` / `seaborn` – visualization 
- `kmodes` / `sklearn` / `Mlxtend` - machine learning

## 📂 Repository Structure  
```
├── data/                # Raw and/or processed data (if available) , metadata 
├── scripts/             # Python scripts for cleaning & analysis  
├── results/             # Figures, tables, and summary outputs  
├── requirements.txt     # Python dependencies  
└── README.md            # Project documentation  
```  

## 🚀 Usage  
1. Clone this repository:  
   ```bash
   git clone https://github.com/yourusername/Italian_analysis.git
   cd Italian_analysis
   ```  
2. (Optional) Create and activate a virtual environment.  
3. Run analysis:  
   ```bash
   jupyter notebook notebooks/analysis.ipynb
   ```  

## 📈 Outputs  
- Distribution of risk perception scores  
- Demographic differences in concern about microplastics  
- Heatmaps and bar plots summarizing awareness levels  

## ✅ Reproducibility  
All analysis is reproducible. Ensure the dataset is available in the `data/` directory before running scripts or notebooks.  

## 📌 Future Work  
- Advanced statistical modeling (e.g., regression, factor analysis).  
- Cross-country comparisons if multi-regional data becomes available.  
- Integration with groundwater contamination measurements.  

## 🤝 Contributing  
Contributions are welcome! Please open an issue or pull request for suggestions, improvements, or additional analysis.  

## 📜 License  
[MIT License](LICENSE)  

## 📧 Contact  
For questions, reach out to:  
- **Barbara Zambelli** – [b.zambelliazevedo@studenti.unipi.it]  
- Project maintained at: [https://github.com/Taiobarbara/Italian_analysis]  
