{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "d1a4621d",
   "metadata": {},
   "source": [
    "# Predicting Wildfire Potential Destructive Power in California"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5fe74246",
   "metadata": {},
   "source": [
    "Author: Dustin Littlefield\\\n",
    "Project Type: Data Science & GIS Portfolio\\\n",
    "Technologies: Python, Pandas, Scikit-learn, XGBoost, GeoPandas, Matplotlib\\\n",
    "Skills: `Data cleaning` `feature engineering` `supervised machine learning` `model evaluation` `class imbalance handling` \\\n",
    "`spatial visualization` `exploratory data analysis` `reproducible workflow design` `results communication`\\\n",
    "Status: In Progress\\\n",
    "Last Updated: July 2025"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "83c914f3",
   "metadata": {},
   "source": [
    "## Overview\n",
    "This project is a work in progress that explores the relationship between environmental and weather-related factors and wildfire severity in California. The goal is to predict a custom severity index `Wildfire Potential Destructive Power` — which incorporates structures damaged, structures destroyed, and fatalities.\n",
    "\n",
    "**Disclaimer:** I am not a climate scientist or wildfire expert. This project is intended to demonstrate data science, geospatial, and machine learning skills. It is not designed for operational use or policy decisions."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e2715b42",
   "metadata": {},
   "source": [
    "Example Output:"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0084941b",
   "metadata": {},
   "source": [
    "![Southern California Wildfire Model Predictions](plots/Palisades_predictions.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b0d6e0c7",
   "metadata": {},
   "source": [
    "## Objectives\n",
    "- Predict wildfire severity based on environmental and weather data.\n",
    "- Test classification models using resampling techniques to handle class imbalance.\n",
    "- Create geospatial visualizations to illustrate regional risk patterns.\n",
    "- Explore second-degree feature interactions and correlation to improve model features.\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "34f5d902",
   "metadata": {},
   "source": [
    "## Project Structure"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5f61c574",
   "metadata": {},
   "source": [
    "![File Structure](plots/file_structure.png)\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5517131e",
   "metadata": {},
   "source": [
    "## Data Sources\n",
    "\n",
    "**Fire Incident Data** – includes structure and fatality impact measures. \\\n",
    "**California CIMIS Weather Data** – daily temperature, wind speed, precipitation, humidity. \\\n",
    "**California Demographic Data** - population density and mean income by county, proxy for firefighting resources \\\n",
    "**GIS Layers** – Shapefiles for spatial visualization.\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ba0ce393",
   "metadata": {},
   "source": [
    "## Data Processing\n",
    "*Located in: notebooks/data_processing.ipynb*\n",
    "- Merged fire records with weather station summaries by location and time.\n",
    "- Created rolling averages for environmental variables.\n",
    "- Engineered interaction features (e.g., Dryness, ETo_x_Vapor_Pressure).\n",
    "- Imputed missing values for stations and derived features."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2ca06c5d",
   "metadata": {},
   "source": [
    "#### Key Features Used:\n",
    "1. **Environmental / Weather Variables**\n",
    "- `Avg Air Temp (F) 7 Day Avg` – Average air temperature over the past 7 days (°F); represents heat conditions.\n",
    "- `Avg Vap Pres (mBars)` – Average vapor pressure; indicates atmospheric moisture.\n",
    "- `Avg Rel Hum (%) 7 Day Avg` – Average relative humidity over 7 days; affects fire ignition and spread.\n",
    "- `Avg Wind Speed (mph) 7 Day Avg` – Average wind speed; higher speeds can drive fire spread.\n",
    "- `Precip (in) 7 Day Avg` – Total precipitation in the past 7 days; influences fuel moisture.\n",
    "- `ETo (in)` – Reference evapotranspiration; approximates water loss from soil and plants.\n",
    "\n",
    "2. **Derived / Interaction Features**\n",
    "- `ETo_x_Vapor_Pressure` – Interaction between evapotranspiration and vapor pressure; models combined dryness effects.\n",
    "- `ETo_x_Temp` – Interaction between evapotranspiration and air temperature; highlights hot, dry conditions.\n",
    "- `Vapor_Pressure_x_Temp` – Interaction capturing the combined effect of heat and moisture.\n",
    "- `Vapor_Pressure_x_Wind_Speed` – Interaction between wind and atmospheric moisture; affects drying conditions.\n",
    "\n",
    "3. **Composite Index**\n",
    "- `Dryness` – Custom dryness proxy combining weather variables; designed to approximate vegetation or fuel dryness.\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7f56f06a",
   "metadata": {},
   "source": [
    "## Class Balancing"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "61502ad4",
   "metadata": {},
   "source": [
    "*Located in: notebooks/class_balancing.ipynb*\n",
    "\n",
    "**Target:** `WPDP` Wildlife Potential Destructive Power - categorized into Low, Moderate, High\n",
    "\n",
    "**Issues:** Moderate and High Damage wildfire events classes are underrepresented.\n",
    "\n",
    "Balancing Techniques Used:\n",
    "- In method class balancing\n",
    "- Manual undersampling of the dominant \"Low\" class.\n",
    "- SMOTE for oversampling\n",
    "\n",
    "Comparison of model performance across balancing strategies.\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1a702d6d",
   "metadata": {},
   "source": [
    "## Modeling\n",
    "*Located in: notebooks/modeling.ipynb*\n",
    "\n",
    "**Models tested:**\n",
    "`Random Forest`\n",
    "`K-Nearest Neighbors`\n",
    "`XGBoost`\n",
    "\n",
    "**Metrics evaluated:**\n",
    "`F1-score (macro-averaged)`\n",
    "`Confusion matrices`\n",
    "`Cross-validation`\n",
    "\n",
    "Feature importance extracted for tree-based models.\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c22d125e",
   "metadata": {},
   "source": [
    "## GIS & Visualization\n",
    "*Located in: notebooks/evaluation_and_visualization.ipynb*\n",
    "\n",
    "- Maps using GeoPandas, Matplotlib, and Seaborn.\n",
    "- IDW interpolation for environmental variables.\n",
    "- Severity overlay by county or fire footprint.\n",
    "\n",
    "Example Output:"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8b893672",
   "metadata": {},
   "source": [
    "<img src=\"plots/Interpolated.png\" alt=\"Southern California Wildfire Model Predictions\" width=\"500\" style=\"display: block; margin-left: 0;\" />\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9c4f9387",
   "metadata": {},
   "source": [
    "## Key Results\n",
    "\n",
    "**Key Findings:** \\\n",
    "Models struggled with distinguishing Moderate and High severity classes.\\\n",
    "KNN achieved the highest F1-score overall.\\\n",
    "Class balancing significantly improved recall for minority classes."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6cd216e3",
   "metadata": {},
   "source": [
    "<img src=\"plots/sampling_metrics.png\" alt=\"Southern California Wildfire Model Predictions\" width=\"400\" style=\"display: block; margin-left: 0;\" />\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9525bae9",
   "metadata": {},
   "source": [
    "## Challenges\n",
    "\n",
    "**Missing Environmental Data** – Gaps in weather stations required imputation.\\\n",
    "**Weak Correlation** – Environmental features don’t fully explain severity outcomes.\\\n",
    "**Class Imbalance** – Severe fires are rare; balancing was essential.\\\n",
    "**Derived Variable Uncertainty** – Proxies like Dryness need validation.\\\n",
    "**Spatial Generalization** – Models may not perform well across regions.\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9ef2ec61",
   "metadata": {},
   "source": [
    "## Next Steps / Potential Improvements\n",
    "- Add land cover, topography, and WUI datasets.\n",
    "- Integrate population density and elevation.\n",
    "- Incorporate days since ignition as a time-based feature.\n",
    "- Try temporal or ensemble models.\n",
    "- Consult domain experts to validate assumptions and feature selection.\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "054f676d",
   "metadata": {},
   "source": [
    "## Installation\n",
    "To run the project locally:\\\n",
    "git clone https://github.com/dustinlit/wildfire-severity.git \\\n",
    "cd wildfire-severity\\\n",
    "pip install -r requirements.txt\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3109cb73",
   "metadata": {},
   "source": [
    "## License\n",
    "This project is released under the MIT License.\n",
    "See LICENSE for details."
   ]
  }
 ],
 "metadata": {
  "hide_input": false,
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
