# STACWARS_-MAHATMA-HACKERS-
Hackathon submission for STAC Wars
Markdown
Welcome to our official submission for the STAC WARS hackathon. 

Given the strict 40-hour timeline, we strategically focused our efforts on delivering high-quality, fully realized solutions rather than rushing through every task. This repository contains our complete deliverables for **PS2** and **PS4**.

## Repository Structure

*   **PS2_SolarData**
    *   Contains our Python analysis script (`PS2_analysis.py`), the generated data visualizations, and our final PDF report.
    *   *Note on the ML Bonus:* We implemented a CatBoost Regressor to handle the non-linear environmental tabular data. It effectively captured the heavy influence of the 'Hour of Day' and 'Temperature' features, pushing our R² score to 0.89 with an RMSE of 103.89.
*   **PS4_Ideathon**
    *   Contains our 6-slide Pitch Deck (`.pdf`) for "Project Aegis" (our Asteroid Mining business model).
    *   We focused the engineering feasibility on near-Earth optical mining and swarm architecture, planning to sell refined propellant strictly in Low Earth Orbit to bypass the Earth gravity well bottleneck entirely.

## How to Run Our PS2 Code
If you want to verify our solar radiation prediction model locally:
1. Ensure the `solar_data.csv` file is placed in the exact same directory as the script.
2. Install the required dependencies via terminal: 
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn catboost
