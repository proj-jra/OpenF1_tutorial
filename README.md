# OpenF1 API: Interactive Strategy Dashboard Tutorial with Streamlit & Plotly

Welcome to this tutorial, where you'll learn to build an interactive Formula 1 strategy dashboard using the OpenF1 API, Streamlit, and Plotly. This hands-on project is ideal for those interested in data visualization, sports analytics, and modern Python web tools.

## 📊 Overview

This dashboard enables users to:
- Select a race by year and country
- View lap times per driver with pit stop flags
- Analyze tire strategy over the race distance
- Compare pit stop durations

### Technologies used:
- **OpenF1 API** for motorsport telemetry data  
- **Pandas** for data handling  
- **Plotly** for interactive charts  
- **Streamlit** for web UI  

---

## 📁 Project Structure

```
openf1-dashboard-tutorial/
├── app/
│   ├── data_loader.py        # Handles OpenF1 API requests
│   ├── data_processor.py     # Cleans and enriches OpenF1 data
│   └── visualizer.py         # Builds interactive visualizations from OpenF1 data
├── main.py                   # Streamlit app logic
├── requirements.txt          # Python dependencies
└── .env                      # Contains BASE_API_URL for OpenF1
```

---

## 📸 Screenshot

![lap_time_chart](./assets/Screenshot1.png)
![tyre_strategy_chart](./assets/Screenshot2.png)
![pit_stop_chart](./assets/Screenshot3.png)

## 🛠️ Setup & Requirements

### 1. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install streamlit pandas plotly python-dotenv
```

### 3. Create a `.env` file
```
BASE_API_URL=https://api.openf1.org/v1/
```

---

## 🚀 Launch the App

```bash
streamlit run main.py
```

This will open the dashboard in your default browser.

---

### 3. 📂 main.py Highlights

#### Features:

Dynamic year/country selection

Granular session data filtering

Lap time, tire strategy, and pit stop visualizations
(8/27: Added Lap Sector Times)

#### Key Flow:

Fetch meeting/session info via data_loader.py

Process raw data in data_processor.py

Visualize with plot_lap_times(), plot_tire_strategy(), plot_pit_stop() from visualizer.py

##### Inline comments in the code guide you through OpenF1 endpoint usage:

meetings returns all races in a season

sessions returns FP1, Quali, Race for a given race (meeting_key)

laps, pit, stints, and drivers use session_key to pull telemetry data


### 4 🔍 File Descriptions
```bash
data_loader.py
```
Handles OpenF1 API calls using requests, with optional pagination logic. Each fetch function:

Specifies the OpenF1 endpoint (e.g., "laps", "drivers")

Applies query filters (like session_key or meeting_key)

Uses @st.cache_data to reduce network calls
```bash
data_processor.py
```
Cleans and formats raw OpenF1 data:

Filters invalid lap or pit rows

Calculates stint lap ranges from lap_start to lap_end

Builds a driver_color_map from drivers.team_colour to use in plots
```bash
visualizer.py
```
Creates interactive charts:

[new] plot_sector_times(): bar chart of sectors (1-3) colored by fastest(magenta), average (green), slowest (red) with driver and lap selection.
plot_lap_times(): line chart of lap_duration colored by driver

plot_tire_strategy(): horizontal bar chart from stints

plot_pit_stop(): vertical bar chart for pit_duration


All charts format hover templates and colors using OpenF1 data fields.

---

## 💡 Extend This Project

Ideas to build on:
- Add tire degradation trends
- Compare qualifying vs. race pace
- Highlight fastest laps and race events
- <u> Integrate sector time analytics </u>

---

## 🎉 Conclusion

You've now built an interactive F1 dashboard using real-world telemetry data from the OpenF1 API. This is a great example of combining API usage, data processing, and visual storytelling in Python.

Fork it, share it, or showcase it in your portfolio!
