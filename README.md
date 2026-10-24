# Water Potability Analysis

An interactive Machine Learning web application designed to evaluate water safety and predict potability based on chemical and environmental data. 

This platform translates complex data science models into an accessible visual tool, allowing users to evaluate water samples against international safety standards.

* **Live Web App:** [https://water-potability-prediction-2-a9wip8mwzstjx6fc6dfu6x.streamlit.app/]
* **Web Developer:** Nzuzo Mkhize

---

## The Mission & Problem Solved
Raw machine learning outputs can be difficult for non-technical users or field analysts to interpret. A model might flag a water sample as "non-potable," but it won't explain the underlying chemical reasons behind that decision. 

This project bridges that gap. It provides a visual storefront where anyone can input water metrics and get an instant, machine-learning assessment alongside a transparent, metric-by-metric breakdown. It answers not just *if* a water sample is unsafe, but *why*.

---

## User Experience & Design Choices
The application is built with specific user-centric design principles to ensure data clarity and visual professionalism:

* **Intentional Typography:** Uses *Bricolage Grotesque* for bold, structured technical headings and *IBM Plex Sans* for clean, readable body text.
* **Meaningful Color Palette:** Styled in deep lagoon teals and cool mist tones. Warning colors (green, amber, red) are strictly reserved to indicate water safety status, preventing visual confusion.
* **The Verdict Panel:** A signature user interface element that displays the overall predictive outcome side-by-side with an automated check against reference guidelines, showing exactly which parameters fall outside safe margins.

---

## Core Analysis Metrics
The application evaluates water quality across three main chemical and physical groupings:

###  Acidity & Minerals
* **pH Levels:** Monitored against standard WHO baseline safe operational ranges (6.5 - 8.5).
* **Hardness:** Measures the concentration of dissolved calcium and magnesium minerals.
* **Total Dissolved Solids (TDS):** Tracks overall mineral palatability thresholds.

###  Treatment Chemicals
* **Chloramines & Sulfates:** Monitored against residual treatment allowances and taste threshold limits.
* **Trihalomethanes:** Evaluated against maximum environmental and industrial chemical safety bounds.

### 🌪 Clarity & Organics
* **Conductivity & Organic Carbon:** Tracks basic purity indicators and electrical conductivity levels.
* **Turbidity:** Measures water clarity and suspended particles against aesthetic guidelines.
