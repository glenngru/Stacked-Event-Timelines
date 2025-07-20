# 📊 Event Log Visualizer

**Event Log Visualizer** is a Python-based GUI tool for visualizing process execution data using **stacked timelines**. It provides an interactive visualization of event frequencies across normalized time intervals, allowing users to explore patterns and behaviors in event logs, especially in the context of **process mining**.

This implementation is part of the seminar paper _"Stacked Timelines for Event Log Analysis"_ and demonstrates how to apply timeline visualization techniques to real-world process data.

---

## 🔍 What It Does

- Loads XES event logs using [PM4Py](https://pm4py.fit.fraunhofer.de/)
- Normalizes timestamps to align traces to a shared time interval
- Bins events into equal time intervals and counts frequencies per event type
- Displays the event types as **stacked area plots** across time
- Supports interactive filtering with:
  - Custom attribute-based sorting (frequency, name, or numeric columns)
  - Adjustable quantifiers (mean, median, min, max, std)
  - Ascending/descending order toggle
- Interactive sliders for time and frequency zoom
- Save the plot as a PNG/JPEG image

---

## 🚀 Installation

### Requirements

- Python 3.7+
- Libraries:
  - `pm4py`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `tqdm`
  - `tkinter` (usually included with Python)

### Install dependencies

```bash
pip install pm4py pandas numpy matplotlib tqdm
```

Note: On some systems, you may need to install 'tkinter' separately.

## USAGE

1. Run the script:

```bash
python event_log_visualizer.py
```

2. Use the GUI:
   - Click `Load Log File` to load a '.xes' event log.
   - Choose an `attribute` to sort event types (e.g., frequency or a numeric column).
   - Select a `quantifier` to apply to the attribute (mean, median, etc.).
   - Choose `ascendingdescending` sort order.
   - Click `Apply` to redraw the visualization.
   - Use the `slider` to zoom into specific time and frequency ranges.
   - Click `Save Screenshot` to export the current view as an image.

## INPUT FORMAT

The program expects XES-formatted event logs. Example logs can be found at the BPI Challenge (https://data.4tu.nl slash collections slash BPI_Challenge slash 5065541) datasets.

Each log should contain at least:
- **'concept:name'** (activity identifier)
- **'case:concept:name'** (trace identifier)
- **'time:timestamp'** (timestamp)

## EXAMPLE USE CASES

- Identify when different types of events typically occur in a process
- Compare the activity distribution across time in a normalized way
- Reveal temporal clusters or bottlenecks
- Rank events by custom numeric attributes (e.g., cost, duration)