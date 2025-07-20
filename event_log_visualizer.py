import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RangeSlider
import tkinter as tk
from tkinter import filedialog, ttk
import pm4py
import warnings
from tqdm import tqdm
import matplotlib.colors as mcolors


trace_identifier = 'case:concept:name'
activity_identifier = 'concept:name'
timestamp_identifier = 'time:timestamp'
n_bins = 100


class EventLogVisualizer:
    def __init__(self, master):
        self.master = master
        self.master.title("Event Log Visualizer")

        self.log = None
        self.norm_log = None
        self.bin_data = None
        self.sorting_key = None
        self.sorting_val = None
        self.color_map = None
        self.stack_layers = None
        self.slider_update_pending = False

        self.frame = tk.Frame(master)
        self.frame.pack()

        self.load_button = tk.Button(self.frame, text="Load Log File", command=self.load_log)
        self.load_button.grid(row=0, column=0, padx=5, pady=5)

        self.save_button = tk.Button(self.frame, text="Save Screenshot", command=self.save_screenshot)
        self.save_button.grid(row=0, column=1, padx=5, pady=5)

        self.attribute_label = tk.Label(self.frame, text="Attributes")
        self.attribute_label.grid(row=0, column=2, padx=5)
        self.attribute_dropdown = ttk.Combobox(self.frame, state='readonly', width=15)
        self.attribute_dropdown.grid(row=0, column=3, padx=5)

        self.quantifier_label = tk.Label(self.frame, text="Quantifiers")
        self.quantifier_label.grid(row=0, column=4, padx=5)
        self.quantifier_dropdown = ttk.Combobox(self.frame, state='readonly', width=10)
        self.quantifier_dropdown['values'] = ['mean', 'median', 'min', 'max', 'std']
        self.quantifier_dropdown.current(0)
        self.quantifier_dropdown.grid(row=0, column=5, padx=5)

        self.ascending_label = tk.Label(self.frame, text="Ascending")
        self.ascending_label.grid(row=0, column=6, padx=5)
        self.ascending_dropdown = ttk.Combobox(self.frame, state='readonly', width=7)
        self.ascending_dropdown['values'] = ['True', 'False']
        self.ascending_dropdown.current(1)
        self.ascending_dropdown.grid(row=0, column=7, padx=5)

        self.apply_button = tk.Button(self.frame, text="Apply", command=self.apply_sorting)
        self.apply_button.grid(row=0, column=8, padx=10, pady=5)

        self.canvas_frame = tk.Frame(master)
        self.canvas_frame.pack()

        self.fig, self.ax = plt.subplots(figsize=(20, 10))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack()

        self.x_slider = None
        self.y_slider = None
        self.x_slider_ax = None
        self.y_slider_ax = None

    def load_log(self):
        file_path = filedialog.askopenfilename(filetypes=[
        ("XES files", "*.xes")])
        if not file_path:
            return

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.log = pm4py.read_xes(file_path)

        self.log['pseudo_cost'] = np.random.rand(len(self.log))

        self.norm_log = self.normalize_timestamps(self.log)
        self.bin_data = self.prepare_plot(self.norm_log)

        self.sorting_key, self.sorting_val = self.compute_sorting("FREQUENCY", None, ascending=False)
        self.color_map = self.get_fixed_color_map(self.sorting_key)

        num_attrs = self.num_attribute_list()
        self.attribute_dropdown['values'] = ["FREQUENCY", "NAME"] + num_attrs
        if num_attrs:
            self.attribute_dropdown.current(0)

        self.draw_plot()
        self.create_sliders(self.bin_data.loc[self.sorting_key].T)

    def normalize_timestamps(self, log):
        copy_log = log.copy()
        timestamps = copy_log[timestamp_identifier].astype('int64')
        grouped = copy_log.groupby(trace_identifier)[timestamp_identifier]
        min_times = grouped.transform('min').astype('int64')
        max_times = grouped.transform('max').astype('int64')
        copy_log[timestamp_identifier] = (timestamps - min_times) / ((max_times - min_times) + 1)
        return copy_log

    def prepare_plot(self, copy_log, n_bins=100):
        activity_types = set(copy_log[activity_identifier])
        arr_df = pd.DataFrame(0, index=list(activity_types), columns=range(n_bins))
        for i in tqdm(range(n_bins), desc="Counting events"):
            upper_bound = (i + 1) / n_bins
            lower_bound = i / n_bins
            bin_slice = copy_log.loc[(copy_log[timestamp_identifier] >= lower_bound) & (copy_log[timestamp_identifier] < upper_bound), activity_identifier]
            for type in activity_types:
                arr_df.loc[type, i] = (bin_slice == type).sum()
        return arr_df

    def get_fixed_color_map(self, activity_types):
        n = len(activity_types)
        colors = []
        saturations = [0.6, 0.8, 0.9]
        values = [0.7, 0.8, 0.9]
        for i in range(n):
            hue = i / n
            sat = saturations[i % len(saturations)]
            val = values[(i // len(saturations)) % len(values)]
            colors.append(mcolors.hsv_to_rgb((hue, sat, val)))
        return {activity: colors[i] for i, activity in enumerate(sorted(activity_types))}

    def draw_plot(self):
        self.ax.clear()

        data = self.bin_data.loc[self.sorting_key].T

        time_bins = data.index.values
        total_per_bin = data.sum(axis=1)
        ylim = (0, total_per_bin.max())

        # Filter out activities with all-zero values
        nonzero_columns = data.columns[(data != 0).any()]
        data = data[nonzero_columns]
        
        labels = [(data.columns[i] + ": " +str(round(self.sorting_val[i],2))) for i in range(len(data.columns))] if self.sorting_val is not None else data.columns
        colors = [self.color_map[activity] for activity in data.columns]
        self.ax.stackplot(time_bins, [data[col] for col in data.columns], labels=labels, colors=colors, alpha=0.8)
        self.ax.set_xlim((time_bins[0], time_bins[-1]))
        self.ax.set_ylim(ylim)
        self.ax.set_title("Stacked Frequency of Events")
        self.ax.set_xlabel("Time bins")
        self.ax.set_ylabel("Frequency")

        if len(data.columns) > 0:
            self.ax.legend(loc='upper right', fontsize='small', ncol=2, handlelength=1)

        self.canvas.draw()


    def create_sliders(self, data):
        if self.x_slider_ax:
            self.x_slider_ax.remove()
        if self.y_slider_ax:
            self.y_slider_ax.remove()

        self.fig.subplots_adjust(bottom=0.10)
        axcolor = 'lightgoldenrodyellow'
        self.x_slider_ax = self.fig.add_axes([0.2, 0.01, 0.65, 0.03], facecolor=axcolor)
        self.y_slider_ax = self.fig.add_axes([0.95, 0.15, 0.015, 0.7], facecolor=axcolor)

        self.x_slider = RangeSlider(self.x_slider_ax, 'Time Bin', 0, len(data.index.values),
                                    valinit=(0, len(data.index.values)), valstep=1)
        self.y_slider = RangeSlider(self.y_slider_ax, 'Frequency', 0, data.values.max(),
                                    orientation='vertical', valinit=(0, data.values.max()))

        self.slider_update_job = None  # <-- for tracking after() job

        def update(val):
            if self.slider_update_job is not None:
                self.master.after_cancel(self.slider_update_job)

            self.slider_update_job = self.master.after(300, delayed_update)

        def delayed_update():
            x_start, x_end = self.x_slider.val
            y_start, y_end = self.y_slider.val
            self.update_plot_limits(int(x_start), int(x_end), y_start, y_end)
            self.slider_update_job = None

        self.x_slider.on_changed(update)
        self.y_slider.on_changed(update)
        self.canvas.draw_idle()


    def update_plot_limits(self, x_start, x_end, y_start, y_end):
        data = self.bin_data.loc[self.sorting_key].T
        data = data.iloc[x_start:x_end]

        # Filter out columns with only zero values in the current range
        nonzero_columns = data.columns[(data != 0).any()]
        data = data[nonzero_columns]

        self.ax.clear()
        labels = [(data.columns[i] + ": " +str(round(self.sorting_val[i],2))) for i in range(len(data.columns))] if self.sorting_val is not None else data.columns

        colors = [self.color_map[activity] for activity in data.columns]
        self.ax.stackplot(data.index.values, [data[col] for col in data.columns],
                        labels=labels, colors=colors, alpha=0.8)
        self.ax.set_xlim((data.index[0], data.index[-1]))
        self.ax.set_ylim((y_start, y_end))
        self.ax.set_title("Stacked Frequency of Events")
        self.ax.set_xlabel("Time bins")
        self.ax.set_ylabel("Frequency")

        if len(data.columns) > 0:
            self.ax.legend(loc='upper right', fontsize='small', ncol=2, handlelength=1)

        self.canvas.draw()

    def save_screenshot(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"),
                                                            ("JPEG files", "*.jpg"),
                                                            ("All files", "*.*")])
        if file_path:
            self.fig.savefig(file_path, dpi=300, bbox_inches='tight')

    def num_attribute_list(self):
        if self.log is None:
            return []
        numeric_cols = []
        for col in self.log.columns:
            if col in [timestamp_identifier, activity_identifier, trace_identifier]:
                continue
            converted = pd.to_numeric(self.log[col], errors='coerce')
            if converted.notna().sum() > 0:
                numeric_cols.append(col)
        return numeric_cols

    def compute_sorting(self, attribute: str, quantity, ascending: bool = False):
        if self.log is None:
            print("No log loaded.")
            return [], None

        if attribute == "FREQUENCY":
            freq_series = self.log[activity_identifier].value_counts()
            #print(freq_series.sort_values(ascending=ascending))
            inter = freq_series.sort_values(ascending=ascending)
            return inter.index.tolist(), inter.values.tolist()

        if attribute == "NAME":
            activity_types = set(self.log[activity_identifier])
            return sorted(activity_types, reverse=not ascending), None

        if attribute not in self.log.columns:
            print(f"Attribute '{attribute}' not found in log.")
            return [], None

        if not callable(quantity):
            print("Provided quantity is not a function.")
            return [], None

        converted_col = pd.to_numeric(self.log[attribute], errors='coerce')
        grouped = self.log.copy()
        grouped[attribute] = converted_col

        try:
            values = grouped.groupby(grouped[activity_identifier])[attribute].agg(quantity)
            if isinstance(values.iloc[0], pd.Series):
                values = values.apply(lambda x: x.iloc[0] if not x.empty else np.nan)
        except Exception as e:
            print(f"Aggregation failed: {e}")
            return []

        values = values.dropna()
        #print(values.sort_values(ascending=ascending))
        inter = values.sort_values(ascending=ascending)
        return inter.index.tolist(), inter.values.tolist()

    def apply_sorting(self):
        attr = self.attribute_dropdown.get()
        quantifier = self.quantifier_dropdown.get()
        ascending = self.ascending_dropdown.get() == "True"

        func_map = {
            'mean': np.mean,
            'median': np.median,
            'min': np.min,
            'max': np.max,
            'std': np.std
        }

        quantity = None if attr in ["FREQUENCY", "NAME"] else func_map.get(quantifier, np.mean)

        new_sorting = self.compute_sorting(attr, quantity, ascending)
        if new_sorting:
            self.sorting_key = new_sorting[0]
            self.sorting_val = new_sorting[1] 
            self.draw_plot()
            self.create_sliders(self.bin_data.loc[self.sorting_key].T)


if __name__ == "__main__":
    root = tk.Tk()
    app = EventLogVisualizer(root)
    root.mainloop()
