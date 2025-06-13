"""
This file allows interactive display of a df with enhanced search functionality:
Requirements: pip install pandas PyQt5

To use in another file simply import it at the top of the file:
from display_df import display_df
"""

# pylint: disable=E0611,W0613,C0103,C0415
import sys

import pandas as pd
from PyQt5.QtWidgets import (
    QApplication,
    QTableView,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QLabel,
    QComboBox,
)
from PyQt5.QtCore import QAbstractTableModel, Qt, QModelIndex


class PandasModel(QAbstractTableModel):
    """Model for pandas DataFrame"""

    def __init__(self, data):
        """Initialize with data"""
        super().__init__()
        self._data = data

    def rowCount(self, parent=QModelIndex()):
        """Return number of rows"""
        return self._data.shape[0]

    def columnCount(self, parent=QModelIndex()):
        """Return number of columns"""
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        """Return data at index"""
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, section, orientation, role):
        """Return header data"""
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])
            if orientation == Qt.Vertical:
                return str(self._data.index[section])
        return None


def display_df(df_, window_title="DataFrame Viewer"):
    """Display the df with enhanced search functionality"""
    app = QApplication(sys.argv)

    # Main window
    window = QMainWindow()
    window.setWindowTitle(window_title)
    window.resize(1000, 700)

    # Central widget
    central_widget = QWidget()
    window.setCentralWidget(central_widget)
    layout = QVBoxLayout(central_widget)

    # Search area
    search_layout = QHBoxLayout()

    # Column selector
    column_label = QLabel("Column:")
    column_combo = QComboBox()
    column_combo.addItem("All Columns")
    column_combo.addItems(df_.columns.tolist())

    # Search input
    search_label = QLabel("Search:")
    search_input = QLineEdit()
    search_input.setPlaceholderText("Enter search term...")

    # Buttons
    search_button = QPushButton("Search")
    reset_button = QPushButton("Reset")

    # Status label
    status_label = QLabel("Showing all records")

    # Add widgets to search layout
    search_layout.addWidget(column_label)
    search_layout.addWidget(column_combo)
    search_layout.addWidget(search_label)
    search_layout.addWidget(search_input)
    search_layout.addWidget(search_button)
    search_layout.addWidget(reset_button)

    layout.addLayout(search_layout)
    layout.addWidget(status_label)

    # Table view
    view = QTableView()
    original_df = df_.copy()
    model = PandasModel(df_)
    view.setModel(model)
    layout.addWidget(view)

    # Enhanced search functionality
    def search():
        search_text = search_input.text().lower()
        selected_column = column_combo.currentText()

        if not search_text:
            reset()
            return

        if selected_column == "All Columns":
            # Search across all columns (existing functionality)
            filtered_df = original_df[
                original_df.astype(str).apply(
                    lambda row: row.str.lower().str.contains(search_text).any(), axis=1
                )
            ]
        else:
            # Search in specific column
            column_data = original_df[selected_column].astype(str).str.lower()
            filtered_df = original_df[column_data.str.contains(search_text)]

        view.setModel(PandasModel(filtered_df))
        status_label.setText(f"Found {len(filtered_df)} of {len(original_df)} records")

    def reset():
        view.setModel(PandasModel(original_df))
        search_input.clear()
        status_label.setText(f"Showing all {len(original_df)} records")

    search_button.clicked.connect(search)
    reset_button.clicked.connect(reset)
    search_input.returnPressed.connect(search)

    window.show()
    app.exec_()


if __name__ == "__main__":

    def create_random_df():
        """create a random df"""
        import random

        rows = 1000
        random.seed(42)
        data = {
            "name": ["Name" + str(i) for i in range(rows)],
            "category": [random.choice(["A", "B", "C", "D", "E"]) for _ in range(rows)],
            "value1": [random.randint(1, 1000) for _ in range(rows)],
            "value2": [round(random.normalvariate(100, 25), 2) for _ in range(rows)],
            "active": [random.choice([True, False]) for _ in range(rows)],
            "rating": [random.randint(1, 5) for _ in range(rows)],
            "email": ["user" + str(i) + "@example.com" for i in range(rows)],
            "description": ["Description for record " + str(i) for i in range(rows)],
        }
        return pd.DataFrame(data)

    # df = create_random_df()
    # display_df(df, "full_df")

    CSV_TO_READ = "Datasets/dataset.csv"
    display_df(pd.read_csv(CSV_TO_READ), "dataset")
