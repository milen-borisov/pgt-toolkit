"""
Main GUI Interface entry point for the PGT-Toolkit.
"""

# Import external modules
import sys
import os
import time
import platform
import subprocess
import shutil
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QCheckBox,
    QTabWidget,
    QGridLayout,
    QMessageBox,
    QFileDialog,
    QProgressBar,
    QSizePolicy,
)
from PySide6.QtCore import QObject, Signal, Qt, QThread, QSettings, QCoreApplication
from PySide6.QtGui import QIntValidator, QAction
from pathlib import Path

# Import internal modules
curr_file_parent = Path(__file__).resolve().parent
sys.path.insert(0, os.path.join(curr_file_parent, "pgt_toolkit"))
from pgt_toolkit.core import run_find_targets, run_generate_targets


# --------------------------------------------------------------------------
# --- SETTINGS
# --------------------------------------------------------------------------

# --- File Constants ---
SETTINGS_FILE_PATH = "settings.yaml"
SETTINGS_DEF_FILE_PATH = "settings_def.yaml"
LOG_FILE_PATH = "logs/PGT_Toolkit.log"
README_FILE_PATH = "README.md"

# --- Target Type List ---
TARGET_TYPES = [
    "HRCT_9dots_black",
    "CCT_8bit_black",
    "CCT_12bit_black",
]


# --------------------------------------------------------------------------
# --- Core Operations (Find, Generate)
# --------------------------------------------------------------------------
def worker_run_find_targets(progress_callback=None, **kwargs):
    # Call the core function with the corrected argument
    numb_found_targets = run_find_targets(
        target_type=kwargs["target_type"],
        input_images_folder=kwargs["input_images_folder"],
        res_folder_path=kwargs["res_folder_path"],
        res_csv_file_name=kwargs["res_csv_file_name"],
        threshold_modes=kwargs["threshold_modes"],
        threshold_values=kwargs["threshold_values"],
        draw_found_targets_in_images=kwargs["draw_found_targets_in_images"],
        export_found_targets_in_csv_meshroom=kwargs[
            "export_found_targets_in_csv_meshroom"
        ],
        export_found_targets_in_csv_metashape=kwargs[
            "export_found_targets_in_csv_metashape"
        ],
        progress_callback=progress_callback,
    )

    # Return message
    return f"{numb_found_targets} '{kwargs['target_type']}' targets were found in the images!"


def worker_run_generate_targets(progress_callback=None, **kwargs):
    # Convert the input arguments
    res_max_images_to_generate = kwargs["res_max_images_to_generate"]
    if isinstance(res_max_images_to_generate, str):
        if res_max_images_to_generate.lower() in ("all", "-1"):
            res_max_images_to_generate = -1

    # Generate targets images
    numb_generated_images = run_generate_targets(
        target_type=kwargs["target_type"],
        res_folder_path=kwargs["res_folder_path"],
        res_image_type=kwargs["res_image_type"],
        res_image_size=kwargs["res_image_size"],
        res_max_images_to_generate=res_max_images_to_generate,
        progress_callback=progress_callback,
    )

    # Return message
    return f"{numb_generated_images} targets were generated!"


# --------------------------------------------------------------------------
# --- Worker Class
# --------------------------------------------------------------------------
class PGTWorker(QThread):
    result = Signal(object)
    error = Signal(str)
    progress = Signal(int)

    def __init__(self, target_func, kwargs):
        super().__init__()
        self.target_func = target_func
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.target_func(progress_callback=self.progress, **self.kwargs)
            self.result.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# --------------------------------------------------------------------------
# --- PGTToolkitGUI Class
# --------------------------------------------------------------------------
class PGTToolkitGUI(QMainWindow):
    """Main application window for the PGT-Toolkit GUI."""

    # --- Initial method ---
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PGT-Toolkit GUI")
        self.resize(700, 440)

        # QSettings stores GUI window/input states, NOT file config.
        self.settings = QSettings("PGTCompany", "PGT-Toolkit")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self._setup_find_tab()
        self._setup_generate_tab()
        self._setup_progress_bar(main_layout)
        self._setup_menu_bar()

        self.worker = None
        self.progress_bar.setValue(0)

        self._load_settings()

    # --- Settings Persistence (GUI State) ---
    def closeEvent(self, event):
        self._save_settings()
        event.accept()

    def _save_settings(self):
        self.settings.setValue("App/LastTab", self.tabs.currentIndex())
        self.settings.setValue("FindTab/TargetType", self.find_type.currentText())
        self.settings.setValue("FindTab/InputFolder", self.find_input_folder.text())
        self.settings.setValue("FindTab/ResultsFolder", self.find_res_folder.text())
        self.settings.setValue("FindTab/CSVName", self.find_csv_name.text())
        self.settings.setValue("FindTab/ModeAdaptive", self.find_mode_adaptive.isChecked())
        self.settings.setValue("FindTab/ModeOtsu", self.find_mode_otsu.isChecked())
        self.settings.setValue("FindTab/ModeTriangle", self.find_mode_triangle.isChecked())
        self.settings.setValue("FindTab/ModeMax", self.find_mode_max.isChecked())
        self.settings.setValue("FindTab/FindDarkTargets", self.find_dark_checkbox.isChecked())
        self.settings.setValue("FindTab/DarkThreshold", self.find_dark_threshold.text())
        self.settings.setValue("FindTab/FindBrightTargets", self.find_bright_checkbox.isChecked())
        self.settings.setValue("FindTab/BrightThreshold", self.find_bright_threshold.text())
        self.settings.setValue("FindTab/DrawTargets", self.find_draw.isChecked())
        self.settings.setValue("FindTab/ExportMeshroom", self.find_meshroom.isChecked())
        self.settings.setValue("FindTab/ExportMetashape", self.find_metashape.isChecked())
        self.settings.setValue("GenerateTab/TargetType", self.generate_type.currentText())
        self.settings.setValue("GenerateTab/ImageType", self.generate_img_type.currentText())
        self.settings.setValue("GenerateTab/ImageSize", self.generate_img_size.text())
        self.settings.setValue("GenerateTab/MaxImages", self.generate_max_images.text())
        self.settings.setValue("GenerateTab/ResultsFolder", self.generate_res_folder.text())

    def _load_settings(self):
        def set_combobox_text(combo, key, default):
            value = self.settings.value(key, default)
            combo.setCurrentText(value)

        def get_bool_setting(key, default):
            value = self.settings.value(key, default)
            if isinstance(value, str):
                return value.lower() == "true"
            return bool(value)

        last_tab_index = self.settings.value("App/LastTab", 0, type=int)
        self.tabs.setCurrentIndex(last_tab_index)

        set_combobox_text(self.find_type, "FindTab/TargetType", "HRCT_9dots_black")
        self.find_input_folder.setText(self.settings.value("FindTab/InputFolder", ""))
        self.find_res_folder.setText(self.settings.value("FindTab/ResultsFolder", "results/gui_tests/test_find_hrct"))
        self.find_csv_name.setText(self.settings.value("FindTab/CSVName", "found_targets.csv"))
        self.find_mode_adaptive.setChecked(get_bool_setting("FindTab/ModeAdaptive", True))
        self.find_mode_otsu.setChecked(get_bool_setting("FindTab/ModeOtsu", True))
        self.find_mode_triangle.setChecked(get_bool_setting("FindTab/ModeTriangle", True))
        self.find_mode_max.setChecked(get_bool_setting("FindTab/ModeMax", False))
        self.find_dark_checkbox.setChecked(get_bool_setting("FindTab/FindDarkTargets", False))
        self.find_dark_threshold.setText(self.settings.value("FindTab/DarkThreshold", "10"))
        self.find_bright_checkbox.setChecked(get_bool_setting("FindTab/FindBrightTargets", False))
        self.find_bright_threshold.setText(self.settings.value("FindTab/BrightThreshold", "245"))
        self.find_draw.setChecked(get_bool_setting("FindTab/DrawTargets", True))
        self.find_meshroom.setChecked(get_bool_setting("FindTab/ExportMeshroom", True))
        self.find_metashape.setChecked(get_bool_setting("FindTab/ExportMetashape", True))

        set_combobox_text(
            self.generate_type, "GenerateTab/TargetType", "HRCT_9dots_black"
        )
        self.generate_img_type.setCurrentText(
            self.settings.value("GenerateTab/ImageType", "png")
        )
        self.generate_img_size.setText(
            self.settings.value("GenerateTab/ImageSize", "480")
        )
        self.generate_max_images.setText(
            self.settings.value("GenerateTab/MaxImages", "100")
        )
        self.generate_res_folder.setText(
            self.settings.value("GenerateTab/ResultsFolder", "")
        )

    # --- Configuration File Management ---
    def _restore_default_config_file(self):
        """Copies settings_def.yaml to settings.yaml, restoring the default file configuration."""

        # 1. Ask for confirmation
        reply = QMessageBox.question(
            self,
            "Confirm Restore Configuration",
            "Are you sure you want to **restore the configuration file** to its default state?\n\n"
            "This will overwrite any manual changes in settings.yaml with the contents of settings_def.yaml and requires an application restart.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                # Check if the source (default) file exists
                if not Path(SETTINGS_DEF_FILE_PATH).exists():
                    QMessageBox.critical(
                        self,
                        "Error",
                        f"Default configuration file not found: {SETTINGS_DEF_FILE_PATH}",
                    )
                    return

                # 2. Copy the default file over the current config file
                shutil.copy(SETTINGS_DEF_FILE_PATH, SETTINGS_FILE_PATH)

                # 3. Inform user and quit the application
                QMessageBox.information(
                    self,
                    "Configuration Restored",
                    "The configuration file has been restored to its default settings. \n\n"
                    "**Please restart the application now** for the changes to take full effect.",
                    QMessageBox.StandardButton.Ok,
                )

                # Quit the application gracefully
                QCoreApplication.quit()

            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Failed to restore default configuration: {e}"
                )

    def _open_config_file(self):
        """Opens the main configuration file (settings.yaml)."""
        self._open_file_in_editor(SETTINGS_FILE_PATH, is_folder=False)

    def _open_config_with_notepad(self):
        """Opens the main configuration file (settings.yaml) with Notepad (Windows only)."""
        self._open_file_in_editor_with_notepad(SETTINGS_FILE_PATH)

    # --- UI Setup and File Opening Methods ---
    def _setup_menu_bar(self):
        """Creates the menu bar."""
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Add spacer to push menus to the right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        menu_bar.setCornerWidget(spacer, Qt.Corner.TopRightCorner)

        settings_menu = menu_bar.addMenu("&Settings")

        open_config_action = QAction("&Open Configuration File (Default Editor)", self)
        open_config_action.triggered.connect(self._open_config_file)
        settings_menu.addAction(open_config_action)
        open_config_notepad_action = QAction("Open Configuration File (&Notepad)", self)
        open_config_notepad_action.triggered.connect(self._open_config_with_notepad)
        settings_menu.addAction(open_config_notepad_action)
        settings_menu.addSeparator()
        restore_config_action = QAction("Restore Default &Configuration", self)
        restore_config_action.triggered.connect(self._restore_default_config_file)
        settings_menu.addAction(restore_config_action)

        help_menu = menu_bar.addMenu("&Help")
        open_log_action = QAction("&Open Log File", self)
        open_log_action.triggered.connect(self._open_log_file)
        help_menu.addAction(open_log_action)
        open_readme_action = QAction("&Open README", self)
        open_readme_action.triggered.connect(self._open_readme_file)
        help_menu.addAction(open_readme_action)
        help_menu.addSeparator()
        about_action = QAction("&About PGT-Toolkit", self)
        about_action.triggered.connect(self._show_about_dialog)
        help_menu.addAction(about_action)

    def _open_file_in_editor(self, file_path, is_folder=False):
        """Generic method to open a file or folder using the system's default program/explorer."""
        full_path = str(Path(file_path).resolve())

        if not Path(file_path).exists():
            QMessageBox.warning(
                self, "Warning", f"Path not found: {full_path}. Ensure the path exists."
            )
            return

        system = platform.system()
        try:
            if system == "Windows":
                if is_folder:
                    subprocess.Popen(["explorer", full_path])
                else:
                    os.startfile(full_path)
            elif system == "Darwin":
                subprocess.call(["open", full_path])
            else:
                subprocess.call(["xdg-open", full_path])

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Could not open path in default system app.\n"
                f"Please navigate manually to: {full_path}\nError: {e}",
            )

    def _open_file_in_editor_with_notepad(self, file_path):
        full_path = str(Path(file_path).resolve())
        if not Path(file_path).exists():
            QMessageBox.critical(self, "Error", f"File not found at: {full_path}")
            return
        if platform.system() == "Windows":
            try:
                subprocess.Popen(["notepad.exe", full_path])
            except FileNotFoundError:
                QMessageBox.critical(
                    self,
                    "Error",
                    "Notepad.exe not found. This command only works on Windows.",
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Could not open file with Notepad:\nError: {e}"
                )
        else:
            QMessageBox.information(
                self,
                "Notepad Command",
                "The 'Open with Notepad' command is only available on Windows. "
                "The file was not opened.",
            )

    def _open_log_file(self):
        self._open_file_in_editor(LOG_FILE_PATH, is_folder=False)

    def _open_readme_file(self):
        self._open_file_in_editor(README_FILE_PATH, is_folder=False)

    def _open_results_folder(self, results_folder_line_edit: QLineEdit):
        folder_path = results_folder_line_edit.text().strip()
        if not folder_path:
            QMessageBox.warning(self, "Warning", "Results Folder Path cannot be empty.")
            return
        self._open_file_in_editor(folder_path, is_folder=True)

    def _open_csv_file(self):
        folder_path = self.find_res_folder.text().strip()
        file_name = self.find_csv_name.text().strip()

        if not folder_path or not file_name:
            QMessageBox.warning(
                self,
                "Warning",
                "Results Folder Path and CSV File Name must both be set.",
            )
            return

        full_csv_path = Path(folder_path) / file_name
        self._open_file_in_editor(str(full_csv_path), is_folder=False)

    # --- Other UI setup and Runner Methods (unchanged for brevity) ---
    def _show_about_dialog(self):
        title = "About PGT-Toolkit"
        about_text = (
            "PGT-Toolkit: Photogrammetry Target Toolkit (v1.0.0)\n\n"
            "PGT-Toolkit is developed by 'PGT Team'. "
            "It is designed to simplify the management and analysis of various Coded Targets (Markers) "
            "commonly used in photogrammetry software. It delivers a full suite of tools for generating, "
            "detecting, and decoding targets, along with comprehensive visualization and data export "
            "functions for software like 'Metashape' and 'Meshroom', and other photogrammetry "
            "applications.\n\n"
            "Contact: milen_kb@math.bas.bg"
        )
        QMessageBox.about(self, title, about_text)

    def _setup_progress_bar(self, main_layout):
        self.status_label = QLabel("Ready.")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        # self.progress_bar.setFormat("%p% - %v/100")
        self.progress_bar.setFormat("%p%")
        status_container = QHBoxLayout()
        status_container.addWidget(self.status_label, 1)
        status_container.addWidget(self.progress_bar, 2)
        main_layout.addLayout(status_container)

    def _browse_folder(self, line_edit):
        folder = QFileDialog.getExistingDirectory(self, "Select Directory")
        if folder:
            line_edit.setText(folder)

    def _setup_find_tab(self):
        """Sets up the widgets for the 'Find Targets' tab."""
        find_widget = QWidget()
        v_layout = QVBoxLayout(find_widget)
        grid = QGridLayout()

        # --- Input Settings ---
        # Row 0: Target Type Selection
        grid.addWidget(QLabel("Target Type:"), 0, 0, Qt.AlignLeft)
        self.find_type = QComboBox()
        self.find_type.setEditable(True)
        self.find_type.addItems(TARGET_TYPES)
        self.find_type.setCurrentText("HRCT_9dots_black")
        grid.addWidget(self.find_type, 0, 1, 1, 3)

        # Row 1: Input Images Folder Path and Controls
        grid.addWidget(QLabel("Images Folder:"), 1, 0, Qt.AlignLeft)
        self.find_input_folder = QLineEdit("")
        input_folder_h_layout = QHBoxLayout()
        input_folder_h_layout.addWidget(self.find_input_folder, 1)

        # Browse Button
        browse_input_btn = QPushButton("Browse...")
        browse_input_btn.clicked.connect(
            lambda: self._browse_folder(self.find_input_folder)
        )
        input_folder_h_layout.addWidget(browse_input_btn)

        # Open Folder Button
        open_input_folder_btn = QPushButton("Open Folder")
        open_input_folder_btn.clicked.connect(
            lambda: self._open_results_folder(self.find_input_folder)
        )
        input_folder_h_layout.addWidget(open_input_folder_btn)

        grid.addLayout(input_folder_h_layout, 1, 1, 1, 3)

        # Row 2: Results Folder Path and Controls
        grid.addWidget(QLabel("Results Folder:"), 2, 0, Qt.AlignLeft)
        self.find_res_folder = QLineEdit("results/latest_find")
        res_folder_h_layout = QHBoxLayout()
        res_folder_h_layout.addWidget(self.find_res_folder, 1)

        # Browse Button
        browse_res_btn = QPushButton("Browse...")
        browse_res_btn.clicked.connect(
            lambda: self._browse_folder(self.find_res_folder)
        )
        res_folder_h_layout.addWidget(browse_res_btn)

        # Open Folder Button
        open_res_btn_r2 = QPushButton("Open Folder")
        open_res_btn_r2.clicked.connect(
            lambda: self._open_results_folder(self.find_res_folder)
        )
        res_folder_h_layout.addWidget(open_res_btn_r2)

        grid.addLayout(res_folder_h_layout, 2, 1, 1, 3)

        # Row 3: CSV File Name and Quick Open Controls
        grid.addWidget(QLabel("CSV File Name:"), 3, 0, Qt.AlignLeft)
        self.find_csv_name = QLineEdit("found_targets.csv")
        grid.addWidget(self.find_csv_name, 3, 1)

        # Open CSV and Open Folder buttons in a nested layout for columns 2 and 3
        open_buttons_h_layout = QHBoxLayout()

        # Open CSV Button
        open_csv_btn = QPushButton("Open CSV")
        open_csv_btn.clicked.connect(self._open_csv_file)
        open_buttons_h_layout.addWidget(open_csv_btn)

        # Open Folder Button (The folder of the CSV file)
        open_folder_from_csv_btn = QPushButton("Open Folder")
        open_folder_from_csv_btn.clicked.connect(
            lambda: self._open_results_folder(self.find_res_folder)
        )
        open_buttons_h_layout.addWidget(open_folder_from_csv_btn)

        grid.addLayout(open_buttons_h_layout, 3, 2, 1, 2)

        v_layout.addLayout(grid)

        # --- Find Settings ---
        v_layout.addWidget(QLabel("\nImage binarization/thresholding modes:"))

        # --- 1. Threshold Mode Selection (New Checkboxes) ---
        modes_h_layout = QHBoxLayout()

        # Adaptive Mode
        self.find_mode_adaptive = QCheckBox("Adaptive (Local, Robust)")
        self.find_mode_adaptive.setChecked(True)  # Based on config comment
        modes_h_layout.addWidget(self.find_mode_adaptive)

        # OTSU Mode
        self.find_mode_otsu = QCheckBox("OTSU (Global, Fast)")
        self.find_mode_otsu.setChecked(True)  # Based on config comment
        modes_h_layout.addWidget(self.find_mode_otsu)

        # Triangle Mode
        self.find_mode_triangle = QCheckBox("Triangle (Global, Fast)")
        self.find_mode_triangle.setChecked(True)  # Based on config comment
        modes_h_layout.addWidget(self.find_mode_triangle)

        # MAX Mode
        self.find_mode_max = QCheckBox("MAX (Peak, Dark Targets)")
        self.find_mode_max.setChecked(False)  # Based on config comment
        modes_h_layout.addWidget(self.find_mode_max)

        modes_h_layout.addStretch(1)  # Push checkboxes to the left
        v_layout.addLayout(modes_h_layout)

        # --- 2. Global Intensity Thresholds (Existing Dark/Bright) ---
        # v_layout.addWidget(QLabel("Global Intensity Values (0-255):")) # Updated Label

        int_validator = QIntValidator(0, 255, self)

        # Find Very Dark Targets
        dark_target_h_layout = QHBoxLayout()
        self.find_dark_checkbox = QCheckBox("Enable dark target detection: threshold =")
        self.find_dark_checkbox.setChecked(False)
        self.find_dark_threshold = QLineEdit("10")
        self.find_dark_threshold.setValidator(int_validator)
        self.find_dark_threshold.setMaximumWidth(60)
        self.find_dark_threshold.setDisabled(True)

        self.find_dark_checkbox.toggled.connect(self.find_dark_threshold.setEnabled)

        dark_target_h_layout.addWidget(self.find_dark_checkbox)
        dark_target_h_layout.addWidget(self.find_dark_threshold)
        dark_target_h_layout.addStretch(1)
        v_layout.addLayout(dark_target_h_layout)

        # Find Very Bright Targets
        bright_target_h_layout = QHBoxLayout()
        self.find_bright_checkbox = QCheckBox(
            "Enable bright target detection: threshold ="
        )
        self.find_bright_checkbox.setChecked(False)
        self.find_bright_threshold = QLineEdit("245")
        self.find_bright_threshold.setValidator(int_validator)
        self.find_bright_threshold.setMaximumWidth(60)
        self.find_bright_threshold.setDisabled(True)

        self.find_bright_checkbox.toggled.connect(self.find_bright_threshold.setEnabled)

        bright_target_h_layout.addWidget(self.find_bright_checkbox)
        bright_target_h_layout.addWidget(self.find_bright_threshold)
        bright_target_h_layout.addStretch(1)
        v_layout.addLayout(bright_target_h_layout)

        # --- Output Settings ---
        # Output Controls (Checkboxes)
        v_layout.addWidget(QLabel("\nOutput Settings:"))

        # Output Control 1: Draw Targets
        self.find_draw = QCheckBox("Draw found targets on images")
        self.find_draw.setChecked(True)
        v_layout.addWidget(self.find_draw)

        # Output Control 2: Export Meshroom
        self.find_meshroom = QCheckBox("Export Meshroom CSV")
        self.find_meshroom.setChecked(True)
        v_layout.addWidget(self.find_meshroom)

        # Output Control 3: Export Metashape
        self.find_metashape = QCheckBox("Export Metashape CSV")
        self.find_metashape.setChecked(True)
        v_layout.addWidget(self.find_metashape)

        v_layout.addStretch(1)

        # --- Run Button ---
        # Run Button
        self.find_run_btn = QPushButton("Run Find Targets")
        self.find_run_btn.setStyleSheet("font-size: 14pt; padding: 10px;")
        self.find_run_btn.clicked.connect(self._run_find)
        v_layout.addWidget(self.find_run_btn)

        self.tabs.addTab(find_widget, "Find Targets")

    def _setup_generate_tab(self):
        """Sets up the widgets for the 'Generate Targets' tab."""
        generate_widget = QWidget()
        v_layout = QVBoxLayout(generate_widget)

        grid = QGridLayout()
        # Row 0: Target Type
        grid.addWidget(QLabel("Target Type:"), 0, 0, Qt.AlignLeft)
        self.generate_type = QComboBox()
        self.generate_type.setEditable(True)
        self.generate_type.addItems(TARGET_TYPES)
        self.generate_type.setCurrentText("HRCT_9dots_black")
        grid.addWidget(self.generate_type, 0, 1, 1, 3)  # Span 3 columns

        # Row 1: Result Image Type
        grid.addWidget(QLabel("Result Image Type:"), 1, 0, Qt.AlignLeft)
        self.generate_img_type = QComboBox()
        self.generate_img_type.addItems(["png", "jpg", "tif", "svg", "pdf"])
        self.generate_img_type.setCurrentText("png")
        grid.addWidget(self.generate_img_type, 1, 1, 1, 3)  # Span 3 columns

        # Row 2: Result Image Size
        grid.addWidget(QLabel("Result Image Size (pixels):"), 2, 0, Qt.AlignLeft)
        self.generate_img_size = QLineEdit("480")
        self.generate_img_size.setValidator(QIntValidator(1, 10000, self))
        grid.addWidget(self.generate_img_size, 2, 1, 1, 3)  # Span 3 columns

        # Row 3: Max Images to Generate
        grid.addWidget(QLabel("Max Images to Generate:"), 3, 0, Qt.AlignLeft)
        self.generate_max_images = QLineEdit("100")
        grid.addWidget(self.generate_max_images, 3, 1, 1, 3)  # Span 3 columns

        # Row 4: Results Folder Path (Text, Edit Box, Buttons - all on one row)
        grid.addWidget(QLabel("Results Folder Path:"), 4, 0, Qt.AlignLeft)

        self.generate_res_folder = QLineEdit("")
        grid.addWidget(self.generate_res_folder, 4, 1)  # Line edit in column 1

        # Buttons in a single QHBoxLayout spanning the remaining columns
        folder_buttons_h_layout = QHBoxLayout()
        folder_buttons_h_layout.setContentsMargins(0, 0, 0, 0)  # Remove margin/padding

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(
            lambda: self._browse_folder(self.generate_res_folder)
        )
        folder_buttons_h_layout.addWidget(browse_btn)

        open_res_btn = QPushButton("Open Folder")
        open_res_btn.clicked.connect(
            lambda: self._open_results_folder(self.generate_res_folder)
        )
        folder_buttons_h_layout.addWidget(open_res_btn)

        grid.addLayout(
            folder_buttons_h_layout, 4, 2, 1, 2
        )  # Layout spans columns 2 and 3

        v_layout.addLayout(grid)  # Add the entire grid

        v_layout.addStretch(1)
        self.generate_run_btn = QPushButton("Run Generate Targets")
        self.generate_run_btn.setStyleSheet("font-size: 14pt; padding: 10px;")
        self.generate_run_btn.clicked.connect(self._run_generate)
        v_layout.addWidget(self.generate_run_btn)

        self.tabs.addTab(generate_widget, "Generate Targets")

    def _set_ui_enabled(self, enabled, operation=""):
        self.find_run_btn.setDisabled(not enabled)
        self.generate_run_btn.setDisabled(not enabled)
        self.tabs.setDisabled(not enabled)
        if not enabled:
            self.setWindowTitle(f"PGT-Toolkit GUI [Running {operation}...] ")
            self.status_label.setText(f"Running {operation}...")
            self.progress_bar.setValue(0)
        else:
            self.setWindowTitle("PGT-Toolkit GUI")
            self.status_label.setText("Ready.")
            self.progress_bar.setValue(0)

    def _run_find(self):
        input_folder = self.find_input_folder.text().strip()
        if not input_folder:
            QMessageBox.critical(
                self, "Input Error", "Input Images Folder is required."
            )
            return

        # 1. Get Base Arguments
        args = {
            "target_type": self.find_type.currentText().strip(),
            "input_images_folder": input_folder,
            "res_folder_path": self.find_res_folder.text().strip(),
            "res_csv_file_name": self.find_csv_name.text().strip(),
            "draw_found_targets_in_images": self.find_draw.isChecked(),
            "export_found_targets_in_csv_meshroom": self.find_meshroom.isChecked(),
            "export_found_targets_in_csv_metashape": self.find_metashape.isChecked(),
        }

        # 2. Get Threshold Modes Arguments
        threshold_modes = []
        if self.find_mode_adaptive.isChecked():
            threshold_modes.append("adaptive")
        if self.find_mode_otsu.isChecked():
            threshold_modes.append("otsu")
        if self.find_mode_triangle.isChecked():
            threshold_modes.append("triangle")
        if self.find_mode_max.isChecked():
            threshold_modes.append("max")

        args["threshold_modes"] = threshold_modes if threshold_modes else None

        # 3. Get Dark Threshold Arguments
        args["find_very_dark_targets"] = self.find_dark_checkbox.isChecked()
        dark_threshold_val = None
        if args["find_very_dark_targets"]:
            try:
                # The QLineEdit is already validated to be an integer (0-255) by QIntValidator
                dark_threshold_val = int(self.find_dark_threshold.text())
            except ValueError:
                QMessageBox.critical(
                    self, "Input Error", "Dark threshold value is invalid."
                )
                return
        args["dark_threshold"] = dark_threshold_val

        # 4. Get Bright Threshold Arguments
        args["find_very_bright_targets"] = self.find_bright_checkbox.isChecked()
        bright_threshold_val = None
        if args["find_very_bright_targets"]:
            try:
                # The QLineEdit is already validated to be an integer (0-255) by QIntValidator
                bright_threshold_val = int(self.find_bright_threshold.text())
            except ValueError:
                QMessageBox.critical(
                    self, "Input Error", "Bright threshold value is invalid."
                )
                return
        args["bright_threshold"] = bright_threshold_val

        # 4. Assemble the 'threshold_values' argument
        threshold_values = []
        # Check if dark thresholding is enabled and a value was passed (it's validated in _run_find)
        if args["find_very_dark_targets"] and args["dark_threshold"] is not None:
            threshold_values.append(args["dark_threshold"])

        # Check if bright thresholding is enabled and a value was passed (it's validated in _run_find)
        if args["find_very_bright_targets"] and args["bright_threshold"] is not None:
            threshold_values.append(args["bright_threshold"])

        # Set to None if no thresholds were enabled/valid, which reverts to default behavior
        if not threshold_values:
            threshold_values = None
        args["threshold_values"] = threshold_values

        # 5. Start the worker
        self._start_worker(worker_run_find_targets, args, "Find Targets")

    def _run_generate(self):
        res_folder = self.generate_res_folder.text().strip()
        if not res_folder:
            QMessageBox.critical(
                self, "Input Error", "Results Folder Path is required."
            )
            return
        img_size_str = self.generate_img_size.text().strip()
        if not img_size_str or int(img_size_str) <= 0:
            QMessageBox.critical(
                self, "Input Error", "Image Size must be a positive integer."
            )
            return
        max_img_str = self.generate_max_images.text().strip()
        if max_img_str.lower() in ("all", "-1"):
            res_max_images_to_generate = -1
        elif max_img_str.isdigit() and int(max_img_str) >= 0:
            res_max_images_to_generate = int(max_img_str)
        else:
            QMessageBox.critical(
                self,
                "Input Error",
                "Max Images must be a non-negative integer, '-1', or 'all'.",
            )
            return
        args = {
            "target_type": self.generate_type.currentText().strip(),
            "res_folder_path": res_folder,
            "res_image_type": self.generate_img_type.currentText(),
            "res_image_size": int(img_size_str),
            "res_max_images_to_generate": res_max_images_to_generate,
        }
        self._start_worker(worker_run_generate_targets, args, "Generate Targets")

    def _start_worker(self, func, args, operation_name):
        self._set_ui_enabled(False, operation_name)
        self.worker = PGTWorker(func, args)
        self.worker.progress.connect(self._update_progress)
        self.worker.result.connect(
            lambda res: self._on_worker_success(res, operation_name)
        )
        self.worker.error.connect(
            lambda err: self._on_worker_error(err, operation_name)
        )
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    def _update_progress(self, value):
        self.progress_bar.setValue(value)
        self.status_label.setText(f"Processing... {value}% complete.")

    def _on_worker_success(self, result, operation_name):
        self.progress_bar.setValue(100)
        QMessageBox.information(
            self,
            f"{operation_name} Complete",
            f"{operation_name} completed successfully.\nResult: {result}",
        )

    def _on_worker_error(self, error_message, operation_name):
        self.progress_bar.setValue(0)
        QMessageBox.critical(
            self,
            f"{operation_name} Error",
            f"A fatal error occurred during {operation_name}:\n{error_message}",
        )
        print(f"\n--- FATAL RUNTIME ERROR (GUI) ---")
        print(f"Error: {error_message}")

    def _on_worker_finished(self):
        self._set_ui_enabled(True)
        self.worker = None


# --------------------------------------------------------------------------
# --- Main execution block
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Ensure files exists (settings, readme, log)
    def ensure_file_exists(path, content=""):
        if not os.path.exists(path):
            try:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w") as f:
                    f.write(content)
                print(f"Created dummy {path}")
            except Exception as e:
                print(f"Could not create dummy {path}: {e}")

    default_config_content = "# PGT-Toolkit DEFAULT Configuration File\n"
    ensure_file_exists(SETTINGS_DEF_FILE_PATH, default_config_content)

    if not os.path.exists(SETTINGS_FILE_PATH):
        print(f"settings.yaml not found. Creating it from {SETTINGS_DEF_FILE_PATH}.")
        try:
            shutil.copy(SETTINGS_DEF_FILE_PATH, SETTINGS_FILE_PATH)
        except Exception as e:
            print(f"Error copying default config: {e}")

    log_content = f"--- Log file for PGT-Toolkit (Created: {time.ctime()}) ---\n"
    ensure_file_exists(LOG_FILE_PATH, log_content)

    readme_content = "# PGT-Toolkit Documentation."
    ensure_file_exists(README_FILE_PATH, readme_content)

    # Run GUI Application
    app = QApplication(sys.argv)
    window = PGTToolkitGUI()
    window.show()
    sys.exit(app.exec())
