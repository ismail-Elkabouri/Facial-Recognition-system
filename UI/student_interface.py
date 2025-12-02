import sys
from datetime import datetime

import qtawesome as qta
from PyQt5.QtCore import QSize, Qt, pyqtSignal, QDate
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDateEdit,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStackedLayout,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget, QHeaderView, QFileDialog,
)

from database.db_helper import DatabaseHelper


class StudentDashboard(QMainWindow):
    """Student dashboard for viewing attendance, requesting leave, and managing settings."""

    logout_requested = pyqtSignal()

    def __init__(self, auth_manager, student_data):
        """Initialize the StudentDashboard with authentication and student data."""
        super().__init__()
        self.auth_manager = auth_manager
        self.student_id = student_data['id']
        self.student_name = student_data['name']
        self.db = DatabaseHelper()

        try:
            self.conn = self.db.get_connection()
            if self.conn is None:
                raise Exception("Failed to connect to the database")
        except Exception as e:
            QMessageBox.critical(self, "Database Error", f"Could not connect to the database: {e}")
            sys.exit()

        print(f"StudentDashboard initialized for {self.student_name} (ID: {self.student_id})")

        self.setup_ui()
        self.load_data()

    def setup_ui(self):
        """Set up the main UI components of the dashboard."""
        self.setWindowTitle(f"Student Dashboard - {self.student_name}")
        self.setMinimumSize(1800, 1200)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        top_bar = QWidget()
        top_bar.setStyleSheet("background-color: #0d6efd; min-height: 100px;")
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(20, 0, 20, 0)

        header_label = QLabel(f"Welcome, {self.student_name}")
        header_label.setStyleSheet("color: white; font-size: 20px; font-weight: bold;")
        top_bar_layout.addWidget(header_label)
        top_bar_layout.addStretch()

        logout_button = QPushButton(QIcon(qta.icon("fa5s.sign-out-alt", color="white")), "Logout")
        logout_button.setStyleSheet("""
            QPushButton {
                color: white;
                background-color: transparent;
                border: none;
                padding: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.1);
            }
        """)
        logout_button.clicked.connect(self.logout_requested.emit)
        top_bar_layout.addWidget(logout_button)
        main_layout.addWidget(top_bar)

        content_widget = QWidget()
        content_widget.setStyleSheet("background-color: #f0f2f5;")
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        main_layout.addWidget(content_widget)

        self.setup_sidebar(content_layout)
        self.setup_main_content_area(content_layout)
        self.show_section("home")

    def setup_sidebar(self, parent_layout):
        """Set up the sidebar with navigation buttons."""
        self.sidebar = QWidget()
        self.sidebar.setFixedWidth(250)
        self.sidebar.setStyleSheet("background-color: #E6E6E6;")
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        sidebar_buttons = [
            ("Home", "fa5s.home", "home"),
            ("Attendance History", "fa5s.calendar-alt", "history"),
            ("Ask for Leave", "fa5s.calendar-times", "ask-for-leave"),
            ("Settings", "fa5s.cogs", "settings"),
        ]

        self.sidebar_button_group = []
        for label_text, icon_name, section_id in sidebar_buttons:
            button = QPushButton(QIcon(qta.icon(icon_name, color="#495057")), label_text)
            button.setIconSize(QSize(24, 24))
            button.setProperty("section_id", section_id)
            button.setStyleSheet(self.sidebar_button_style(False))
            button.setCheckable(True)
            button.clicked.connect(lambda checked, btn=button: self.show_section(btn.property("section_id")))
            sidebar_layout.addWidget(button)
            self.sidebar_button_group.append(button)

        sidebar_layout.addStretch()
        parent_layout.addWidget(self.sidebar)

    def setup_main_content_area(self, parent_layout):
        """Set up the main content area with a stacked layout."""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        parent_layout.addWidget(scroll_area)

        self.main_content = QWidget()
        self.main_content.setStyleSheet("background-color: #f7f7f7;")
        scroll_area.setWidget(self.main_content)

        self.stack_layout = QStackedLayout(self.main_content)

        self.setup_home_section()
        self.setup_history_section()
        self.setup_ask_for_leave_section()
        self.setup_settings_section()

    def setup_home_section(self):
        """Set up the home section with statistics and recent attendance."""
        home_widget = QWidget()
        layout = QVBoxLayout(home_widget)

        overview_layout = QHBoxLayout()
        self.total_classes_card, self.total_classes_value = self.create_stat_card("Total Classes", "fa5s.chalkboard")
        self.attended_classes_card, self.attended_classes_value = self.create_stat_card("Attended Classes", "fa5s.chart-line")
        self.missed_classes_card, self.missed_classes_value = self.create_stat_card("Missed Classes", "fa5s.calendar-times")
        overview_layout.addWidget(self.total_classes_card)
        overview_layout.addWidget(self.attended_classes_card)
        overview_layout.addWidget(self.missed_classes_card)
        layout.addLayout(overview_layout)

        class_filter_layout = QHBoxLayout()
        class_filter_layout.addWidget(QLabel("Select Class:"))
        self.home_class_combo = QComboBox()
        self.home_class_combo.currentIndexChanged.connect(self.load_recent_attendance)
        self.home_class_combo.setStyleSheet("""
            QComboBox {
                padding: 8px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                font-size: 14px;
                background-color: white;
                min-width: 200px;
                color: #212529;
            }
            QComboBox:hover {
                border: 1px solid #0d6efd;
            }
            QComboBox::drop-down {
                width: 20px;
                border-left: 1px solid #ced4da;
            }
            QComboBox QAbstractItemView {
                background-color: white;
                border: 1px solid #ced4da;
                selection-background-color: #0d6efd;
                selection-color: white;
            }
            QComboBox QAbstractItemView::item {
                min-height: 24px;
                padding: 4px;
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: #e9ecef;
                color: #212529;
            }
            QComboBox QAbstractItemView::item:selected {
                background-color: #0d6efd;
                color: white;
            }
        """)
        class_filter_layout.addWidget(self.home_class_combo)
        class_filter_layout.addStretch()
        layout.addLayout(class_filter_layout)

        layout.addWidget(QLabel(f"Recent Attendance for {self.student_name}"))
        self.recent_attendance_table = QTableWidget()
        self.recent_attendance_table.setColumnCount(3)
        self.recent_attendance_table.setHorizontalHeaderLabels(["Class", "Timestamp", "Status"])
        self.recent_attendance_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.recent_attendance_table.setShowGrid(True)
        self.recent_attendance_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #ced4da;
                gridline-color: #ced4da;
                margin-top: 20px;
            }
            QTableWidget::item {
                text-align: center;
            }
            QHeaderView::section {
                background-color: #e9ecef;
                color: #495057;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.recent_attendance_table)

        self.stack_layout.addWidget(home_widget)

    def setup_history_section(self):
        """Set up the attendance history section with filters and table."""
        history_widget = QWidget()
        layout = QVBoxLayout(history_widget)

        filter_group = QGroupBox("Filter Attendance History")
        filter_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #ced4da;
                border-radius: 8px;
                padding: 15px;
                background-color: #f8f9fa;
                margin-bottom: 20px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                color: #495057;
                font-weight: bold;
                font-size: 14px;
            }
        """)
        filter_layout = QVBoxLayout()

        filter_inner_layout = QHBoxLayout()
        filter_inputs_layout = QFormLayout()

        self.class_combo = QComboBox()
        self.class_combo.setStyleSheet("""
            QComboBox {
                padding: 10px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                font-size: 14px;
                background-color: white;
                min-width: 300px;
                color: #212529;
            }
            QComboBox:hover {
                border: 1px solid #0d6efd;
            }
            QComboBox::drop-down {
                width: 20px;
                border-left: 1px solid #ced4da;
            }
            QComboBox QAbstractItemView {
                background-color: white;
                border: 1px solid #ced4da;
                selection-background-color: #0d6efd;
                selection-color: white;
            }
            QComboBox QAbstractItemView::item {
                min-height: 24px;
                padding: 4px;
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: #e9ecef;
                color: #212529;
            }
            QComboBox QAbstractItemView::item:selected {
                background-color: #0d6efd;
                color: white;
            }
        """)
        filter_inputs_layout.addRow("Class:", self.class_combo)

        self.history_date = QDateEdit()
        self.history_date.setCalendarPopup(True)
        self.history_date.setDate(QDate.currentDate())
        self.history_date.setStyleSheet("""
            QDateEdit {
                padding: 10px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                font-size: 14px;
                background-color: white;
                min-width: 300px;
                color: #212529;
            }
            QDateEdit:hover {
                border: 1px solid #0d6efd;
            }
        """)
        filter_inputs_layout.addRow("Date:", self.history_date)

        filter_button = QPushButton("Apply Filter")
        filter_button.setStyleSheet("""
            QPushButton {
                background-color: #0d6efd;
                color: white;
                padding: 10px;
                border-radius: 4px;
                font-size: 14px;
                max-width: 80px;
                margin-left: 50px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        filter_button.clicked.connect(self.load_attendance_history)
        filter_inputs_layout.addRow(filter_button)

        filter_inner_layout.addLayout(filter_inputs_layout)
        filter_inner_layout.addStretch()

        export_button = QPushButton(QIcon(qta.icon("fa5s.file-export", color="white")), "Export History")
        export_button.setStyleSheet("""
            QPushButton {
                background-color: #0d6efd;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        export_button.clicked.connect(self.export_history)
        filter_inner_layout.addWidget(export_button)

        filter_layout.addLayout(filter_inner_layout)
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(3)
        self.history_table.setHorizontalHeaderLabels(["Class", "Timestamp", "Status"])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.history_table.setShowGrid(True)
        self.history_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #ced4da;
                gridline-color: #ced4da;
                margin-top: 20px;
            }
            QTableWidget::item {
                text-align: center;
            }
            QHeaderView::section {
                background-color: #e9ecef;
                color: #495057;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.history_table)

        self.stack_layout.addWidget(history_widget)

    def setup_ask_for_leave_section(self):
        """Set up the leave request section with class selection and text input."""
        leave_widget = QWidget()
        layout = QVBoxLayout(leave_widget)

        layout.addWidget(QLabel("Ask for Leave"))
        class_layout = QHBoxLayout()
        class_layout.addWidget(QLabel("Select Class:"))
        self.leave_class_combo = QComboBox()
        self.leave_class_combo.setStyleSheet("""
            QComboBox {
                padding: 8px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                font-size: 14px;
                background-color: white;
                min-width: 200px;
                color: #212529;
            }
            QComboBox:hover {
                border: 1px solid #0d6efd;
            }
            QComboBox::drop-down {
                width: 20px;
                border-left: 1px solid #ced4da;
            }
            QComboBox QAbstractItemView {
                background-color: white;
                border: 1px solid #ced4da;
                selection-background-color: #0d6efd;
                selection-color: white;
            }
            QComboBox QAbstractItemView::item {
                min-height: 24px;
                padding: 4px;
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: #e9ecef;
                color: #212529;
            }
            QComboBox QAbstractItemView::item:selected {
                background-color: #0d6efd;
                color: white;
            }
        """)
        class_layout.addWidget(self.leave_class_combo)
        layout.addLayout(class_layout)

        self.leave_textbox = QTextEdit()
        self.leave_textbox.setPlaceholderText(
            "Enter your leave request here (e.g., 'Requesting leave on March 7, 2025 due to illness')..."
        )
        self.leave_textbox.setStyleSheet("""
            QTextEdit {
                padding: 8px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                font-size: 14px;
                background-color: white;
                min-height: 150px;
                min-width: 400px;
            }
            QTextEdit:focus {
                border: 1px solid #0d6efd;
            }
        """)
        layout.addWidget(self.leave_textbox)

        submit_button = QPushButton("Submit Leave Request")
        submit_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
                min-width: 200px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:pressed {
                background-color: #1e7e34;
            }
        """)
        submit_button.clicked.connect(self.submit_leave_request)
        layout.addWidget(submit_button, alignment=Qt.AlignCenter)

        layout.addStretch()
        self.stack_layout.addWidget(leave_widget)

    def setup_settings_section(self):
        """Set up the settings section for changing password."""
        settings_widget = QWidget()
        layout = QVBoxLayout(settings_widget)

        password_group = QGroupBox("Change Password")
        password_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #ced4da;
                border-radius: 8px;
                padding: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                color: black;
            }
        """)
        password_layout = QFormLayout()

        self.current_password = QLineEdit()
        self.current_password.setEchoMode(QLineEdit.Password)
        self.current_password.setStyleSheet("""
            QLineEdit {
                padding: 5px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: white;
                color: black;
                max-width: 300px;
                max-height: 20px;
                margin: 10px;
            }
        """)

        self.new_password = QLineEdit()
        self.new_password.setEchoMode(QLineEdit.Password)
        self.new_password.setStyleSheet("""
            QLineEdit {
                padding: 5px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: white;
                color: black;
                max-width: 300px;
                max-height: 20px;
                margin: 10px;
            }
        """)

        self.confirm_password = QLineEdit()
        self.confirm_password.setEchoMode(QLineEdit.Password)
        self.confirm_password.setStyleSheet("""
            QLineEdit {
                padding: 5px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: white;
                color: black;
                max-width: 300px;
                max-height: 20px;
                margin: 10px;
            }
        """)

        password_layout.addRow("Current Password:", self.current_password)
        password_layout.addRow("New Password:", self.new_password)
        password_layout.addRow("Confirm Password:", self.confirm_password)

        change_password_button = QPushButton("Change Password")
        change_password_button.setStyleSheet("""
            QPushButton {
                background-color: #0d6efd;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                max-width: 150px;
                margin-left: 150px;
            }
        """)
        change_password_button.clicked.connect(self.change_password)
        password_layout.addRow(change_password_button)

        password_group.setLayout(password_layout)
        layout.addWidget(password_group)
        layout.addStretch()
        self.stack_layout.addWidget(settings_widget)

    def create_stat_card(self, title, icon_name):
        """Create a statistics card for the home section."""
        card = QWidget()
        card.setStyleSheet("""
            QWidget {
                background-color: #C0C2C2;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 10px;
            }
        """)
        layout = QVBoxLayout(card)

        icon_label = QLabel()
        icon_label.setPixmap(qta.icon(icon_name, color="#0d6efd").pixmap(32, 32))
        layout.addWidget(icon_label, alignment=Qt.AlignCenter)

        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: bold; font-size: 16px; color: #495057;")
        layout.addWidget(title_label, alignment=Qt.AlignCenter)

        value_label = QLabel("0")
        value_label.setStyleSheet("font-size: 24px; color: #0d6efd;")
        value_label.setObjectName(f"{title.lower().replace(' ', '_')}_value")
        layout.addWidget(value_label, alignment=Qt.AlignCenter)

        return card, value_label

    def sidebar_button_style(self, is_active):
        """Return stylesheet for sidebar buttons based on active state."""
        base_style = """
            QPushButton {
                text-align: left;
                padding: 12px;
                border: none;
                font-size: 16px;
            }
        """
        if is_active:
            return base_style + "background-color: #e9ecef; color: #0d6efd;"
        return base_style + "background-color: transparent; color: #495057;"

    def show_section(self, section_id):
        """Show the specified section in the stacked layout."""
        for button in self.sidebar_button_group:
            is_active = button.property("section_id") == section_id
            button.setStyleSheet(self.sidebar_button_style(is_active))
            button.setChecked(is_active)

        if section_id == "home":
            self.stack_layout.setCurrentIndex(0)
        elif section_id == "history":
            self.stack_layout.setCurrentIndex(1)
            self.load_attendance_history()
        elif section_id == "ask-for-leave":
            self.stack_layout.setCurrentIndex(2)
        elif section_id == "settings":
            self.stack_layout.setCurrentIndex(3)

    def load_data(self):
        """Load initial data for the dashboard."""
        classes = self.db.get_student_classes(self.student_id)
        print(f"Classes for {self.student_name} (ID: {self.student_id}): {classes}")
        self.class_combo.clear()
        self.home_class_combo.clear()
        self.leave_class_combo.clear()
        self.class_combo.addItem("All Classes", None)
        self.home_class_combo.addItem("All Classes", None)
        self.leave_class_combo.addItem("Select Class", None)
        for class_data in classes:
            self.class_combo.addItem(class_data[1], class_data[0])
            self.home_class_combo.addItem(class_data[1], class_data[0])
            self.leave_class_combo.addItem(class_data[1], class_data[0])

        self.update_statistics()
        self.load_recent_attendance()

    def update_statistics(self):
        """Update statistics for total, attended, and missed classes."""
        total_records = 0
        present_records = 0
        classes = set()

        for class_data in self.db.get_student_classes(self.student_id):
            classes.add(class_data[0])
            records = self.db.get_attendance_records(class_id=class_data[0], student_id=self.student_id)
            total_records += len(records)
            present_records += len([r for r in records if r[4] == 'present'])

        missed_records = total_records - present_records
        print(
            f"Stats for {self.student_name}: Total Classes={len(classes)}, "
            f"Total Records={total_records}, Present={present_records}, Missed={missed_records}"
        )
        self.total_classes_value.setText(str(len(classes)))
        self.missed_classes_value.setText(str(missed_records))
        self.attended_classes_value.setText(str(present_records))

    def load_recent_attendance(self):
        """Load recent attendance records for the home section."""
        selected_class = self.home_class_combo.currentData()
        records = []

        print(f"Loading recent attendance for student ID {self.student_id}, selected class: {selected_class}")
        if selected_class:
            class_data = next((c for c in self.db.get_student_classes(self.student_id) if c[0] == selected_class), None)
            if class_data:
                class_id = class_data[0]
                class_name = class_data[1]
                class_records = self.db.get_attendance_records(class_id=class_id, student_id=self.student_id)
                print(f"Records for class {class_name} (ID: {class_id}): {class_records}")
                for record in class_records:
                    records.append({
                        'class': class_name,
                        'timestamp': record[3],
                        'status': record[4],
                    })
        else:
            for class_data in self.db.get_student_classes(self.student_id):
                class_id = class_data[0]
                class_name = class_data[1]
                class_records = self.db.get_attendance_records(class_id=class_id, student_id=self.student_id)
                print(f"Records for class {class_name} (ID: {class_id}): {class_records}")
                for record in class_records:
                    records.append({
                        'class': class_name,
                        'timestamp': record[3],
                        'status': record[4],
                    })

        records.sort(key=lambda x: x['timestamp'], reverse=True)
        records = records[:10]
        print(f"Final recent attendance records: {records}")

        self.recent_attendance_table.setRowCount(len(records))
        for i, record in enumerate(records):
            self.recent_attendance_table.setItem(i, 0, QTableWidgetItem(record['class']))
            self.recent_attendance_table.setItem(i, 1, QTableWidgetItem(str(record['timestamp'])))
            self.recent_attendance_table.setItem(i, 2, QTableWidgetItem(record['status']))

    def load_attendance_history(self):
        """Load attendance history based on selected class and date."""
        selected_date = self.history_date.date().toString("yyyy-MM-dd")
        selected_class = self.class_combo.currentData()

        records = []
        classes = [selected_class] if selected_class else [c[0] for c in self.db.get_student_classes(self.student_id)]

        print(
            f"Loading history for student ID {self.student_id}, date {selected_date}, selected class: {selected_class}"
        )
        for class_id in classes:
            class_data = next((c for c in self.db.get_student_classes(self.student_id) if c[0] == class_id), None)
            if class_data:
                class_name = class_data[1]
                class_records = self.db.get_attendance_records(
                    class_id=class_id,
                    student_id=self.student_id,
                    date=selected_date,
                )
                print(f"Records for class {class_name} (ID: {class_id}), date {selected_date}: {class_records}")
                for record in class_records:
                    records.append({
                        'class': class_name,
                        'timestamp': record[3],
                        'status': record[4],
                    })

        print(f"Final history records: {records}")
        self.history_table.setRowCount(len(records))
        for i, record in enumerate(records):
            self.history_table.setItem(i, 0, QTableWidgetItem(record['class']))
            self.history_table.setItem(i, 1, QTableWidgetItem(str(record['timestamp'])))
            self.history_table.setItem(i, 2, QTableWidgetItem(record['status']))

    def export_history(self):
        """Export attendance history to a CSV file."""
        try:
            filename, _ = QFileDialog.getSaveFileName(self, "Export History", "", "CSV Files (*.csv)")
            if filename:
                with open(filename, 'w') as f:
                    f.write("Class,Timestamp,Status\n")
                    for row in range(self.history_table.rowCount()):
                        row_data = []
                        for col in range(self.history_table.columnCount()):
                            item = self.history_table.item(row, col)
                            row_data.append(item.text() if item else "")
                        f.write(",".join(row_data) + "\n")
                QMessageBox.information(self, "Success", "Attendance history exported successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export history: {str(e)}")

    def change_password(self):
        """Change the student's password."""
        if not all([self.current_password.text(), self.new_password.text(), self.confirm_password.text()]):
            QMessageBox.warning(self, "Error", "Please fill in all password fields")
            return
        if self.new_password.text() != self.confirm_password.text():
            QMessageBox.warning(self, "Error", "New passwords do not match")
            return
        if self.auth_manager.change_password(self.student_id, self.current_password.text(), self.new_password.text()):
            QMessageBox.information(self, "Success", "Password changed successfully")
            self.current_password.clear()
            self.new_password.clear()
            self.confirm_password.clear()
        else:
            QMessageBox.warning(self, "Error", "Failed to change password. Please check your current password.")

    def submit_leave_request(self):
        """Submit a leave request for the selected class."""
        leave_request = self.leave_textbox.toPlainText().strip()
        class_id = self.leave_class_combo.currentData()
        if not leave_request:
            QMessageBox.warning(self, "Error", "Please enter a leave request")
            return
        if not class_id or class_id is None:
            QMessageBox.warning(self, "Error", "Please select a class")
            return
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO leave_requests (student_id, class_id, request) VALUES (?, ?, ?)",
                (self.student_id, class_id, leave_request),
            )
            conn.commit()
            conn.close()
            QMessageBox.information(self, "Success", "Leave request submitted successfully")
            self.leave_textbox.clear()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to submit leave request: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StudentDashboard(None, {'id': 2, 'name': 'ismail'})
    window.show()
    sys.exit(app.exec_())