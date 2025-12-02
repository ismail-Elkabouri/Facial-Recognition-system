import sys
import sqlite3
import cv2
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
    QScrollArea,
    QGroupBox,
    QFormLayout,
    QDateEdit,
    QLineEdit,
    QMessageBox,
    QFileDialog,
    QListWidget,
    QListWidgetItem, QHeaderView, QStackedLayout
)
from PyQt5.QtCore import Qt, QTimer, QDate, pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QImage, QPixmap
import qtawesome as qta
from src.recognition.recognize_faces import FaceRecognition
from database.db_helper import DatabaseHelper


class TeacherDashboard(QMainWindow):
    """Teacher dashboard for managing classes, attendance, and leave requests."""
    logout_requested = pyqtSignal()

    def __init__(self, auth_manager, teacher_data):
        super().__init__()
        self.auth_manager = auth_manager
        self.teacher_id = teacher_data['id']
        self.teacher_name = teacher_data['name']
        self.db = DatabaseHelper()

        # Initialize database connection
        try:
            self.conn = self.db.get_connection()
            if self.conn is None:
                raise Exception("Failed to connect to the database")
        except Exception as e:
            QMessageBox.critical(self, "Database Error", f"Could not connect to the database: {e}")
            sys.exit()

        # Initialize face recognition and camera
        self.face_recognition = None
        self.capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.recognized_students = set()
        self.current_class_id = None
        self.current_session_timestamp = None
        self.class_students = {}

        self.setup_ui()
        self.load_data()

    def setup_ui(self):
        """Set up the main UI components of the dashboard."""
        self.setWindowTitle(f"Teacher Dashboard - {self.teacher_name}")
        self.setMinimumSize(1800, 1200)

        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Top bar
        top_bar = QWidget()
        top_bar.setStyleSheet("background-color: #0d6efd; min-height: 100px;")
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(20, 0, 20, 0)

        header_label = QLabel(f"Welcome {self.teacher_name}")
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
        logout_button.clicked.connect(self.handle_logout)
        top_bar_layout.addWidget(logout_button)
        main_layout.addWidget(top_bar)

        # Content area
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
        self.sidebar.setStyleSheet("background-color: #cccccc;")
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        sidebar_buttons = [
            ("Home", "fa5s.home", "home"),
            ("Take Attendance", "fa5s.camera", "take-attendance"),
            ("Attendance Records", "fa5s.calendar-check", "attendance-records"),
            ("Class Management", "fa5s.book", "class-management"),
            ("Leave Requests", "fa5s.calendar-times", "leave-requests"),
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
        self.setup_attendance_section()
        self.setup_records_section()
        self.setup_class_management_section()
        self.setup_leave_requests_section()
        self.setup_settings_section()

    def setup_home_section(self):
        """Set up the home section with statistics and recent attendance."""
        home_widget = QWidget()
        layout = QVBoxLayout(home_widget)

        overview_layout = QHBoxLayout()

        self.total_students_card = self.create_stat_card("Total Students", "fa5s.user-graduate")
        self.total_classes_card = self.create_stat_card("Total Classes", "fa5s.chalkboard")
        self.attendance_rate_card = self.create_stat_card("Attendance Rate", "fa5s.chart-line")

        overview_layout.addWidget(self.total_students_card)
        overview_layout.addWidget(self.total_classes_card)
        overview_layout.addWidget(self.attendance_rate_card)
        layout.addLayout(overview_layout)

        layout.addWidget(QLabel("Recent Attendance"))
        self.recent_attendance_table = QTableWidget()
        self.recent_attendance_table.setColumnCount(4)
        self.recent_attendance_table.setHorizontalHeaderLabels(["Student", "Class", "Date and Time", "Status"])
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

    def setup_attendance_section(self):
        """Set up the attendance section with class selection and video feed."""
        attendance_widget = QWidget()
        layout = QVBoxLayout(attendance_widget)

        # Class selection
        class_layout = QHBoxLayout()
        class_layout.addWidget(QLabel("Select Class:"))
        self.class_combo = QComboBox()
        self.class_combo.setStyleSheet("""
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
        """)
        class_layout.addWidget(self.class_combo)
        layout.addLayout(class_layout)

        # Main container for video and buttons
        main_container = QHBoxLayout()

        # Video label
        self.video_label = QLabel("Webcam Feed")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setScaledContents(True)
        self.video_label.setStyleSheet("""
            background-color: black;
            color: red;
            border: 4px solid green;
            font-size: 18px;
            border-radius: 10px;
        """)
        main_container.addWidget(self.video_label, stretch=1)

        # Button sidebar
        button_sidebar = QWidget()
        button_sidebar_layout = QVBoxLayout(button_sidebar)
        button_sidebar.setFixedWidth(200)
        button_sidebar.setStyleSheet("background-color: #000000; border-radius: 2px; max-width: 500px;")

        self.stats_label = QLabel("Recognized Students:\nNone")
        self.stats_label.setStyleSheet("font-size: 16px; color: green; padding: 10px; text-align: left;")
        self.stats_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        button_sidebar_layout.addWidget(self.stats_label)

        button_layout = QHBoxLayout()
        button_layout.setAlignment(Qt.AlignCenter)

        self.start_button = QPushButton("Start Attendance")
        self.start_button.setStyleSheet("""
            background-color: #28a745;
            color: white;
            max-height: 50px;
            max-width: 200px;
            padding: 10px;
            font-size: 16px;
            border-radius: 5px;
        """)
        self.start_button.clicked.connect(self.start_attendance)
        button_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Attendance")
        self.stop_button.setStyleSheet("""
            background-color: #dc3545;
            color: white;
            max-height: 50px;
            max-width: 150px;
            padding: 10px;
            font-size: 16px;
            border-radius: 5px;
        """)
        self.stop_button.clicked.connect(self.stop_attendance)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)

        button_sidebar_layout.addStretch()
        button_sidebar_layout.addLayout(button_layout)
        main_container.addWidget(button_sidebar)

        layout.addLayout(main_container)
        self.stack_layout.addWidget(attendance_widget)

    def setup_records_section(self):
        """Set up the attendance records section with filtering and export options."""
        records_widget = QWidget()
        layout = QVBoxLayout(records_widget)

        # Filter section
        filter_group = QGroupBox("Filter Attendance Records")
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
        self.records_class_combo = QComboBox()
        self.records_class_combo.setStyleSheet("""
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
        """)
        filter_inputs_layout.addRow("Class:", self.records_class_combo)

        self.records_date = QDateEdit()
        self.records_date.setCalendarPopup(True)
        self.records_date.setDate(QDate.currentDate())
        self.records_date.setStyleSheet("""
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
        filter_inputs_layout.addRow("Date:", self.records_date)

        filter_button = QPushButton("Apply Filter")
        filter_button.setStyleSheet("""
            QPushButton {
                background-color: #0d6efd;
                max-width: 80px;
                margin-left: 50px;
                color: white;
                padding: 10px;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        filter_button.clicked.connect(self.load_attendance_records)
        filter_inputs_layout.addRow(filter_button)

        filter_inner_layout.addLayout(filter_inputs_layout)
        filter_inner_layout.addStretch()

        export_button = QPushButton(QIcon(qta.icon("fa5s.file-export", color="white")), "Export Records")
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
        export_button.clicked.connect(self.export_records)
        filter_inner_layout.addWidget(export_button)

        filter_layout.addLayout(filter_inner_layout)
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)

        # Records table
        self.records_table = QTableWidget()
        self.records_table.setColumnCount(4)
        self.records_table.setHorizontalHeaderLabels(["Student", "Class", "Date and Time", "Status"])
        self.records_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.records_table.setShowGrid(True)
        self.records_table.setStyleSheet("""
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
        layout.addWidget(self.records_table)

        self.stack_layout.addWidget(records_widget)
        self.populate_records_class_combo()

    def setup_class_management_section(self):
        """Set up the class management section for enrolling students and viewing classes."""
        class_management_widget = QWidget()
        layout = QVBoxLayout(class_management_widget)
        layout.setSpacing(20)

        enroll_group = QGroupBox("Enroll Students")
        enroll_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #ced4da;
                border-radius: 8px;
                padding: 15px;
                background-color: #f8f9fa;
                margin-top: 10px;
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
        enroll_layout = QVBoxLayout()
        enroll_layout.setSpacing(10)

        enroll_layout.addWidget(QLabel("Select Class:"))
        self.enroll_class_combo = QComboBox()
        self.enroll_class_combo.setStyleSheet("""
            QComboBox {
                padding: 10px;
                max-width: 400px;
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
        """)
        enroll_layout.addWidget(self.enroll_class_combo)

        enroll_layout.addWidget(QLabel("Select Student:"))
        self.enroll_student_combo = QComboBox()
        self.enroll_student_combo.setStyleSheet("""
            QComboBox {
                padding: 10px;
                max-width: 400px;
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
        """)
        enroll_layout.addWidget(self.enroll_student_combo)

        enroll_button = QPushButton("Enroll Student")
        enroll_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
                min-width: 140px;
                max-width: 140px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:pressed {
                background-color: #1e7e34;
            }
        """)
        enroll_button.clicked.connect(self.enroll_student)
        enroll_layout.addWidget(enroll_button, alignment=Qt.AlignLeft)
        enroll_layout.addStretch()
        enroll_group.setLayout(enroll_layout)
        layout.addWidget(enroll_group)

        class_list_group = QGroupBox("My Classes")
        class_list_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #ced4da;
                border-radius: 8px;
                padding: 15px;
                background-color: #f8f9fa;
                margin-top: 20px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                color: black;
                font-weight: bold;
                font-size: 14px;
            }
        """)
        class_list_layout = QVBoxLayout()
        self.class_list_widget = QListWidget()
        self.class_list_widget.setStyleSheet("""
            QListWidget {
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: white;
                font-size: 14px;
            }
            QListWidget::item {
                margin: 5px;
            }
        """)
        class_list_layout.addWidget(self.class_list_widget)
        class_list_group.setLayout(class_list_layout)
        layout.addWidget(class_list_group)

        layout.addStretch()
        self.load_teacher_classes()
        self.load_enrollment_options()
        self.stack_layout.addWidget(class_management_widget)

    def setup_leave_requests_section(self):
        """Set up the leave requests section with a table of requests."""
        leave_requests_widget = QWidget()
        layout = QVBoxLayout(leave_requests_widget)

        layout.addWidget(QLabel("Pending Leave Requests"))
        self.leave_requests_table = QTableWidget()
        self.leave_requests_table.setColumnCount(5)
        self.leave_requests_table.setHorizontalHeaderLabels(["Student", "Request", "Status", "Timestamp", "Action"])
        self.leave_requests_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.leave_requests_table.setShowGrid(True)
        self.leave_requests_table.setStyleSheet("""
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
        layout.addWidget(self.leave_requests_table)

        self.load_leave_requests()
        self.stack_layout.addWidget(leave_requests_widget)

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
            color: black;
            padding: 5px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            background-color: white;
            max-width: 300px;
            max-height: 20px;
            margin: 10px;
        """)

        self.new_password = QLineEdit()
        self.new_password.setEchoMode(QLineEdit.Password)
        self.new_password.setStyleSheet("""
            color: black;
            padding: 5px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            background-color: white;
            max-width: 300px;
            max-height: 20px;
            margin: 10px;
        """)

        self.confirm_password = QLineEdit()
        self.confirm_password.setEchoMode(QLineEdit.Password)
        self.confirm_password.setStyleSheet("""
            color: black;
            padding: 5px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            background-color: white;
            max-width: 300px;
            max-height: 20px;
            margin: 10px;
        """)

        password_layout.addRow("Current Password:", self.current_password)
        password_layout.addRow("New Password:", self.new_password)
        password_layout.addRow("Confirm Password:", self.confirm_password)

        change_password_button = QPushButton("Change Password")
        change_password_button.setStyleSheet("""
            QPushButton {
                max-width: 150px;
                margin-left: 150px;
                background-color: #0d6efd;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
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
                background-color: #cccccc;
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
        title_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(title_label, alignment=Qt.AlignCenter)

        value_label = QLabel("0")
        value_label.setStyleSheet("font-size: 24px; color: #0d6efd;")
        value_label.setObjectName(f"{title.lower().replace(' ', '_')}_value")
        layout.addWidget(value_label, alignment=Qt.AlignCenter)

        return card

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
            button.setStyleSheet(self.sidebar_button_style(False))
            button.setChecked(False)

        section_indices = {
            "home": 0,
            "take-attendance": 1,
            "attendance-records": 2,
            "class-management": 3,
            "leave-requests": 4,
            "settings": 5
        }
        self.stack_layout.setCurrentIndex(section_indices.get(section_id, 0))

        for button in self.sidebar_button_group:
            if button.property("section_id") == section_id:
                button.setStyleSheet(self.sidebar_button_style(True))
                button.setChecked(True)
                break

    def load_data(self):
        """Load initial data for the dashboard."""
        self.populate_class_combo()
        self.update_statistics()
        self.load_recent_attendance()

    def populate_class_combo(self):
        """Populate the class selection combo box."""
        self.class_combo.clear()
        classes = self.db.get_teacher_classes(self.teacher_id)
        for class_id, class_name, _ in classes:
            self.class_combo.addItem(class_name, class_id)

    def populate_records_class_combo(self):
        """Populate the class selection combo box for records."""
        self.records_class_combo.clear()
        self.records_class_combo.addItem("All Classes", -1)
        classes = self.db.get_teacher_classes(self.teacher_id)
        for class_id, class_name, _ in classes:
            self.records_class_combo.addItem(class_name, class_id)

    def update_statistics(self):
        """Update statistics displayed on the home section."""
        classes = self.db.get_teacher_classes(self.teacher_id)
        total_students = 0
        total_records = 0
        present_records = 0

        for class_data in classes:
            class_id = class_data[0]
            students = set()
            records = self.db.get_attendance_records(class_id=class_id)
            for record in records:
                students.add(record[2])  # student_id
                total_records += 1
                if record[4] == 'present':
                    present_records += 1
            total_students += len(students)

        self.total_students_card.findChild(QLabel, "total_students_value").setText(str(total_students))
        self.total_classes_card.findChild(QLabel, "total_classes_value").setText(str(len(classes)))
        attendance_rate = (present_records / total_records * 100) if total_records > 0 else 0
        self.attendance_rate_card.findChild(QLabel, "attendance_rate_value").setText(f"{attendance_rate:.1f}%")

    def load_recent_attendance(self):
        """Load recent attendance records for the selected class."""
        class_id = self.class_combo.currentData()
        if not class_id:
            self.recent_attendance_table.setRowCount(0)
            return

        students = self.db.get_students_in_class(class_id)
        if not students:
            self.recent_attendance_table.setRowCount(0)
            return

        records = self.db.get_attendance_records(class_id=class_id)
        student_statuses = {}
        for record in records:
            student_id = record[2]
            try:
                date_time = datetime.strptime(record[3], '%Y-%m-%d %H:%M:%S')
            except ValueError:
                try:
                    date_time = datetime.strptime(record[3], '%Y-%m-%d')
                    date_time = date_time.replace(hour=0, minute=0, second=0)
                except ValueError:
                    date_time = datetime.now()
            if student_id not in student_statuses or date_time > student_statuses[student_id]['datetime']:
                student_statuses[student_id] = {
                    'status': record[4],
                    'datetime': date_time
                }

        self.recent_attendance_table.setRowCount(len(students))
        for i, (student_id, student_name) in enumerate(students):
            class_name = self.class_combo.currentText()
            status = 'absent'
            datetime_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if self.current_session_timestamp:
                if student_id in self.recognized_students:
                    status = 'present'
                datetime_str = self.current_session_timestamp.strftime('%Y-%m-%d %H:%M:%S')
            elif student_id in student_statuses:
                status = student_statuses[student_id]['status']
                datetime_str = student_statuses[student_id]['datetime'].strftime('%Y-%m-%d %H:%M:%S')

            self.recent_attendance_table.setItem(i, 0, QTableWidgetItem(student_name))
            self.recent_attendance_table.setItem(i, 1, QTableWidgetItem(class_name))
            self.recent_attendance_table.setItem(i, 2, QTableWidgetItem(datetime_str))
            self.recent_attendance_table.setItem(i, 3, QTableWidgetItem(status))

    def start_camera(self):
        """Initialize and start the webcam."""
        if self.capture is None:
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                QMessageBox.critical(self, "Error", "Failed to open webcam")
                self.capture = None
                return False

            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            actual_width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"Webcam resolution set to: {actual_width}x{actual_height}")
        return True

    def stop_camera(self):
        """Stop and release the webcam."""
        if self.capture is not None:
            self.timer.stop()
            self.capture.release()
            self.capture = None
            self.video_label.setText("Webcam Feed")
            QApplication.processEvents()

    def update_frame(self):
        """Update the video frame and process face recognition."""
        if self.capture is None or not self.class_combo.currentData():
            return

        ret, frame = self.capture.read()
        if not ret:
            return

        if not self.start_button.isEnabled() and self.face_recognition:
            # Handle face recognition output (name, box)
            recognized_faces = self.face_recognition.recognize_faces(frame)
            print(f"Recognized faces: {recognized_faces}")
            for name, box in recognized_faces:
                print(f"Name: {name}, Box: {box}")
                if name != "Unknown":
                    student_id = self.face_recognition.name_to_id.get(name)
                    if student_id:
                        self.recognized_students.add(student_id)
                        print(f"Added student ID {student_id} to recognized_students")
                    else:
                        print(f"No ID for name: {name}")

                    # Draw bounding box and name on the frame
                    x, y, w, h = box
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Update stats label with recognized student names
            if self.recognized_students:
                student_names = [
                    self.auth_manager.get_user_info(sid)['name']
                    for sid in self.recognized_students
                    if self.auth_manager.get_user_info(sid)
                ]
                display_text = "Recognized Students:\n" + "\n".join(sorted(student_names))
                print(f"Updating sidebar with: {display_text}")
                self.stats_label.setText(display_text)
            else:
                self.stats_label.setText("Recognized Students:\nNone")

        # Convert frame to QPixmap for display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap)

    def start_attendance(self):
        """Start an attendance session with face recognition."""
        if not self.class_combo.currentData():
            QMessageBox.warning(self, "Error", "Please select a class first")
            return

        try:
            if not self.face_recognition:
                self.face_recognition = FaceRecognition(self.auth_manager, self.db)
                print(f"Initialized FaceRecognition with auth_manager: {self.auth_manager}, db: {self.db}")
                print(f"Name-to-ID mapping: {self.face_recognition.name_to_id}")

            self.current_class_id = self.class_combo.currentData()
            self.face_recognition.start_attendance_session(self.current_class_id)
            self.recognized_students.clear()
            self.current_session_timestamp = datetime.now()

            self.class_students.clear()
            students = self.db.get_students_in_class(self.current_class_id)
            for student_id, student_name in students:
                self.class_students[student_id] = {
                    'name': student_name,
                    'status': 'absent',
                    'timestamp': self.current_session_timestamp
                }

            self.stats_label.setText("Recognized Students:\nNone")
            self.load_recent_attendance()
            if self.start_camera():
                self.start_button.setEnabled(False)
                self.stop_button.setEnabled(True)
                self.class_combo.setEnabled(False)
                self.timer.start(30)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start attendance: {str(e)}")
            print(f"Error in start_attendance: {str(e)}")

    def stop_attendance(self):
        """Stop the attendance session and record attendance."""
        self.stop_camera()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.class_combo.setEnabled(True)

        if self.current_class_id:
            now = self.current_session_timestamp or datetime.now()
            timestamp_str = now.strftime('%Y-%m-%d %H:%M:%S')
            for student_id, student_data in self.class_students.items():
                status = 'present' if student_id in self.recognized_students else 'absent'
                self.db.record_attendance(self.current_class_id, student_id, timestamp_str, status)
                print(f"Attendance recorded: {student_data['name']} (ID: {student_id}) "
                      f"for class {self.current_class_id} on {timestamp_str} as {status}")
            self.load_recent_attendance()
            self.update_statistics()
            self.recognized_students.clear()
            self.current_session_timestamp = None
            self.current_class_id = None

    def load_attendance_records(self):
        """Load attendance records based on selected filters."""
        selected_date = self.records_date.date().toPyDate()
        selected_class_id = self.records_class_combo.currentData()
        start_of_day = datetime.combine(selected_date, datetime.min.time())
        end_of_day = datetime.combine(selected_date, datetime.max.time())

        records = []
        classes = self.db.get_teacher_classes(self.teacher_id)
        for class_data in classes:
            class_id = class_data[0]
            class_name = class_data[1]
            if selected_class_id != -1 and class_id != selected_class_id:
                continue
            class_records = self.db.get_attendance_records(class_id=class_id)
            for record in class_records:
                student_id = record[2]
                timestamp_str = record[3]
                status = record[4]
                try:
                    try:
                        date_time = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
                    except ValueError:
                        date_time = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    if start_of_day <= date_time <= end_of_day:
                        student = self.auth_manager.get_user_info(student_id)
                        if student:
                            records.append({
                                'student': student['name'],
                                'class': class_name,
                                'datetime': date_time,
                                'status': status
                            })
                except ValueError as e:
                    print(f"Error parsing timestamp '{timestamp_str}' for record {record}: {e}")
                    continue

        records.sort(key=lambda x: x['datetime'], reverse=True)
        self.records_table.setRowCount(len(records))
        if not records:
            QMessageBox.information(self, "No Records",
                                    f"No attendance records found for {selected_date.strftime('%Y-%m-%d')}.")
        for i, record in enumerate(records):
            self.records_table.setItem(i, 0, QTableWidgetItem(record['student']))
            self.records_table.setItem(i, 1, QTableWidgetItem(record['class']))
            date_str = record['datetime'].strftime('%Y-%m-%d %H:%M:%S')
            self.records_table.setItem(i, 2, QTableWidgetItem(date_str))
            self.records_table.setItem(i, 3, QTableWidgetItem(record['status']))

    def export_records(self):
        """Export attendance records to a CSV file."""
        try:
            filename, _ = QFileDialog.getSaveFileName(self, "Export Records", "", "CSV Files (*.csv)")
            if filename:
                with open(filename, 'w') as f:
                    f.write("Student,Class,Date and Time,Status\n")
                    for row in range(self.records_table.rowCount()):
                        row_data = []
                        for col in range(self.records_table.columnCount()):
                            item = self.records_table.item(row, col)
                            row_data.append(item.text() if item else "")
                        f.write(",".join(row_data) + "\n")
                QMessageBox.information(self, "Success", "Records exported successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export records: {str(e)}")

    def delete_class(self):
        """Delete a class after confirmation."""
        button = self.sender()
        class_id = button.property("class_id")
        reply = QMessageBox.question(self, 'Delete Class',
                                     "Are you sure you want to delete this class?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            if self.db.delete_class(class_id):
                QMessageBox.information(self, "Success", "Class deleted successfully.")
                self.load_teacher_classes()
                self.load_data()
            else:
                QMessageBox.critical(self, "Error", "Failed to delete class.")

    def load_teacher_classes(self):
        """Load the teacher's classes into the class list widget."""
        self.class_list_widget.clear()
        classes = self.db.get_teacher_classes(self.teacher_id)
        for class_id, class_name, _ in classes:
            item_widget = QWidget()
            item_layout = QHBoxLayout(item_widget)
            item_layout.setContentsMargins(0, 0, 0, 0)
            class_label = QLabel(f"{class_name} (ID: {class_id})")
            item_layout.addWidget(class_label)
            delete_button = QPushButton("Delete")
            delete_button.setIcon(qta.icon('fa5s.trash-alt', color='white'))
            delete_button.setStyleSheet("""
                QPushButton {
                    max-width: 100px;
                    background-color: #dc3545;
                    color: white;
                    padding: 10px 10px;
                    border-radius: 4px;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #c82333;
                }
            """)
            delete_button.setProperty("class_id", class_id)
            delete_button.clicked.connect(self.delete_class)
            item_layout.addWidget(delete_button)
            item_layout.addStretch()
            list_item = QListWidgetItem()
            list_item.setSizeHint(item_widget.sizeHint())
            self.class_list_widget.addItem(list_item)
            self.class_list_widget.setItemWidget(list_item, item_widget)

    def load_enrollment_options(self):
        """Load classes and students for enrollment options."""
        classes = self.db.get_teacher_classes(self.teacher_id)
        self.enroll_class_combo.clear()
        for class_data in classes:
            self.enroll_class_combo.addItem(class_data[1], class_data[0])

        students = self.auth_manager.get_all_users(role='student')
        self.enroll_student_combo.clear()
        for student in students:
            self.enroll_student_combo.addItem(student['name'], student['id'])

    def enroll_student(self):
        """Enroll a student in a class."""
        class_id = self.enroll_class_combo.currentData()
        student_id = self.enroll_student_combo.currentData()
        if not class_id or not student_id:
            QMessageBox.warning(self, "Error", "Please select a class and a student")
            return

        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO class_enrollments (class_id, student_id)
                VALUES (?, ?)
            """, (class_id, student_id))
            conn.commit()
            conn.close()
            QMessageBox.information(self, "Success", f"Student ID {student_id} enrolled in class ID {class_id}")
            print(f"Enrolled student ID {student_id} in class ID {class_id}")
            self.load_teacher_classes()
        except sqlite3.IntegrityError:
            QMessageBox.warning(self, "Error", "Student is already enrolled in this class")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to enroll student: {str(e)}")

    def change_password(self):
        """Change the teacher's password."""
        if not all([self.current_password.text(), self.new_password.text(), self.confirm_password.text()]):
            QMessageBox.warning(self, "Error", "Please fill in all password fields")
            return
        if self.new_password.text() != self.confirm_password.text():
            QMessageBox.warning(self, "Error", "New passwords do not match")
            return
        if self.auth_manager.change_password(self.teacher_id,
                                            self.current_password.text(),
                                            self.new_password.text()):
            QMessageBox.information(self, "Success", "Password changed successfully")
            self.current_password.clear()
            self.new_password.clear()
            self.confirm_password.clear()
        else:
            QMessageBox.warning(self, "Error", "Failed to change password. Please check your current password.")

    def handle_logout(self):
        """Handle logout action."""
        self.stop_camera()
        self.logout_requested.emit()

    def closeEvent(self, event):
        """Handle window close event."""
        self.stop_camera()
        super().closeEvent(event)

    def load_leave_requests(self):
        """Load pending leave requests for the teacher's classes."""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        try:
            classes = [class_data[0] for class_data in self.db.get_teacher_classes(self.teacher_id)]
            if not classes:
                self.leave_requests_table.setRowCount(0)
                return
            cursor.execute("""
                SELECT lr.id, u.name, lr.request, lr.status, lr.timestamp
                FROM leave_requests lr
                JOIN users u ON u.id = lr.student_id
                WHERE lr.class_id IN ({})
                GROUP BY lr.id, u.name, lr.request, lr.status, lr.timestamp
            """.format(','.join('?' * len(classes))), classes)
            requests = cursor.fetchall()
            print(f"Loaded leave requests for teacher {self.teacher_name}: {requests}")

            self.leave_requests_table.setRowCount(len(requests))
            for i, (request_id, student_name, request_text, status, timestamp) in enumerate(requests):
                self.leave_requests_table.setItem(i, 0, QTableWidgetItem(student_name))
                self.leave_requests_table.setItem(i, 1, QTableWidgetItem(request_text))
                self.leave_requests_table.setItem(i, 2, QTableWidgetItem(status))
                self.leave_requests_table.setItem(i, 3, QTableWidgetItem(timestamp))

                if status == 'pending':
                    approve_button = QPushButton("Approve")
                    approve_button.setStyleSheet("""
                        QPushButton {
                            background-color: #28a745;
                            color: white;
                            max-width: 50px;
                            max-height: 20px;
                            padding: 10px;
                            border-radius: 4px;
                            font-size: 12px;
                        }
                        QPushButton:hover {
                            background-color: #218838;
                        }
                    """)
                    approve_button.clicked.connect(
                        lambda checked, rid=request_id: self.update_leave_request_status(rid, "approved"))

                    reject_button = QPushButton("Reject")
                    reject_button.setStyleSheet("""
                        QPushButton {
                            background-color: #dc3545;
                            color: white;
                            max-width: 50px;
                            max-height: 20px;
                            padding: 10px;
                            border-radius: 4px;
                            font-size: 12px;
                            margin-left: 5px;
                        }
                        QPushButton:hover {
                            background-color: #c82333;
                        }
                    """)
                    reject_button.clicked.connect(
                        lambda checked, rid=request_id: self.update_leave_request_status(rid, "rejected"))

                    widget = QWidget()
                    layout = QHBoxLayout(widget)
                    layout.addWidget(approve_button)
                    layout.addWidget(reject_button)
                    layout.setContentsMargins(0, 0, 0, 0)
                    self.leave_requests_table.setCellWidget(i, 4, widget)
                else:
                    self.leave_requests_table.setItem(i, 4, QTableWidgetItem(""))
        except sqlite3.Error as e:
            print(f"Error loading leave requests: {e}")
            self.leave_requests_table.setRowCount(0)
        finally:
            conn.close()

    def update_leave_request_status(self, request_id, status):
        """Update the status of a leave request."""
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE leave_requests SET status = ? WHERE id = ?
            """, (status, request_id))
            conn.commit()
            conn.close()
            QMessageBox.information(self, "Success", f"Leave request {request_id} marked as {status}")
            self.load_leave_requests()
        except sqlite3.Error as e:
            QMessageBox.critical(self, "Error", f"Failed to update leave request: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TeacherDashboard(None, {'id': 1, 'name': 'Test Teacher'})
    window.show()
    sys.exit(app.exec_())