import sys
from datetime import datetime

import qtawesome as qta
import sqlite3
from PyQt5.QtCore import QDate, Qt, pyqtSignal, QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStackedLayout,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget, QFileDialog,
)

from database.db_helper import DatabaseHelper


class AdminDashboard(QMainWindow):
    """Admin dashboard for managing users, classes, reports, and settings."""

    logout_requested = pyqtSignal()

    def __init__(self, auth_manager, admin_data):
        """Initialize the AdminDashboard with authentication and admin data."""
        super().__init__()
        self.auth_manager = auth_manager
        self.admin_id = admin_data['id']
        self.admin_name = admin_data['name']
        self.db = DatabaseHelper()

        try:
            self.conn = self.db.get_connection()
            if self.conn is None:
                raise Exception("Failed to connect to the database")
        except Exception as e:
            QMessageBox.critical(self, "Database Error", f"Could not connect to the database: {e}")
            sys.exit()

        print(f"AdminDashboard initialized for {self.admin_name} (ID: {self.admin_id})")

        self.setup_ui()
        self.load_data()

    def setup_ui(self):
        """Set up the main UI components of the dashboard."""
        self.setWindowTitle(f"Admin Dashboard - {self.admin_name}")
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

        header_label = QLabel(f"Welcome, {self.admin_name}")
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
        self.sidebar.setStyleSheet("background-color: #E6E6E6")
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        sidebar_buttons = [
            ("Home", "fa5s.home", "home"),
            ("User Management", "fa5s.users", "users"),
            ("Class Management", "fa5s.book", "classes"),
            ("Reports", "fa5s.chart-bar", "reports"),
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
        self.setup_users_section()
        self.setup_classes_section()
        self.setup_reports_section()
        self.setup_settings_section()

    def setup_home_section(self):
        """Set up the home section with statistics and recent activity."""
        home_widget = QWidget()
        layout = QVBoxLayout(home_widget)

        overview_layout = QHBoxLayout()
        self.total_users_card, self.total_users_value = self.create_stat_card("Total Users", "fa5s.users")
        self.total_classes_card, self.total_classes_value = self.create_stat_card("Total Classes", "fa5s.chalkboard")
        self.attendance_rate_card, self.attendance_rate_value = self.create_stat_card("Overall Attendance",
                                                                                      "fa5s.chart-line")
        overview_layout.addWidget(self.total_users_card)
        overview_layout.addWidget(self.total_classes_card)
        overview_layout.addWidget(self.attendance_rate_card)
        layout.addLayout(overview_layout)

        layout.addWidget(QLabel("Recent Activity"))
        self.activity_table = QTableWidget()
        self.activity_table.setColumnCount(4)
        self.activity_table.setHorizontalHeaderLabels(["User", "Action", "Date", "Time"])
        self.activity_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.activity_table.setShowGrid(True)
        self.activity_table.setStyleSheet("""
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
        layout.addWidget(self.activity_table)

        self.stack_layout.addWidget(home_widget)

    def setup_users_section(self):
        """Set up the user management section with user table and controls."""
        users_widget = QWidget()
        layout = QVBoxLayout(users_widget)

        controls_layout = QHBoxLayout()
        role_layout = QHBoxLayout()
        role_layout.setAlignment(Qt.AlignCenter)
        role_layout.addWidget(QLabel("Role:"))
        self.role_combo = QComboBox()
        self.role_combo.addItems(["All", "Admin", "Teacher", "Student"])
        self.role_combo.currentTextChanged.connect(self.load_users)
        self.role_combo.setStyleSheet("""
            QComboBox {
                padding: 8px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                font-size: 14px;
                background-color: white;
                min-width: 150px;
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
                selection-background-color: #0d6efd;
                selection-color: white;
            }
        """)
        role_layout.addWidget(self.role_combo)
        controls_layout.addLayout(role_layout)
        controls_layout.addStretch()

        add_user_button = QPushButton(QIcon(qta.icon("fa5s.user-plus", color="white")), "Add User")
        add_user_button.setStyleSheet("""
            QPushButton {
                background-color: #0d6efd;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #0b5ed7;
            }
            QPushButton:pressed {
                background-color: #094c9e;
            }
        """)
        add_user_button.clicked.connect(self.show_add_user_dialog)
        controls_layout.addWidget(add_user_button)
        layout.addLayout(controls_layout)

        self.users_table = QTableWidget()
        self.users_table.setColumnCount(6)
        self.users_table.setHorizontalHeaderLabels(["ID", "Name", "Email", "Role", "Edit", "Delete"])
        self.users_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.users_table.setStyleSheet("""
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
        layout.addWidget(self.users_table)

        self.stack_layout.addWidget(users_widget)

    def setup_classes_section(self):
        """Set up the class management section for creating and enrolling classes."""
        classes_widget = QWidget()
        layout = QVBoxLayout(classes_widget)
        layout.setSpacing(20)

        create_group = QGroupBox("Create Class")
        create_group.setStyleSheet("""
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
        create_layout = QVBoxLayout()
        create_layout.setSpacing(10)

        create_layout.addWidget(QLabel("Class Name:"))
        self.class_name_input = QLineEdit()
        self.class_name_input.setPlaceholderText("Enter class name")
        self.class_name_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                font-size: 14px;
                background-color: white;
                min-width: 200px;
            }
            QLineEdit:focus {
                border: 1px solid #0d6efd;
            }
        """)
        create_layout.addWidget(self.class_name_input)

        create_layout.addWidget(QLabel("Assign Teacher:"))
        self.class_teacher_combo = QComboBox()
        self.class_teacher_combo.setStyleSheet("""
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
        create_layout.addWidget(self.class_teacher_combo)

        create_button = QPushButton("Create Class")
        create_button.setStyleSheet("""
            QPushButton {
                background-color: #0d6efd;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
                min-width: 140px;
            }
            QPushButton:hover {
                background-color: #0b5ed7;
            }
            QPushButton:pressed {
                background-color: #094c9e;
            }
        """)
        create_button.clicked.connect(self.create_class)
        create_layout.addWidget(create_button, alignment=Qt.AlignLeft)
        create_layout.addStretch()
        create_group.setLayout(create_layout)
        layout.addWidget(create_group)

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

        enroll_layout.addWidget(QLabel("Class Name:"))
        self.enroll_class_input = QLineEdit()
        self.enroll_class_input.setPlaceholderText("Type class name")
        self.enroll_section = QLineEdit()
        self.enroll_class_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                font-size: 14px;
                background-color: white;
                min-width: 200px;
            }
            QLineEdit:focus {
                border: 1px solid #0d6efd;
            }
        """)
        enroll_layout.addWidget(self.enroll_class_input)

        enroll_layout.addWidget(QLabel("Select Student:"))
        self.admin_enroll_student_combo = QComboBox()
        self.admin_enroll_student_combo.setStyleSheet("""
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
        enroll_layout.addWidget(self.admin_enroll_student_combo)

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

        classes_table_group = QGroupBox("Current Classes and Enrollments")
        classes_table_group.setStyleSheet("""
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
                color: #495057;
                font-weight: bold;
                font-size: 14px;
            }
        """)
        table_layout = QVBoxLayout()
        self.classes_table = QTableWidget()
        self.classes_table.setColumnCount(5)
        self.classes_table.setHorizontalHeaderLabels(
            ["Class ID", "Class Name", "Teacher", "Students Enrolled", "Remove"])
        self.classes_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.classes_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #ced4da;
                gridline-color: #ced4da;
                margin-top: 10px;
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
        table_layout.addWidget(self.classes_table)
        classes_table_group.setLayout(table_layout)
        layout.addWidget(classes_table_group)

        layout.addStretch()
        self.stack_layout.addWidget(classes_widget)

    def setup_reports_section(self):
        """Set up the reports section with report type selection and table."""
        reports_widget = QWidget()
        layout = QVBoxLayout(reports_widget)

        controls_layout = QHBoxLayout()
        report_type_layout = QHBoxLayout()
        report_type_layout.addWidget(QLabel("Report Type:"))
        self.report_type_combo = QComboBox()
        self.report_type_combo.addItems(["Attendance by Class", "Attendance by Student", "Overall Statistics"])
        self.report_type_combo.currentTextChanged.connect(self.load_report)
        self.report_type_combo.setStyleSheet("""
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
        report_type_layout.addWidget(self.report_type_combo)
        controls_layout.addLayout(report_type_layout)

        date_range_layout = QHBoxLayout()
        date_range_layout.addWidget(QLabel("Date Range:"))
        self.start_date = QDateEdit()
        self.start_date.setDate(QDate.currentDate().addMonths(-1))
        self.start_date.setStyleSheet("""
            QDateEdit {
                padding: 8px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                font-size: 14px;
                background-color: white;
                min-width: 120px;
            }
            QDateEdit:hover {
                border: 1px solid #0d6efd;
            }
        """)
        self.end_date = QDateEdit()
        self.end_date.setDate(QDate.currentDate())
        self.end_date.setStyleSheet("""
            QDateEdit {
                padding: 8px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                font-size: 14px;
                background-color: white;
                min-width: 120px;
            }
            QDateEdit:hover {
                border: 1px solid #0d6efd;
            }
        """)
        date_range_layout.addWidget(self.start_date)
        date_range_layout.addWidget(QLabel("to"))
        date_range_layout.addWidget(self.end_date)
        controls_layout.addLayout(date_range_layout)
        controls_layout.addStretch()

        generate_button = QPushButton("Generate Report")
        generate_button.setIcon(QIcon(qta.icon("fa5s.file-excel", color="white")))
        generate_button.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
                min-width: 160px;
            }
            QPushButton:hover {
                background-color: #138496;
            }
            QPushButton:pressed {
                background-color: #117a8b;
            }
        """)
        generate_button.clicked.connect(self.generate_report)
        controls_layout.addWidget(generate_button)

        layout.addLayout(controls_layout)

        self.report_table = QTableWidget()
        self.report_table.setStyleSheet("""
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
        layout.addWidget(self.report_table)

        export_button = QPushButton("Export Report")
        export_button.setIcon(QIcon(qta.icon("fa5s.file-export", color="white")))
        export_button.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
                min-width: 160px;
            }
            QPushButton:hover {
                background-color: #138496;
            }
            QPushButton:pressed {
                background-color: #117a8b;
            }
        """)
        export_button.clicked.connect(self.export_report)
        layout.addWidget(export_button, alignment=Qt.AlignLeft)

        self.stack_layout.addWidget(reports_widget)

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
                background-color: #BABABA;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 10px;
                font-family: Georgia;
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
            self.load_data()
        elif section_id == "users":
            self.stack_layout.setCurrentIndex(1)
            self.load_users()
        elif section_id == "classes":
            self.stack_layout.setCurrentIndex(2)
            self.load_classes()
        elif section_id == "reports":
            self.stack_layout.setCurrentIndex(3)
            self.load_report()
        elif section_id == "settings":
            self.stack_layout.setCurrentIndex(4)

    def load_data(self):
        """Load initial data for the dashboard."""
        # Update statistics
        users = self.auth_manager.get_all_users()
        total_users = len(users)
        print(f"Total users: {total_users}")
        self.total_users_value.setText(str(total_users))

        # Count classes directly from the database to avoid reliance on get_teacher_classes
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM classes")
        total_classes = cursor.fetchone()[0]
        print(f"Total classes: {total_classes}")
        self.total_classes_value.setText(str(total_classes))
        conn.close()

        # Calculate overall attendance rate
        total_records = 0
        present_records = 0
        for class_id in range(1, total_classes + 1):
            try:
                records = self.db.get_attendance_records(class_id=class_id)
                total_records += len(records)
                present_records += len([r for r in records if r[4] == 'present'])
            except Exception as e:
                print(f"Error fetching records for class {class_id}: {e}")
        attendance_rate = (present_records / total_records * 100) if total_records > 0 else 0
        print(f"Attendance rate: {attendance_rate:.1f}%")
        self.attendance_rate_value.setText(f"{attendance_rate:.1f}%")

        self.load_recent_activity()

    def load_recent_activity(self):
        """Load recent activity logs into the activity table."""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT u.name, a.action, DATE(a.timestamp), TIME(a.timestamp)
                FROM activity_log a
                JOIN users u ON u.id = a.user_id
                ORDER BY a.timestamp DESC
                LIMIT 10
            """)
            activities = cursor.fetchall()
        except sqlite3.Error:
            activities = [
                ("Admin", "Logged in", "2025-03-02", "10:00:00"),
                ("Teacher", "Took attendance", "2025-03-02", "09:30:00"),
            ]
        conn.close()

        self.activity_table.setRowCount(len(activities))
        for i, (user, action, date, time) in enumerate(activities):
            self.activity_table.setItem(i, 0, QTableWidgetItem(user))
            self.activity_table.setItem(i, 1, QTableWidgetItem(action))
            self.activity_table.setItem(i, 2, QTableWidgetItem(date))
            self.activity_table.setItem(i, 3, QTableWidgetItem(time))

    def load_users(self):
        """Load users into the users table based on role filter."""
        role_filter = self.role_combo.currentText()
        users = self.auth_manager.get_all_users(
            role=role_filter.lower() if role_filter != "All" else None
        )
        self.users_table.setRowCount(len(users))
        for i, user in enumerate(users):
            self.users_table.setItem(i, 0, QTableWidgetItem(str(user['id'])))
            self.users_table.setItem(i, 1, QTableWidgetItem(user['name']))
            self.users_table.setItem(i, 2, QTableWidgetItem(user['email']))
            self.users_table.setItem(i, 3, QTableWidgetItem(user['role']))

            edit_button = QPushButton(QIcon(qta.icon("fa5s.edit", color="blue")), "")
            edit_button.clicked.connect(lambda checked, u=user: self.show_edit_user_dialog(u))
            self.users_table.setCellWidget(i, 4, edit_button)

            delete_button = QPushButton(QIcon(qta.icon("fa5s.trash", color="red")), "")
            delete_button.clicked.connect(lambda checked, u=user: self.delete_user(u))
            self.users_table.setCellWidget(i, 5, delete_button)

    def load_classes(self):
        """Load class data and populate teacher and student combo boxes."""
        teachers = self.auth_manager.get_all_users(role='teacher')
        self.class_teacher_combo.clear()
        for teacher in teachers:
            self.class_teacher_combo.addItem(teacher['name'], teacher['id'])
        print("Available teachers:", [(t['id'], t['name']) for t in teachers])  # Debug

        students = self.auth_manager.get_all_users(role='student')
        self.admin_enroll_student_combo.clear()
        for student in students:
            self.admin_enroll_student_combo.addItem(student['name'], student['id'])

        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT c.id, c.name, c.teacher_id, u.name, u.role
            FROM classes c
            LEFT JOIN users u ON u.id = c.teacher_id
        """)
        class_data = cursor.fetchall()
        print("Class data:", class_data)  # Debug: Check fetched data

        self.classes_table.setRowCount(len(class_data))
        for i, (class_id, class_name, teacher_id, teacher_name, teacher_role) in enumerate(class_data):
            self.classes_table.setItem(i, 0, QTableWidgetItem(str(class_id)))
            self.classes_table.setItem(i, 1, QTableWidgetItem(class_name))
            self.classes_table.setItem(i, 2, QTableWidgetItem(teacher_name if teacher_name else "Unassigned"))
            print(
                f"Class {class_id} - Name: {class_name}, Teacher ID: {teacher_id}, Teacher Name: {teacher_name}, Role: {teacher_role}")  # Debug

            cursor.execute("SELECT COUNT(*) FROM class_enrollments WHERE class_id = ?", (class_id,))
            student_count = cursor.fetchone()[0]
            self.classes_table.setItem(i, 3, QTableWidgetItem(str(student_count)))

            remove_button = QPushButton(QIcon(qta.icon("fa5s.trash", color="red")), "")
            remove_button.clicked.connect(lambda checked, cid=class_id: self.remove_class(cid))
            self.classes_table.setCellWidget(i, 4, remove_button)

        conn.close()

    def create_class(self):
        """Create a new class with the specified name and teacher."""
        class_name = self.class_name_input.text().strip()
        teacher_id = self.class_teacher_combo.currentData()
        if not class_name:
            QMessageBox.warning(self, "Error", "Class name cannot be empty")
            return
        if not teacher_id:
            QMessageBox.warning(self, "Error", "Please select a teacher")
            return
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            cursor.execute("INSERT INTO classes (name, teacher_id) VALUES (?, ?)", (class_name, teacher_id))
            conn.commit()
            conn.close()
            QMessageBox.information(self, "Success", f"Class '{class_name}' created with teacher ID {teacher_id}")
            print(f"Created class '{class_name}' with teacher ID {teacher_id}")
            self.class_name_input.clear()
            self.load_classes()
            self.log_activity("Created class")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create class: {str(e)}")

    def enroll_student(self):
        """Enroll a student in a specified class."""
        class_name = self.enroll_class_input.text().strip()
        student_id = self.admin_enroll_student_combo.currentData()
        if not class_name:
            QMessageBox.warning(self, "Error", "Please enter a class name")
            return
        if not student_id:
            QMessageBox.warning(self, "Error", "Please select a student")
            return
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM classes WHERE name = ?", (class_name,))
            class_row = cursor.fetchone()
            if class_row:
                class_id = class_row[0]
            else:
                reply = QMessageBox.question(
                    self,
                    "Class Not Found",
                    f"Class '{class_name}' does not exist. Would you like to create it with no teacher assigned?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply == QMessageBox.Yes:
                    cursor.execute("INSERT INTO classes (name) VALUES (?)", (class_name,))
                    class_id = cursor.lastrowid
                    print(f"Created new class '{class_name}' with ID {class_id} (no teacher assigned)")
                else:
                    conn.close()
                    return

            cursor.execute("INSERT INTO class_enrollments (class_id, student_id) VALUES (?, ?)", (class_id, student_id))
            conn.commit()
            conn.close()
            QMessageBox.information(self, "Success",
                                    f"Student ID {student_id} enrolled in class '{class_name}' (ID: {class_id})")
            print(f"Enrolled student ID {student_id} in class '{class_name}' (ID: {class_id})")
            self.enroll_class_input.clear()
            self.load_classes()
            self.log_activity("Enrolled student")
        except sqlite3.IntegrityError:
            QMessageBox.warning(self, "Error", "Student is already enrolled in this class")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to enroll student: {str(e)}")

    def remove_class(self, class_id):
        """Remove a class and its associated enrollments."""
        reply = QMessageBox.question(
            self,
            "Confirm Removal",
            f"Are you sure you want to remove class ID {class_id}? This will also remove all enrollments.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            try:
                conn = self.db.get_connection()
                cursor = conn.cursor()
                cursor.execute("DELETE FROM class_enrollments WHERE class_id = ?", (class_id,))
                cursor.execute("DELETE FROM attendance_records WHERE class_id = ?", (class_id,))
                cursor.execute("DELETE FROM classes WHERE id = ?", (class_id,))
                conn.commit()
                conn.close()
                QMessageBox.information(self, "Success", f"Class ID {class_id} removed")
                print(f"Removed class ID {class_id}")
                self.load_classes()
                self.log_activity("Removed class")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to remove class: {str(e)}")

    def log_activity(self, action):
        """Log an admin activity to the database."""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS activity_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    action TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            cursor.execute("INSERT INTO activity_log (user_id, action) VALUES (?, ?)", (self.admin_id, action))
            conn.commit()
        except sqlite3.Error as e:
            print(f"Failed to log activity: {e}")
        finally:
            conn.close()

    def show_add_user_dialog(self):
        """Show a dialog for adding a new user."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add User")
        layout = QFormLayout(dialog)

        name_input = QLineEdit()
        name_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                font-size: 14px;
                background-color: white;
                min-width: 200px;
            }
            QLineEdit:focus {
                border: 1px solid #0d6efd;
            }
        """)
        email_input = QLineEdit()
        email_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                font-size: 14px;
                background-color: white;
                min-width: 200px;
            }
            QLineEdit:focus {
                border: 1px solid #0d6efd;
            }
        """)
        password_input = QLineEdit()
        password_input.setEchoMode(QLineEdit.Password)
        password_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                font-size: 14px;
                background-color: white;
                min-width: 200px;
            }
            QLineEdit:focus {
                border: 1px solid #0d6efd;
            }
        """)
        role_combo = QComboBox()
        role_combo.addItems(["admin", "teacher", "student"])
        role_combo.setStyleSheet("""
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

        layout.addRow("Name:", name_input)
        layout.addRow("Email:", email_input)
        layout.addRow("Password:", password_input)
        layout.addRow("Role:", role_combo)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec_() == QDialog.Accepted:
            try:
                self.auth_manager.register(
                    name_input.text(),
                    email_input.text(),
                    password_input.text(),
                    role_combo.currentText(),
                )
                self.load_users()
                QMessageBox.information(self, "Success", "User added successfully")
                self.log_activity("Added user")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to add user: {str(e)}")

    def show_edit_user_dialog(self, user):
        """Show a dialog for editing an existing user."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit User")
        layout = QFormLayout(dialog)

        name_input = QLineEdit(user['name'])
        name_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                font-size: 14px;
                background-color: white;
                min-width: 200px;
            }
            QLineEdit:focus {
                border: 1px solid #0d6efd;
            }
        """)
        email_input = QLineEdit(user['email'])
        email_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                font-size: 14px;
                background-color: white;
                min-width: 200px;
            }
            QLineEdit:focus {
                border: 1px solid #0d6efd;
            }
        """)
        role_combo = QComboBox()
        role_combo.addItems(["admin", "teacher", "student"])
        role_combo.setCurrentText(user['role'])
        role_combo.setStyleSheet("""
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

        layout.addRow("Name:", name_input)
        layout.addRow("Email:", email_input)
        layout.addRow("Role:", role_combo)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec_() == QDialog.Accepted:
            try:
                self.auth_manager.update_user(
                    user['id'],
                    name=name_input.text(),
                    email=email_input.text(),
                    role=role_combo.currentText(),
                )
                self.load_users()
                QMessageBox.information(self, "Success", "User updated successfully")
                self.log_activity("Edited user")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to update user: {str(e)}")

    def delete_user(self, user):
        """Delete a user after confirmation."""
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete user {user['name']}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            try:
                self.auth_manager.delete_user(user['id'])
                self.load_users()
                QMessageBox.information(self, "Success", "User deleted successfully")
                self.log_activity("Deleted user")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete user: {str(e)}")

    def load_report(self):
        """Load the report table structure based on report type."""
        report_type = self.report_type_combo.currentText()
        self.report_table.clear()
        self.report_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.report_table.setShowGrid(True)
        self.report_table.setStyleSheet("""
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

        if report_type == "Attendance by Class":
            self.report_table.setColumnCount(4)
            self.report_table.setHorizontalHeaderLabels(["Class", "Total Records", "Present", "Attendance Rate"])
        elif report_type == "Attendance by Student":
            self.report_table.setColumnCount(4)
            self.report_table.setHorizontalHeaderLabels(["Student", "Total Classes", "Present", "Attendance Rate"])
        else:
            self.report_table.setColumnCount(3)
            self.report_table.setHorizontalHeaderLabels(["Metric", "Value", "Percentage"])

    def generate_report(self):
        """Generate a report based on the selected report type and date range."""
        report_type = self.report_type_combo.currentText()
        start_date = self.start_date.date().toPyDate()
        end_date = self.end_date.date().toPyDate()

        if report_type == "Attendance by Class":
            self.generate_class_report(start_date, end_date)
        elif report_type == "Attendance by Student":
            self.generate_student_report(start_date, end_date)
        else:
            self.generate_overall_report(start_date, end_date)

    def generate_class_report(self, start_date, end_date):
        """Generate an attendance report by class."""
        classes = self.db.get_teacher_classes(None)
        report_data = []

        for class_data in classes:
            records = self.db.get_attendance_records(class_id=class_data[0])
            # Parse the full timestamp with microseconds
            records = [
                r for r in records
                if start_date <= datetime.strptime(r[3], '%Y-%m-%d %H:%M:%S.%f').date() <= end_date
            ]
            total_records = len(records)
            present_records = len([r for r in records if r[4] == 'present'])
            attendance_rate = (present_records / total_records * 100) if total_records > 0 else 0
            report_data.append({
                'class': class_data[1],
                'total': total_records,
                'present': present_records,
                'rate': attendance_rate,
            })

        self.report_table.setRowCount(len(report_data))
        for i, data in enumerate(report_data):
            self.report_table.setItem(i, 0, QTableWidgetItem(data['class']))
            self.report_table.setItem(i, 1, QTableWidgetItem(str(data['total'])))
            self.report_table.setItem(i, 2, QTableWidgetItem(str(data['present'])))
            self.report_table.setItem(i, 3, QTableWidgetItem(f"{data['rate']:.1f}%"))

    def generate_student_report(self, start_date, end_date):
        """
        Generates a report of student attendance records within a specified date range.
        Handles potential time format errors in the 'timestamp' field.

        Args:
            start_date (datetime.date): The start date for the report.
            end_date (datetime.date): The end date for the report.
        """
        students = self.auth_manager.get_all_users(role='student')
        report_data = []

        for student in students:
            records = []
            for class_data in self.db.get_student_classes(student['id']):
                class_records = self.db.get_attendance_records(class_id=class_data[0], student_id=student['id'])
                valid_records = []
                for r in class_records:
                    try:
                        # Attempt to parse the timestamp with the full datetime format
                        record_date = datetime.strptime(r[3], '%Y-%m-%d %H:%M:%S.%f').date()
                    except ValueError:
                        # If the full datetime format fails, try parsing with just the date format
                        try:
                            record_date = datetime.strptime(r[3], '%Y-%m-%d %H:%M:%S').date()
                        except ValueError:
                            # If both datetime formats fail, try parsing with just the date
                            try:
                                record_date = datetime.strptime(r[3], '%Y-%m-%d').date()
                            except ValueError as e:
                                print(f"Error parsing date string: {r[3]} - {e}")
                                continue  # skips the record if the timestamp cannot be parsed
                    if start_date <= record_date <= end_date:
                        valid_records.append(r)

                records.extend(valid_records)

            total_records = len(records)
            present_records = len([r for r in records if r[4] == 'present'])
            attendance_rate = (present_records / total_records * 100) if total_records > 0 else 0
            report_data.append({
                'student': student['name'],
                'total': total_records,
                'present': present_records,
                'rate': attendance_rate,
            })

        self.report_table.setRowCount(len(report_data))
        for i, data in enumerate(report_data):
            self.report_table.setItem(i, 0, QTableWidgetItem(data['student']))
            self.report_table.setItem(i, 1, QTableWidgetItem(str(data['total'])))
            self.report_table.setItem(i, 2, QTableWidgetItem(str(data['present'])))
            self.report_table.setItem(i, 3, QTableWidgetItem(f"{data['rate']:.1f}%"))

    def generate_overall_report(self, start_date, end_date):
        """Generate an overall statistics report."""
        total_students = len(self.auth_manager.get_all_users(role='student'))
        total_teachers = len(self.auth_manager.get_all_users(role='teacher'))
        total_classes = len(self.db.get_teacher_classes(None))

        records = []
        for class_id in range(1, total_classes + 1):
            class_records = self.db.get_attendance_records(class_id=class_id)
            # Parse the full timestamp with microseconds
            records.extend([
                r for r in class_records
                if start_date <= datetime.strptime(r[3], '%Y-%m-%d %H:%M:%S.%f').date() <= end_date
            ])

        total_records = len(records)
        present_records = len([r for r in records if r[4] == 'present'])
        attendance_rate = (present_records / total_records * 100) if total_records > 0 else 0

        report_data = [
            ("Total Students", total_students, "-"),
            ("Total Teachers", total_teachers, "-"),
            ("Total Classes", total_classes, "-"),
            ("Total Attendance Records", total_records, "-"),
            ("Present Records", present_records, f"{attendance_rate:.1f}%"),
        ]

        self.report_table.setRowCount(len(report_data))
        for i, (metric, value, percentage) in enumerate(report_data):
            self.report_table.setItem(i, 0, QTableWidgetItem(metric))
            self.report_table.setItem(i, 1, QTableWidgetItem(str(value)))
            self.report_table.setItem(i, 2, QTableWidgetItem(percentage))

    def export_report(self):
        """Export the current report to a CSV file."""
        try:
            filename, _ = QFileDialog.getSaveFileName(self, "Export Report", "", "CSV Files (*.csv)")
            if filename:
                with open(filename, 'w') as f:
                    headers = [self.report_table.horizontalHeaderItem(col).text() for col in
                               range(self.report_table.columnCount())]
                    f.write(",".join(headers) + "\n")
                    for row in range(self.report_table.rowCount()):
                        row_data = [self.report_table.item(row, col).text() if self.report_table.item(row, col) else ""
                                    for col in range(self.report_table.columnCount())]
                        f.write(",".join(row_data) + "\n")
                QMessageBox.information(self, "Success", "Report exported successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export report: {str(e)}")

    def change_password(self):
        """Change the admin's password."""
        if not all([self.current_password.text(), self.new_password.text(), self.confirm_password.text()]):
            QMessageBox.warning(self, "Error", "Please fill in all password fields")
            return
        if self.new_password.text() != self.confirm_password.text():
            QMessageBox.warning(self, "Error", "New passwords do not match")
            return
        if self.auth_manager.change_password(self.admin_id, self.current_password.text(), self.new_password.text()):
            QMessageBox.information(self, "Success", "Password changed successfully")
            self.current_password.clear()
            self.new_password.clear()
            self.confirm_password.clear()
            self.log_activity("Changed password")
        else:
            QMessageBox.warning(self, "Error", "Failed to change password. Please check your current password.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AdminDashboard(None, {'id': 1, 'name': 'Test Admin'})
    window.show()
    sys.exit(app.exec_())