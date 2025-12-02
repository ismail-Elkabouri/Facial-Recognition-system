import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import qtawesome as qta
from database.db_helper import DatabaseHelper

class BaseDashboard(QMainWindow):
    logout_requested = pyqtSignal()

    def __init__(self, auth_manager, user_data, title_prefix):
        super().__init__()
        self.auth_manager = auth_manager
        self.user_id = user_data['id']
        self.user_name = user_data['name']
        self.db = DatabaseHelper()
        self.title_prefix = title_prefix

        try:
            self.conn = self.db.get_connection()
            if self.conn is None:
                raise Exception("Failed to connect to the database")
        except Exception as e:
            QMessageBox.critical(self, "Database Error", f"Could not connect to the database: {e}")
            sys.exit()

        print(f"{self.__class__.__name__} initialized for {self.user_name} (ID: {self.user_id})")

        self.setup_ui()
        self.load_data()  # Abstract method to be implemented by subclasses

    def setup_ui(self):
        self.setWindowTitle(f"{self.title_prefix} - {self.user_name}")
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

        header_label = QLabel(f"Welcome, {self.user_name}")
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

    def setup_sidebar(self, parent_layout):
        self.sidebar = QWidget()
        self.sidebar.setFixedWidth(250)
        self.sidebar.setStyleSheet("background-color: #E6E6E6;")
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        self.sidebar_button_group = []
        for label_text, icon_name, section_id in self.get_sidebar_buttons():  # Abstract method
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
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        parent_layout.addWidget(scroll_area)

        self.main_content = QWidget()
        self.main_content.setStyleSheet("background-color: #f7f7f7;")
        scroll_area.setWidget(self.main_content)

        self.stack_layout = QStackedLayout(self.main_content)
        self.setup_sections()  # Abstract method to be implemented by subclasses

    def sidebar_button_style(self, is_active):
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
        for button in self.sidebar_button_group:
            is_active = button.property("section_id") == section_id
            button.setStyleSheet(self.sidebar_button_style(is_active))
            button.setChecked(is_active)

        self.handle_section_display(section_id)  # Abstract method

    def setup_settings_section(self):
        settings_widget = QWidget()
        layout = QVBoxLayout(settings_widget)

        password_group = QGroupBox("Change Password")
        password_group.setStyleSheet("""
            QGroupBox { border: 1px solid #ced4da; border-radius: 8px; padding: 10px; background-color: white;}
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; color: black;}
        """)
        password_layout = QFormLayout()

        self.current_password = QLineEdit()
        self.current_password.setEchoMode(QLineEdit.Password)
        self.current_password.setStyleSheet(
            "color: black; padding: 5px; border: 1px solid #ced4da; border-radius: 4px; background-color: white; max-width: 300px; max-height: 20px; margin: 10px;")

        self.new_password = QLineEdit()
        self.new_password.setEchoMode(QLineEdit.Password)
        self.new_password.setStyleSheet(
            "color: black; padding: 5px; border: 1px solid #ced4da; border-radius: 4px; background-color: white; max-width: 300px; max-height: 20px; margin: 10px;")

        self.confirm_password = QLineEdit()
        self.confirm_password.setEchoMode(QLineEdit.Password)
        self.confirm_password.setStyleSheet(
            "color: black; padding: 5px; border: 1px solid #ced4da; border-radius: 4px; background-color: white; max-width: 300px; max-height: 20px; margin: 10px;")

        password_layout.addRow("Current Password:", self.current_password)
        password_layout.addRow("New Password:", self.new_password)
        password_layout.addRow("Confirm Password:", self.confirm_password)

        change_password_button = QPushButton("Change Password")
        change_password_button.setStyleSheet(
            "QPushButton { max-width:150px; margin-left: 150px; background-color: #0d6efd; color: white; padding: 8px 16px; border-radius: 4px; }")
        change_password_button.clicked.connect(self.change_password)
        password_layout.addRow(change_password_button)

        password_group.setLayout(password_layout)
        layout.addWidget(password_group)
        layout.addStretch()
        return settings_widget

    def change_password(self):
        if not all([self.current_password.text(), self.new_password.text(), self.confirm_password.text()]):
            QMessageBox.warning(self, "Error", "Please fill in all password fields")
            return
        if self.new_password.text() != self.confirm_password.text():
            QMessageBox.warning(self, "Error", "New passwords do not match")
            return
        if self.auth_manager.change_password(self.user_id, self.current_password.text(), self.new_password.text()):
            QMessageBox.information(self, "Success", "Password changed successfully")
            self.current_password.clear()
            self.new_password.clear()
            self.confirm_password.clear()
        else:
            QMessageBox.warning(self, "Error", "Failed to change password. Please check your current password.")

    # Abstract methods to be implemented by subclasses
    def get_sidebar_buttons(self):
        raise NotImplementedError("Subclasses must implement get_sidebar_buttons")

    def setup_sections(self):
        raise NotImplementedError("Subclasses must implement setup_sections")

    def handle_section_display(self, section_id):
        raise NotImplementedError("Subclasses must implement handle_section_display")

    def load_data(self):
        raise NotImplementedError("Subclasses must implement load_data")