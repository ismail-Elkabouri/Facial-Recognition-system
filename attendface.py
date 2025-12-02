import os
import sys

import qtawesome as qta
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import QApplication, QMessageBox

from auth.auth_manager import AuthManager
from database.db_helper import DatabaseHelper
from UI.admin_interface import AdminDashboard
from UI.login_page import LoginPage
from UI.student_interface import StudentDashboard
from UI.teacher_interface import TeacherDashboard


def setup_environment() -> bool:
    """Initialize the application environment by creating directories and database.

    Returns:
        True if setup is successful, False otherwise.
    """
    try:
        # Create necessary directories
        directories = [
            'database',
            'data/dataset',
            'data/training_data',
            'models',
            'logs',
            'assets',
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

        # Initialize database
        db = DatabaseHelper()

        return True

    except Exception as e:
        QMessageBox.critical(None, 'Setup Error', f'Failed to setup environment: {str(e)}')
        return False


class Application:
    """Main application class for the facial recognition attendance system."""

    def __init__(self):
        """Initialize the application with global styles and authentication."""
        self.app = QApplication(sys.argv)
        # Set global font
        font = QFont('Roboto', 14)
        self.app.setFont(font)
        # Set global stylesheet with background image and professional colors
        self.app.setStyleSheet("""
            * {
                font-family: 'Roboto';
                font-size: 14px;
                color: #2C2C2C;
            }
            QLabel {
                background-color: rgba(245, 246, 250, 0.8);
            }
            QPushButton {
                background-color: #3498DB;
                color: white;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
            QComboBox {
                background-color: rgba(245, 246, 250, 0.9);
                border: 1px solid #2C3E50;
                border-radius: 4px;
                padding: 5px;
            }
            QComboBox::drop-down {
                border-left: 1px solid #2C3E50;
            }
            QTableWidget {
                background-color: rgba(245, 246, 250, 0.95);
                alternate-background-color: #E8ECEF;
            }
        """)
        self.auth = AuthManager()
        self.current_window = None

        # Set application style and icon
        self.app.setStyle('Fusion')
        self.app.setWindowIcon(QIcon(qta.icon('fa5s.user-graduate', color='#3498DB').pixmap(64, 64)))

    def start(self) -> int:
        """Start the application and event loop.

        Returns:
            Exit code of the application.
        """
        try:
            # Setup environment
            if not setup_environment():
                return 1

            # Show login window
            self.show_login()

            # Start event loop
            return self.app.exec_()

        except Exception as e:
            QMessageBox.critical(None, 'Error', f'Application error: {str(e)}')
            return 1

    def show_login(self) -> None:
        """Show the login window."""
        try:
            # Close current window if exists
            if self.current_window:
                self.current_window.close()

            # Create and show login window
            login_window = LoginPage()
            login_window.login_successful.connect(self.handle_login)
            login_window.show()

            # Store reference
            self.current_window = login_window

        except Exception as e:
            QMessageBox.critical(None, 'Error', f'Failed to show login window: {str(e)}')

    def handle_login(self, user_data: dict) -> None:
        """Handle successful login and show the appropriate dashboard.

        Args:
            user_data: Dictionary containing user information (name, role, etc.).
        """
        try:
            print(f"Login successful for {user_data['name']} ({user_data['role']})")

            # Close login window
            if self.current_window:
                self.current_window.close()

            # Create appropriate dashboard based on user role
            if user_data['role'] == 'admin':
                dashboard = AdminDashboard(self.auth, user_data)
            elif user_data['role'] == 'teacher':
                dashboard = TeacherDashboard(self.auth, user_data)
            elif user_data['role'] == 'student':
                dashboard = StudentDashboard(self.auth, user_data)
            else:
                raise ValueError(f"Invalid role: {user_data['role']}")

            # Connect logout signal
            dashboard.logout_requested.connect(self.show_login)

            # Show dashboard
            dashboard.show()

            # Store reference
            self.current_window = dashboard

        except Exception as e:
            print(f"Dashboard creation error: {str(e)}")
            QMessageBox.critical(None, 'Error', f'Failed to create dashboard: {str(e)}')
            self.show_login()


def main() -> int:
    """Main entry point for the application.

    Returns:
        Exit code of the application.
    """
    app = Application()
    return app.start()


if __name__ == '__main__':
    sys.exit(main())