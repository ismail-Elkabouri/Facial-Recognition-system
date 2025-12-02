import os
import sqlite3
import sys
import time
from typing import Any, Dict, Optional

import cv2
from PyQt5.QtCore import (
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    QRect,
    QSize,
    Qt,
    QTimer,
    pyqtSignal,
)
from PyQt5.QtGui import (
    QColor,
    QFont,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QResizeEvent,
)
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDesktopWidget,
    QDialog,
    QFileDialog,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from auth.auth_manager import AuthManager


def resource_path(relative_path: str) -> str:
    """Get the absolute path to a resource, works for both development and PyInstaller.

    Args:
        relative_path: Relative path to the resource.

    Returns:
        Absolute path to the resource.
    """
    if getattr(sys, 'frozen', False):
        # Running as a PyInstaller executable
        base_path = sys._MEIPASS
    else:
        # Running as a script
        base_path = os.path.abspath('.')
    return os.path.join(base_path, relative_path)


class OutlinedLabel(QLabel):
    """Custom label with text outline effect for improved visibility."""

    def __init__(self, *args, **kwargs):
        """Initialize the outlined label with default outline properties."""
        super().__init__(*args, **kwargs)
        self.outline_color = QColor('#25828A')
        self.outline_width = 2

    def paintEvent(self, event):
        """Draw text with an outline effect.

        Args:
            event: Paint event.
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        font = QFont('Georgia', 50, QFont.Bold)
        painter.setFont(font)
        pen = QPen(self.outline_color)
        pen.setWidth(self.outline_width)
        painter.setPen(pen)
        path = QPainterPath()
        path.addText(0, 0, font, self.text())
        rect = path.boundingRect()
        x = (self.width() - rect.width()) / 2 - rect.x()
        y = (self.height() - rect.height()) / 2 - rect.y()
        path.translate(x, y)
        painter.drawPath(path)


class TransparentDialog(QDialog):
    """Base class for transparent dialogs with custom styling and dragging."""

    def __init__(self, parent=None):
        """Initialize the transparent dialog with a semi-transparent background."""
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(20, 20, 20, 20)

        self.background = QWidget(self)
        self.background.setStyleSheet("""
            background-color: rgba(0, 0, 0, 180);
            border-radius: 10px;
            width: 500px;
            font-family: Georgia;
            padding: 20px;
        """)
        self.layout.addWidget(self.background)

        self.content_layout = QVBoxLayout(self.background)

        self.is_dragging = False
        self.drag_position = QPoint()

    def mousePressEvent(self, event):
        """Handle mouse press for dragging the dialog.

        Args:
            event: Mouse press event.
        """
        if event.button() == Qt.LeftButton:
            self.is_dragging = True
            self.drag_position = event.globalPos() - self.pos()
            event.accept()
        else:
            if not self.geometry().contains(event.globalPos()):
                self.close()
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse movement for dragging the dialog.

        Args:
            event: Mouse move event.
        """
        if self.is_dragging and event.buttons() & Qt.LeftButton:
            self.move(event.globalPos() - self.drag_position)
            event.accept()

    def mouseReleaseEvent(self, event):
        """Handle mouse release to stop dragging.

        Args:
            event: Mouse release event.
        """
        if event.button() == Qt.LeftButton:
            self.is_dragging = False
            event.accept()


class ForgotPasswordDialog(TransparentDialog):
    """Dialog for requesting a password reset via email."""

    def __init__(self, auth_manager: AuthManager, parent=None):
        """Initialize the forgot password dialog.

        Args:
            auth_manager: Authentication manager instance.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.auth_manager = auth_manager
        self.setFixedSize(600, 400)
        self.setWindowTitle('Forgot Password')

        title = QLabel('Forgot Password')
        title.setStyleSheet('color: white; font-size: 25px; font-weight: bold;')
        self.content_layout.addWidget(title, alignment=Qt.AlignCenter)

        self.email = QLineEdit()
        self.email.setPlaceholderText('Enter your email')
        self.email.setStyleSheet("""
            background-color: rgba(0, 0, 0, 0);
            border: 1px solid #1B898C;
            color: white;
            padding: 10px;
            font-size: 16px;
        """)
        self.content_layout.addWidget(self.email)

        self.reset_button = QPushButton('Reset Password')
        self.reset_button.setStyleSheet("""
            background-color: #2A969E;
            border: none;
            border-radius: 5px;
            color: white;
            font-size: 18px;
            margin-left: 200px;
            max-width: 150px;
            padding: 10px;
        """)
        self.reset_button.clicked.connect(self.handle_reset)
        self.content_layout.addWidget(self.reset_button)

        self.progress_label = QLabel('Sending...')
        self.progress_label.setStyleSheet('color: white; font-size: 14px; margin-left: 100px;')
        self.progress_label.hide()
        self.content_layout.addWidget(self.progress_label)

        self.dots = ''
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_progress_dots)

    def update_progress_dots(self) -> None:
        """Animate the dots in the 'Sending...' label."""
        if len(self.dots) < 3:
            self.dots += '.'
        else:
            self.dots = ''
        self.progress_label.setText(f'Sending{self.dots}')

    def handle_reset(self) -> None:
        """Handle the password reset request."""
        email = self.email.text().strip()
        if not email:
            QMessageBox.warning(self, 'Error', 'Please enter your email')
            return

        print(f'Attempting to reset password for email: {email}')
        self.reset_button.setEnabled(False)
        self.progress_label.show()
        QApplication.processEvents()
        self.animation_timer.start(500)

        try:
            time.sleep(2)  # Simulate delay
            success = self.auth_manager.request_password_reset(email)
            self.animation_timer.stop()
            self.progress_label.hide()
            self.reset_button.setEnabled(True)

            if success:
                QMessageBox.information(
                    self,
                    'Success',
                    f'A password reset token has been sent to {email}. '
                    'Check your inbox and use it in the "Reset Password" option.'
                )
                parent = self.parent()
                if parent and hasattr(parent, 'current_dialog') and isinstance(parent.current_dialog, LoginDialog):
                    parent.close_current_dialog()
                self.close()
            else:
                error_msg = getattr(self.auth_manager, 'last_email_error', 'Unknown error')
                token = getattr(self.auth_manager, 'last_reset_token', None)
                expiry = getattr(self.auth_manager, 'last_reset_expiry', None)

                if token and expiry:
                    QMessageBox.warning(
                        self,
                        'Email Failed',
                        f'Failed to send the reset email.\nDetails: {error_msg}\n\n'
                        f'Use this token to reset your password:\n{token}\n\nValid until: {expiry}'
                    )
                else:
                    QMessageBox.warning(
                        self,
                        'Error',
                        f'No account found with that email or failed to send the reset email.\n'
                        f'Details: {error_msg}\nPlease check your network connection and try again.'
                    )
        except Exception as e:
            self.animation_timer.stop()
            self.progress_label.hide()
            self.reset_button.setEnabled(True)
            print(f'Reset password failed: {str(e)}')
            QMessageBox.critical(self, 'Error', f'Failed to request password reset: {str(e)}')


class ResetPasswordDialog(TransparentDialog):
    """Dialog for resetting a password using a token."""

    def __init__(self, auth_manager: AuthManager, parent=None):
        """Initialize the reset password dialog.

        Args:
            auth_manager: Authentication manager instance.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.auth_manager = auth_manager
        self.setFixedSize(600, 400)
        self.setWindowTitle('Reset Password')

        title = QLabel('Reset Password')
        title.setStyleSheet('color: white; font-size: 25px; font-weight: bold;')
        self.content_layout.addWidget(title, alignment=Qt.AlignCenter)

        self.token = QLineEdit()
        self.token.setPlaceholderText('Enter reset token from email')
        self.token.setStyleSheet("""
            background-color: rgba(0, 0, 0, 0);
            border: 1px solid #1B898C;
            color: white;
            padding: 10px;
            font-size: 16px;
        """)
        self.content_layout.addWidget(self.token)

        self.new_password = QLineEdit()
        self.new_password.setPlaceholderText('Enter new password')
        self.new_password.setEchoMode(QLineEdit.Password)
        self.new_password.setStyleSheet("""
            background-color: rgba(0, 0, 0, 0);
            border: 1px solid #1B898C;
            color: white;
            padding: 10px;
            font-size: 16px;
        """)
        self.content_layout.addWidget(self.new_password)

        self.confirm_password = QLineEdit()
        self.confirm_password.setPlaceholderText('Confirm new password')
        self.confirm_password.setEchoMode(QLineEdit.Password)
        self.confirm_password.setStyleSheet("""
            background-color: rgba(0, 0, 0, 0);
            border: 1px solid #1B898C;
            color: white;
            padding: 10px;
            font-size: 16px;
        """)
        self.content_layout.addWidget(self.confirm_password)

        self.reset_button = QPushButton('Reset Password')
        self.reset_button.setStyleSheet("""
            background-color: #2A969E;
            border: none;
            border-radius: 5px;
            color: white;
            font-size: 18px;
            margin-left: 200px;
            max-width: 150px;
            padding: 10px;
        """)
        self.reset_button.clicked.connect(self.handle_reset)
        self.content_layout.addWidget(self.reset_button)

    def handle_reset(self) -> None:
        """Handle password reset with the provided token."""
        token = self.token.text().strip()
        new_password = self.new_password.text().strip()
        confirm_password = self.confirm_password.text().strip()

        if not all([token, new_password, confirm_password]):
            QMessageBox.warning(self, 'Error', 'Please fill in all fields')
            return
        if new_password != confirm_password:
            QMessageBox.warning(self, 'Error', 'Passwords do not match')
            return

        success = self.auth_manager.reset_password(token, new_password)
        if success:
            QMessageBox.information(
                self,
                'Success',
                'Password reset successfully. Please login with your new password.'
            )
            self.close()
        else:
            QMessageBox.warning(self, 'Error', 'Invalid or expired reset token.')


class LoginDialog(TransparentDialog):
    """Dialog for user login with email and password."""

    def __init__(self, auth_manager: AuthManager, parent=None):
        """Initialize the login dialog.

        Args:
            auth_manager: Authentication manager instance.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.auth_manager = auth_manager
        self.parent = parent
        self.setFixedSize(600, 600)

        title = QLabel('Login')
        title.setStyleSheet('color: white; font-size: 25px; font-weight: bold;')
        self.content_layout.addWidget(title, alignment=Qt.AlignCenter)

        self.username = QLineEdit()
        self.username.setPlaceholderText('Email')
        self.username.setStyleSheet("""
            background-color: rgba(0, 0, 0, 0);
            border: 1px solid #1B898C;
            color: white;
            padding: 10px;
            font-size: 16px;
        """)
        self.content_layout.addWidget(self.username)

        self.password = QLineEdit()
        self.password.setPlaceholderText('Password')
        self.password.setEchoMode(QLineEdit.Password)
        self.password.setStyleSheet("""
            background-color: rgba(0, 0, 0, 0);
            border: 1px solid #1B898C;
            color: white;
            padding: 10px;
            font-size: 16px;
        """)
        self.content_layout.addWidget(self.password)

        self.login_button = QPushButton('Login')
        self.login_button.setStyleSheet("""
            background-color: #2A969E;
            border: none;
            border-radius: 5px;
            color: white;
            font-size: 18px;
            margin-left: 200px;
            max-width: 100px;
            padding: 10px;
        """)
        self.login_button.clicked.connect(self.handle_login)
        self.content_layout.addWidget(self.login_button)

        self.password.returnPressed.connect(self.handle_login)

        self.forgot_password = QPushButton('Forgot Password?')
        self.forgot_password.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 0);
                border: none;
                color: #2A969E;
                font-size: 14px;
                margin-left: 100px;
                padding: 5px;
                text-decoration: underline;
            }
            QPushButton:hover {
                color: #1B898C;
            }
        """)
        self.forgot_password.clicked.connect(self.show_forgot_password_dialog)
        self.content_layout.addWidget(self.forgot_password)

    def handle_login(self) -> None:
        """Handle the login attempt with email and password."""
        email = self.username.text().strip()
        password = self.password.text().strip()

        if not email or not password:
            QMessageBox.warning(self, 'Error', 'Please enter both email and password')
            return

        result = self.auth_manager.login(email, password)
        if result:
            print(f'Login successful: {result}')
            self.parent.handle_login_success(result)
            self.close()
        else:
            QMessageBox.warning(self, 'Error', 'Invalid email or password')
            self.password.clear()

    def show_forgot_password_dialog(self) -> None:
        """Show the forgot password dialog."""
        dialog = ForgotPasswordDialog(self.auth_manager, self)
        self.parent.show_dialog_centered(dialog)
        dialog.exec_()


class RegisterDialog(TransparentDialog):
    """Dialog for registering a new user."""

    def __init__(self, auth_manager: AuthManager, parent=None):
        """Initialize the registration dialog.

        Args:
            auth_manager: Authentication manager instance.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.auth_manager = auth_manager
        self.setFixedSize(600, 600)

        title = QLabel('Register')
        title.setStyleSheet('color: white; font-size: 28px; font-weight: bold;')
        self.content_layout.addWidget(title, alignment=Qt.AlignCenter)

        self.username = QLineEdit()
        self.username.setPlaceholderText('Name')
        self.username.setStyleSheet("""
            background-color: rgba(0, 0, 0, 0);
            border: 1px solid #1B898C;
            color: white;
            padding: 10px;
            font-size: 16px;
        """)
        self.content_layout.addWidget(self.username)

        self.email = QLineEdit()
        self.email.setPlaceholderText('Email')
        self.email.setStyleSheet("""
            background-color: rgba(0, 0, 0, 0);
            border: 1px solid #1B898C;
            color: white;
            padding: 10px;
            font-size: 16px;
        """)
        self.content_layout.addWidget(self.email)

        self.password = QLineEdit()
        self.password.setPlaceholderText('Password')
        self.password.setEchoMode(QLineEdit.Password)
        self.password.setStyleSheet("""
            background-color: rgba(0, 0, 0, 0);
            border: 1px solid #1B898C;
            color: white;
            padding: 10px;
            font-size: 16px;
        """)
        self.content_layout.addWidget(self.password)

        self.role_combo = QComboBox()
        self.role_combo.addItems(['student', 'teacher'])
        self.role_combo.setStyleSheet("""
            QComboBox {
                background-color: rgba(0, 0, 0, 180);
                border: 1px solid #1B898C;
                color: white;
                font-size: 16px;
                padding: 5px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
            }
            QComboBox QAbstractItemView {
                background-color: rgba(0, 0, 0, 180);
                color: white;
                selection-background-color: #1B898C;
            }
        """)
        self.content_layout.addWidget(self.role_combo)

        self.register_button = QPushButton('Register')
        self.register_button.setStyleSheet("""
            background-color: #2A969E;
            border: none;
 AUDIO
            border-radius: 5px;
            color: white;
            font-size: 18px;
            margin-left: 200px;
            max-width: 100px;
            padding: 10px;
        """)
        self.register_button.clicked.connect(self.handle_register)
        self.content_layout.addWidget(self.register_button)

    def handle_register(self) -> None:
        """Handle the registration attempt."""
        name = self.username.text().strip()
        email = self.email.text().strip()
        password = self.password.text().strip()
        role = self.role_combo.currentText()

        if not all([name, email, password]):
            QMessageBox.warning(self, 'Error', 'Please fill in all fields')
            return

        try:
            user = self.auth_manager.register(name, email, password, role)
            if user:
                QMessageBox.information(self, 'Success', 'Registration successful! You can now login.')
                self.close()
            else:
                raise RuntimeError('Registration failed due to an unknown error')
        except ValueError as e:
            QMessageBox.warning(self, 'Error', str(e))
        except sqlite3.IntegrityError:
            QMessageBox.warning(self, 'Error', 'Email already exists. Please use a different email.')
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'Registration failed: {str(e)}')


class LoginPage(QMainWindow):
    """Main login page for the facial recognition attendance system."""

    login_successful = pyqtSignal(dict)

    def __init__(self):
        """Initialize the login page with video background and authentication."""
        super().__init__()
        self.auth_manager = AuthManager()
        self.original_size = None
        self.setWindowTitle('AttendFace')
        self.setGeometry(400, 300, 1800, 1024)

        screen = QDesktopWidget().availableGeometry(self)
        self.setMaximumSize(screen.width(), screen.height())

        self.current_dialog = None

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QGridLayout(self.central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.video_label, 0, 0)

        self.welcome_label = OutlinedLabel('Welcome To AttendFace', self)
        self.welcome_label.setAlignment(Qt.AlignCenter)
        self.welcome_label.setStyleSheet("""
            background-color: rgba(0, 0, 0, 50);
            border-radius: 10px;
            color: #25828A;
            margin-top: 100px;
            padding: 20px;
        """)
        self.welcome_label.setFixedSize(1500, 200)

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 200))
        shadow.setOffset(2, 2)
        self.welcome_label.setGraphicsEffect(shadow)

        self.layout.addWidget(self.welcome_label, 0, 0, Qt.AlignTop | Qt.AlignCenter)

        self.buttons_widget = QWidget(self)
        self.buttons_layout = QHBoxLayout(self.buttons_widget)
        self.buttons_layout.setAlignment(Qt.AlignCenter)

        self.login_button = QPushButton('Login', self)
        self.login_button.setFixedSize(200, 60)
        self.login_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 0);
                border: 2px solid #2A969E;
                border-radius: 5px;
                color: white;
                font-family: Georgia;
                font-size: 16px;
                margin-right: 20px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: rgba(42, 150, 158, 150);
                border: 2px solid #FFFFFF;
            }
        """)

        self.register_button = QPushButton('Register', self)
        self.register_button.setFixedSize(200, 60)
        self.register_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 0);
                border: 2px solid #2A969E;
                border-radius: 5px;
                color: white;
                font-family: Georgia;
                font-size: 16px;
                margin-left: 20px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: rgba(42, 150, 158, 150);
                border: 2px solid #FFFFFF;
            }
        """)

        self.reset_password_button = QPushButton('Reset Password', self)
        self.reset_password_button.setFixedSize(200, 60)
        self.reset_password_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 0);
                border: 2px solid #2A969E;
                border-radius: 5px;
                color: white;
                font-family: Georgia;
                font-size: 16px;
                margin-left: 20px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: rgba(42, 150, 158, 150);
                border: 2px solid #FFFFFF;
            }
        """)

        self.buttons_layout.addWidget(self.login_button)
        self.buttons_layout.addWidget(self.register_button)
        self.buttons_layout.addWidget(self.reset_password_button)

        self.layout.addWidget(self.buttons_widget, 0, 0, Qt.AlignBottom | Qt.AlignCenter)

        self._setup_background_video()

        self.login_button.clicked.connect(self.show_login_dialog)
        self.register_button.clicked.connect(self.show_register_dialog)
        self.reset_password_button.clicked.connect(self.show_reset_password_dialog)

    def _setup_background_video(self) -> None:
        """Set up the background video player."""
        video_path = resource_path('assets/background_video.mp4')
        print(f'Video path: {video_path}')  # Debug print

        if not os.path.exists(video_path):
            print(f'Error: Video file not found at {video_path}')

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f'Error: Could not open video file at {video_path}')
        else:
            print('Video file opened successfully')

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~33 fps

    def update_frame(self) -> None:
        """Update the video frame for the background animation."""
        ret, frame = self.cap.read()
        if not ret:
            print('Video ended or failed to read frame; looping back to start')
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w

        convert_to_Qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        self.video_label.setPixmap(QPixmap.fromImage(convert_to_Qt_format).scaled(
            self.width(), self.height(), Qt.KeepAspectRatioByExpanding
        ))

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Handle window resize events to adjust the welcome label.

        Args:
            event: Resize event.
        """
        super().resizeEvent(event)
        new_width = min(self.width() - 100, 1500)
        self.welcome_label.setFixedWidth(new_width)

        self.welcome_label.move(
            (self.width() - self.welcome_label.width()) // 2,
            self.welcome_label.y()
        )

    def closeEvent(self, event) -> None:
        """Handle window close events to clean up resources.

        Args:
            event: Close event.
        """
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        if hasattr(self, 'timer') and self.timer is not None:
            self.timer.stop()
        event.accept()

    def show_login_dialog(self) -> None:
        """Show the login dialog."""
        self.close_current_dialog()
        dialog = LoginDialog(self.auth_manager, self)
        self.show_dialog_centered(dialog)

    def show_register_dialog(self) -> None:
        """Show the registration dialog."""
        self.close_current_dialog()
        dialog = RegisterDialog(self.auth_manager, self)
        self.show_dialog_centered(dialog)

    def show_reset_password_dialog(self) -> None:
        """Show the password reset dialog."""
        self.close_current_dialog()
        dialog = ResetPasswordDialog(self.auth_manager, self)
        self.show_dialog_centered(dialog)

    def show_dialog_centered(self, dialog: QDialog) -> None:
        """Show a dialog centered in the main window.

        Args:
            dialog: Dialog to show.
        """
        main_window_rect = self.geometry()
        dialog_rect = dialog.geometry()
        center_x = main_window_rect.x() + (main_window_rect.width() - dialog_rect.width()) // 2
        center_y = main_window_rect.y() + (main_window_rect.height() - dialog_rect.height()) // 2
        dialog.move(center_x, center_y)
        dialog.show()
        self.current_dialog = dialog

    def close_current_dialog(self) -> None:
        """Close the currently open dialog if any."""
        if self.current_dialog:
            self.current_dialog.close()
            self.current_dialog = None

    def handle_login_success(self, user_data: Dict[str, Any]) -> None:
        """Handle successful login by emitting a signal.

        Args:
            user_data: User data from successful login.
        """
        print(f'Login page handling success: {user_data}')
        self.login_successful.emit(user_data)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LoginPage()
    window.show()
    sys.exit(app.exec_())