import hashlib
import logging
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from database.db_helper import DatabaseHelper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(f'auth_manager_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger('AuthManager')


class AuthManager:
    """Manages user authentication, registration, and password reset functionality."""

    def __init__(self, db_path: str = 'database/database.db'):
        """Initialize the authentication manager with a database path.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db = DatabaseHelper(db_path)
        self._ensure_admin_exists()
        self._ensure_reset_token_column()

    def _hash_password(self, password: str) -> str:
        """Hash a password using SHA-256.

        Args:
            password: Plain-text password to hash.

        Returns:
            Hexadecimal string of the hashed password.
        """
        return hashlib.sha256(password.encode()).hexdigest()

    def _ensure_admin_exists(self) -> None:
        """Ensure at least one admin user exists."""
        admin = self.db.get_all_users(role='admin')
        if not admin:
            self.register(
                name='Admin',
                email='admin@gmail.com',
                password='admin',  # This should be changed on first login
                role='admin',
            )

    def _ensure_reset_token_column(self) -> None:
        """Ensure the reset_token and reset_token_expiry columns exist in the users table."""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('PRAGMA table_info(users)')
            columns = [col[1] for col in cursor.fetchall()]
            if 'reset_token' not in columns:
                cursor.execute('ALTER TABLE users ADD COLUMN reset_token TEXT')
            if 'reset_token_expiry' not in columns:
                cursor.execute('ALTER TABLE users ADD COLUMN reset_token_expiry TEXT')
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f'Error ensuring reset_token column: {e}')
        finally:
            conn.close()

    def register(self, name: str, email: str, password: str, role: str) -> Optional[Dict[str, Any]]:
        """Register a new user.

        Args:
            name: User's full name.
            email: User's email address.
            password: User's plain-text password.
            role: User's role (admin, teacher, student).

        Returns:
            Dictionary with user details if registration succeeds, None otherwise.
        """
        if not all([name, email, password, role]):
            raise ValueError('All fields are required')

        if role not in ['admin', 'teacher', 'student']:
            raise ValueError('Invalid role')

        hashed_password = self._hash_password(password)
        user_id = self.db.create_user(name, email, hashed_password, role)

        if user_id:
            return {
                'id': user_id,
                'name': name,
                'email': email,
                'role': role,
            }
        return None

    def login(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate a user.

        Args:
            email: User's email address.
            password: User's plain-text password.

        Returns:
            Dictionary with user details if authentication succeeds, None otherwise.
        """
        if not email or not password:
            return None

        hashed_password = self._hash_password(password)
        user = self.db.get_user(email, hashed_password)

        if user:
            return {
                'id': user[0],
                'name': user[1],
                'email': user[2],
                'role': user[4],
            }
        return None

    def get_user_info(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user information by ID.

        Args:
            user_id: ID of the user.

        Returns:
            Dictionary with user details if found, None otherwise.
        """
        user = self.db.get_user_by_id(user_id)
        if user:
            return {
                'id': user[0],
                'name': user[1],
                'email': user[2],
                'role': user[4],
            }
        return None

    def get_all_users(self, role: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all users, optionally filtered by role.

        Args:
            role: Role to filter users by (admin, teacher, student), or None for all users.

        Returns:
            List of dictionaries with user details.
        """
        users = self.db.get_all_users(role)
        return [
            {
                'id': user[0],
                'name': user[1],
                'email': user[2],
                'role': user[4],
            }
            for user in users
        ]

    def update_user(self, user_id: int, **kwargs: Any) -> bool:
        """Update user information.

        Args:
            user_id: ID of the user to update.
            **kwargs: Key-value pairs of fields to update (e.g., name, email, password, role).

        Returns:
            True if the update was successful, False otherwise.
        """
        if 'password' in kwargs:
            kwargs['password'] = self._hash_password(kwargs['password'])
        return self.db.update_user(user_id, **kwargs)

    def delete_user(self, user_id: int) -> bool:
        """Delete a user.

        Args:
            user_id: ID of the user to delete.

        Returns:
            True if deletion was successful, False otherwise.
        """
        return self.db.delete_user(user_id)

    def change_password(self, user_id: int, current_password: str, new_password: str) -> bool:
        """Change user password.

        Args:
            user_id: ID of the user.
            current_password: Current plain-text password.
            new_password: New plain-text password.

        Returns:
            True if the password change was successful, False otherwise.
        """
        user = self.db.get_user_by_id(user_id)
        if not user:
            return False

        if self._hash_password(current_password) != user[3]:
            return False

        hashed_new_password = self._hash_password(new_password)
        return self.db.update_user(user_id, password=hashed_new_password)

    def verify_password(self, user_id: int, password: str) -> bool:
        """Verify user password.

        Args:
            user_id: ID of the user.
            password: Plain-text password to verify.

        Returns:
            True if the password is correct, False otherwise.
        """
        user = self.db.get_user_by_id(user_id)
        if not user:
            return False
        return self._hash_password(password) == user[3]

    def request_password_reset(self, email: str) -> bool:
        """Request a password reset for the given email.

        Args:
            email: User's email address.

        Returns:
            True if the reset request was successful, False otherwise.
        """
        conn = self.db.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('SELECT id, email FROM users WHERE email = ?', (email,))
            user = cursor.fetchone()
            if not user:
                return False

            user_id = user[0]
            reset_token = str(uuid.uuid4())
            expiry = (datetime.now() + timedelta(hours=1)).isoformat()

            cursor.execute(
                'UPDATE users SET reset_token = ?, reset_token_expiry = ? WHERE id = ?',
                (reset_token, expiry, user_id),
            )
            conn.commit()

            logger.info(f'Password reset token {reset_token} generated for {email}. Expires at {expiry}.')
            logger.info(f'Send this link to the user: http://yourdomain.com/reset?token={reset_token}')

            return True
        except sqlite3.Error as e:
            logger.error(f'Error requesting password reset: {e}')
            return False
        finally:
            conn.close()

    def verify_reset_token(self, token: str) -> Optional[int]:
        """Verify if the reset token is valid and not expired.

        Args:
            token: Password reset token to verify.

        Returns:
            User ID if the token is valid, None otherwise.
        """
        conn = self.db.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                'SELECT id, reset_token_expiry FROM users WHERE reset_token = ?',
                (token,),
            )
            user = cursor.fetchone()
            if not user:
                return None

            expiry = datetime.fromisoformat(user[1])
            if datetime.now() > expiry:
                return None

            return user[0]
        except sqlite3.Error as e:
            logger.error(f'Error verifying reset token: {e}')
            return None
        finally:
            conn.close()

    def reset_password(self, token: str, new_password: str) -> bool:
        """Reset the password using a valid token.

        Args:
            token: Password reset token.
            new_password: New plain-text password.

        Returns:
            True if the password reset was successful, False otherwise.
        """
        user_id = self.verify_reset_token(token)
        if not user_id:
            return False

        hashed_new_password = self._hash_password(new_password)
        success = self.db.update_user(user_id, password=hashed_new_password, reset_token=None, reset_token_expiry=None)
        return success