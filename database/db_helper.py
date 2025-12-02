import logging
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(f'db_helper_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger('DatabaseHelper')


class DatabaseHelper:
    """Helper class for managing SQLite database operations for the attendance system."""

    def __init__(self, db_path: str = 'database/database.db'):
        """Initialize the database helper with the database path.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self.init_database()

    def get_connection(self) -> sqlite3.Connection:
        """Create and return a database connection.

        Returns:
            SQLite database connection object.
        """
        return sqlite3.connect(self.db_path)

    def init_database(self) -> None:
        """Initialize the database with required tables from schema.sql."""
        try:
            conn = self.get_connection()
            with open('database/schema.sql', 'r') as f:
                conn.executescript(f.read())
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f'Error initializing database: {e}')
            raise

    def create_user(self, name: str, email: str, password: str, role: str) -> Optional[int]:
        """Create a new user.

        Args:
            name: User's full name.
            email: User's email address.
            password: User's password.
            role: User's role (e.g., admin, teacher, student).

        Returns:
            ID of the created user, or None if creation fails (e.g., duplicate email).
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO users (name, email, password, role) VALUES (?, ?, ?, ?)',
                (name, email, password, role),
            )
            conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            return None
        finally:
            conn.close()

    def get_user(self, email: str, password: str) -> Optional[Tuple]:
        """Get user by email and password.

        Args:
            email: User's email address.
            password: User's password.

        Returns:
            User record as a tuple, or None if no match is found.
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM users WHERE email = ? AND password = ?',
                (email, password),
            )
            return cursor.fetchone()
        finally:
            conn.close()

    def get_user_by_id(self, user_id: int) -> Optional[Tuple]:
        """Get user by ID.

        Args:
            user_id: ID of the user.

        Returns:
            User record as a tuple, or None if no match is found.
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
            return cursor.fetchone()
        finally:
            conn.close()

    def get_all_users(self, role: Optional[str] = None) -> List[Tuple]:
        """Get all users, optionally filtered by role.

        Args:
            role: Role to filter users by (e.g., admin, teacher, student), or None for all users.

        Returns:
            List of user records as tuples.
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            if role:
                cursor.execute('SELECT * FROM users WHERE role = ?', (role,))
            else:
                cursor.execute('SELECT * FROM users')
            return cursor.fetchall()
        finally:
            conn.close()

    def update_user(self, user_id: int, **kwargs: Any) -> bool:
        """Update user information.

        Args:
            user_id: ID of the user to update.
            **kwargs: Key-value pairs of fields to update (e.g., name, email, password, role).

        Returns:
            True if the update was successful, False otherwise.
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            updates = ', '.join(f'{key} = ?' for key in kwargs.keys())
            values = list(kwargs.values()) + [user_id]
            cursor.execute(f'UPDATE users SET {updates} WHERE id = ?', values)
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def delete_user(self, user_id: int) -> bool:
        """Delete a user.

        Args:
            user_id: ID of the user to delete.

        Returns:
            True if the deletion was successful, False otherwise.
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def create_class(self, name: str, teacher_id: int) -> Optional[int]:
        """Create a new class.

        Args:
            name: Name of the class.
            teacher_id: ID of the teacher assigned to the class.

        Returns:
            ID of the created class, or None if creation fails.
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO classes (name, teacher_id) VALUES (?, ?)',
                (name, teacher_id),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_teacher_classes(self, teacher_id: int) -> List[Tuple]:
        """Get all classes for a teacher.

        Args:
            teacher_id: ID of the teacher.

        Returns:
            List of class records as tuples.
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM classes WHERE teacher_id = ?',
                (teacher_id,),
            )
            return cursor.fetchall()
        finally:
            conn.close()

    def get_student_classes(self, student_id: int) -> List[Tuple]:
        """Get all classes for a student.

        Args:
            student_id: ID of the student.

        Returns:
            List of class records as tuples (id, name, teacher_id).
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                '''
                SELECT c.id, c.name, c.teacher_id
                FROM classes c
                JOIN class_enrollments ce ON c.id = ce.class_id
                WHERE ce.student_id = ?
                ''',
                (student_id,),
            )
            result = cursor.fetchall()
            logger.debug(f'get_student_classes({student_id}) returned: {result}')
            return result
        finally:
            conn.close()

    def enroll_student(self, class_id: int, student_id: int) -> bool:
        """Enroll a student in a class.

        Args:
            class_id: ID of the class.
            student_id: ID of the student.

        Returns:
            True if enrollment was successful, False if it fails (e.g., duplicate enrollment).
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO class_enrollments (class_id, student_id) VALUES (?, ?)',
                (class_id, student_id),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()

    def delete_class(self, class_id: int) -> bool:
        """Delete a class and its related records.

        Args:
            class_id: ID of the class to delete.

        Returns:
            True if deletion was successful, False otherwise.
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM attendance_records WHERE class_id = ?', (class_id,))
            cursor.execute('DELETE FROM class_enrollments WHERE class_id = ?', (class_id,))
            cursor.execute('DELETE FROM classes WHERE id = ?', (class_id,))
            conn.commit()
            return True
        except Exception as e:
            logger.error(f'Error deleting class: {e}')
            return False
        finally:
            conn.close()

    def record_attendance(self, class_id: int, student_id: int, date: str, status: str) -> bool:
        """Record attendance for a student.

        Args:
            class_id: ID of the class.
            student_id: ID of the student.
            date: Date of the attendance record (YYYY-MM-DD).
            status: Attendance status (e.g., present, absent).

        Returns:
            True if recording was successful, False otherwise.
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                '''
                INSERT INTO attendance_records (class_id, student_id, date, status)
                VALUES (?, ?, ?, ?)
                ''',
                (class_id, student_id, date, status),
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def get_attendance_records(
        self, class_id: Optional[int] = None, student_id: Optional[int] = None, date: Optional[str] = None
    ) -> List[Tuple]:
        """Get attendance records with optional filters, handling date as a range if provided.

        Args:
            class_id: ID of the class to filter by, or None for all classes.
            student_id: ID of the student to filter by, or None for all students.
            date: Date to filter by (YYYY-MM-DD), or None for all dates. Treated as a full-day range.

        Returns:
            List of attendance records as tuples.
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            query = 'SELECT * FROM attendance_records WHERE 1=1'
            params = []

            if class_id:
                query += ' AND class_id = ?'
                params.append(class_id)
            if student_id:
                query += ' AND student_id = ?'
                params.append(student_id)
            if date:
                query += ' AND date >= ? AND date < ?'
                params.append(date)
                params.append(f'{date} 23:59:59'[:10] + ' 23:59:59')

            cursor.execute(query, params)
            return cursor.fetchall()
        finally:
            conn.close()

    def get_attendance_stats(self, class_id: Optional[int] = None, student_id: Optional[int] = None) -> List[Tuple]:
        """Get attendance statistics.

        Args:
            class_id: ID of the class to filter by, or None for all classes.
            student_id: ID of the student to filter by, or None for all students.

        Returns:
            List of statistics tuples, format depends on filters:
            - class_id and student_id: (present_count, total_count)
            - class_id only: (student_id, present_count, total_count) per student
            - student_id only: (class_id, present_count, total_count) per class
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            if class_id and student_id:
                cursor.execute(
                    '''
                    SELECT 
                        COUNT(CASE WHEN status = 'present' THEN 1 END) as present_count,
                        COUNT(*) as total_count
                    FROM attendance_records
                    WHERE class_id = ? AND student_id = ?
                    ''',
                    (class_id, student_id),
                )
            elif class_id:
                cursor.execute(
                    '''
                    SELECT 
                        student_id,
                        COUNT(CASE WHEN status = 'present' THEN 1 END) as present_count,
                        COUNT(*) as total_count
                    FROM attendance_records
                    WHERE class_id = ?
                    GROUP BY student_id
                    ''',
                    (class_id,),
                )
            elif student_id:
                cursor.execute(
                    '''
                    SELECT 
                        class_id,
                        COUNT(CASE WHEN status = 'present' THEN 1 END) as present_count,
                        COUNT(*) as total_count
                    FROM attendance_records
                    WHERE student_id = ?
                    GROUP BY class_id
                    ''',
                    (student_id,),
                )
            return cursor.fetchall()
        finally:
            conn.close()

    def get_students_in_class(self, class_id: int) -> List[Tuple]:
        """Get all students enrolled in a class.

        Args:
            class_id: ID of the class.

        Returns:
            List of student records as tuples (id, name).
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                '''
                SELECT u.id, u.name
                FROM users u
                JOIN class_enrollments ce ON u.id = ce.student_id
                WHERE ce.class_id = ?
                ''',
                (class_id,),
            )
            return cursor.fetchall()
        finally:
            conn.close()