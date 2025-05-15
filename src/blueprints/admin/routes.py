import uuid

import bcrypt
from flask import Blueprint, jsonify, g , url_for
from src.auth import auth_required
from src.db import db
from src.db.models import Teacher
from sqlalchemy.orm import joinedload


bp = Blueprint('admin', __name__, url_prefix='/admin')


@bp.route('/all-teacher', methods=['POST'])
# @auth_required(roles=['admin'])
def getAllTeacher():
    try:
        # if g.role != 'admin':
        #     return jsonify({'error': 'Unauthorized'}), 401

        # Eager load relationships for performance
        teachers = Teacher.query.options(
            joinedload(Teacher.students),
            joinedload(Teacher.classes)
        ).all()

        if not teachers:
            return jsonify({'error': 'No teachers found'}), 201

        serialized_teachers = []
        for teacher in teachers:
            if teacher.role == 'admin':
                continue
            serialized_classes = [{
                'id': str(cls.id),
                'name': cls.name,
                'image': url_for('static', filename=f"uploads/{cls.image}"),
                'description': cls.description
            } for cls in teacher.classes]
            serialized_teachers.append({
                'id': str(teacher.id),
                'full_name': teacher.first_name + ' ' + teacher.last_name,
                'email': teacher.email,
                'role': teacher.role,
                'created_at': teacher.created_at.isoformat() if teacher.created_at else None,
                'student_count': len(teacher.students),
                'course_count': len(teacher.classes),
                'classes': serialized_classes  # Add classes list with image and name
            })

        return jsonify({'teachers': serialized_teachers}), 200

    except Exception as e:
        return jsonify({'error': 'An error occurred', 'message': str(e)}), 500


@bp.route('/seed-teacher', methods=['POST'])
def seed_admin_teacher():
    try:
        # Check if admin already exists
        if Teacher.query.filter_by(email="admin@school.edu").first():
            return jsonify({"message": "Admin teacher already exists"}), 200

        # Create admin teacher
        admin = Teacher(
            id=uuid.uuid4(),
            first_name="Admin",
            last_name="User",
            email="admin@school.edu",
            password_hash=bcrypt.hashpw("Admin123!".encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
            role="admin",
        )

        db.session.add(admin)
        db.session.commit()

        return jsonify({
            "message": "Admin teacher seeded successfully",
            "email": admin.email,
            "role": admin.role
        }), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({
            "message": "Failed to seed admin teacher",
            "error": str(e)
        }), 500

