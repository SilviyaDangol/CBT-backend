import uuid

import bcrypt
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import joinedload

from ..teacher.routes import generate_token
from flask import Blueprint, request, jsonify, g, url_for
from ...db import db
from ...db.models import Student, Teacher, Classroom, student_class_association, Session
from ...auth import teacher_required, student_required
from ...utils.uploads import generate_strong_password
bp = Blueprint('student', __name__, url_prefix='/student')



@bp.route('/login', methods=['POST'])
def login_student():
    data = request.get_json()
    email = data['email']
    password = data['password']

    student = Student.query.filter_by(email=email).first()
    if email and password is None:
        return jsonify(
            {
                'message': 'Email and password are required'
            }
        )

    if student and bcrypt.checkpw(password.encode('utf-8'), student.password_hash.encode('utf-8')):
        return jsonify(
            {
                'message': 'Teacher created successfully',
                'token': generate_token(student.id, student.role, student.first_name)
            }), 200
    else:
        return jsonify(
            {
                'message': 'Username or Password incorrect'
            }
        ), 201

@bp.route('/<string:course_id>/bulk_create', methods=['POST'])
@teacher_required
def bulk_create_students(course_id):
    # Validate request data
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    # Validate required fields
    if "students" not in data:
        return jsonify({"error": "Missing students array"}), 400

    students_data = data.get("students", [])
    email_domain = data.get("email_domain", "")  # Optional since we can use a default

    # Set default email domain if not provided
    if not email_domain:
        email_domain = "school.edu"  # Change this to your default domain
    email_domain = email_domain.strip().lower()

    # Validate students data structure
    if not isinstance(students_data, list):
        return jsonify({"error": "students must be an array"}), 400

    # Get the classroom
    classroom = Classroom.query.get(course_id)
    if not classroom:
        return jsonify({"error": "Classroom not found"}), 404

    # Verify the classroom belongs to the current teacher
    if str(classroom.teacher_id) != str(g.current_user.id):
        return jsonify({"error": "Unauthorized - You don't own this classroom"}), 403

    created = []
    errors = []

    # Process each student
    for i, student in enumerate(students_data):
        try:
            # Validate student data
            if not isinstance(student, dict):
                raise ValueError("Student data must be an object")

            first_name = student.get("first_name", "").strip()
            last_name = student.get("last_name", "").strip()

            if not first_name or not last_name:
                raise ValueError("First name and last name cannot be empty")

            # Generate email and password
            email_base = f"{first_name.lower()}.{last_name.lower()}"
            email = f"{email_base}@{email_domain}"

            # Check for existing student
            existing_student = Student.query.filter_by(email=email).first()
            if existing_student:
                # If student exists, just add to classroom if not already enrolled
                if classroom not in existing_student.classes:
                    existing_student.classes.append(classroom)
                    db.session.add(existing_student)
                    created.append({
                        "email": email,
                        "action": "added_to_class",
                        "first_name": first_name,
                        "last_name": last_name
                    })
                else:
                    errors.append({
                        "index": i,
                        "student": student,
                        "error": "Student already exists and is enrolled in this class"
                    })
                continue

            # Create new student
            password = generate_strong_password()
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

            new_student = Student(
                id=uuid.uuid4(),
                first_name=first_name,
                last_name=last_name,
                email=email,
                password_hash=hashed_password.decode('utf-8'),
                role="student",
                teacher_id=g.current_user.id
            )

            # Add to classroom
            new_student.classes.append(classroom)

            db.session.add(new_student)
            created.append({
                "email": email,
                "password": password,  # Only returned for new accounts
                "first_name": first_name,
                "last_name": last_name,
                "action": "created_and_added"
            })

        except Exception as e:
            errors.append({
                "index": i,
                "student": student,
                "error": str(e)
            })

    # Attempt to commit all changes
    try:
        db.session.commit()
    except IntegrityError as e:
        db.session.rollback()
        return jsonify({
            "message": "Database error occurred",
            "error": str(e.orig),
            "created_count": len(created),
            "errors": errors
        }), 400

    # Prepare response
    response = {
        "message": f"Processed {len(created) + len(errors)} students",
        "created_count": len([c for c in created if c["action"] == "created_and_added"]),
        "added_count": len([c for c in created if c["action"] == "added_to_class"]),
        "error_count": len(errors),
        "results": created,
    }

    if errors:
        response["errors"] = errors

    status_code = 207 if (created and errors) else 201 if created else 400

    return jsonify(response), status_code

@bp.route('/all/course', methods=['GET'])
@student_required
def get_teacher_courses():
    try:
        teacher = db.session.query(Teacher).options(joinedload(Teacher.classes)).filter_by(id=g.user_id).first()
        if not teacher:
            return jsonify({'error': 'No classes found'}), 201
        courses = []
        for classroom in teacher.classes:
            courses.append({
                'id': str(classroom.id),
                'name': classroom.name,
                'description': classroom.description,
                'created_at': classroom.created_at,
                'image': url_for('get_image', filename=classroom.image)
            })

        return jsonify({'teacher_id': str(g.user_id), 'courses': courses}), 200

    except Exception as e:
        return jsonify({'error': 'An error occurred', 'message': str(e)}), 500

@bp.route('/classes/<string:classroom_id>', methods=['GET'])
@teacher_required
def get_class(classroom_id):
    classroom = Classroom.query.get(classroom_id)
    if not classroom:
        return jsonify({'error': 'Classroom not found'}), 404

    return jsonify({
        'id': str(classroom.id),
        'name': classroom.name,
        'description': classroom.description,
        'created_at': classroom.created_at,
        'image': url_for('static', filename=f'uploads/{classroom.image}')
    }), 200


@bp.route("/course/<string:course_id>", methods=["GET"])
@teacher_required
def get_students_by_course(course_id):
    # Query the Classroom by its ID and eagerly load students associated with it
    classroom = db.session.query(Classroom).options(joinedload(Classroom.students)).filter(
        Classroom.id == course_id).first()

    if not classroom:
        return jsonify({"detail": "No class found with that course ID"}), 404

    students_data = []

    # Collect students associated with the classroom
    for student in classroom.students:
        students_data.append({
            "id": str(student.id),
            "first_name": student.first_name,
            "last_name": student.last_name,
            "email": student.email,
            "role": student.role,
            "created_at": student.created_at.isoformat(),
        })

    return jsonify({"course_id": str(course_id), "students": students_data}), 200