import os
from flask import Blueprint, request, jsonify, url_for
from sqlalchemy import func
import traceback
import uuid
import torch
import numpy as np
import cv2
from ...config import Config
from ...db import db
from ...db.models import Session, Behaviour, Classroom
from ...auth import teacher_required
from ...models.predict import detect_behavior, draw_detections
import time
import logging

print = logging.error

bp = Blueprint('session', __name__, url_prefix='/session')


@bp.route('/create/<string:uuid_value>', methods=['POST', 'GET'])
@teacher_required
def create_or_check_session(uuid_value):
    try:
        if request.method == 'GET':
            session = Session.query.filter_by(class_id=uuid_value).order_by(Session.created_at.desc()).first()

            if not session:
                return jsonify({'message': 'No session found for this class.'}), 200

            if session.status:
                return jsonify({
                    'message': 'Session is already active.',
                    'session_id': str(session.id),
                    'status': True
                }), 200
            else:
                return jsonify({
                    'message': 'Session exists but is not active.',
                    'session_id': str(session.id)
                }), 200

        elif request.method == 'POST':
            existing_session = Session.query.filter_by(class_id=uuid_value, status=True).first()
            if existing_session:
                existing_session.status = False
                db.session.commit()
                return jsonify({
                    'message': 'Existing active session has been deactivated',
                    'session_id': str(existing_session.session_id)
                }), 200
            else:
                new_session = Session(
                    id=uuid.uuid4(),
                    session_id=uuid.uuid4(),
                    class_id=uuid_value,
                    status=True,
                )
                db.session.add(new_session)
                db.session.commit()

                return jsonify({
                    'message': 'Session created successfully',
                    'session_id': str(new_session.id)
                }), 201

        # Handle any other HTTP methods
        return jsonify({'message': 'Method not allowed'}), 405

    except Exception as e:
        db.session.rollback()  # Rollback on error
        return jsonify({'error': f"{str(e)}"}), 500
@bp.route('/stats/session/<string:session_id>', methods=['GET'])
@teacher_required
def get_session_stats(session_id):
    """Get behavior statistics for a specific session"""
    try:
        # Get basic session info
        session = db.session.query(Session).filter(Session.id == session_id).first()
        if not session:
            return jsonify({'message': 'Session not found'}), 404

        # Get behavior counts
        behavior_results = (
            db.session.query(Behaviour.behaviour, func.count(Behaviour.id))
            .filter(Behaviour.session_id == session_id)
            .group_by(Behaviour.behaviour)
            .all()
        )
        # Format results with default values
        stats = {
            'session_id': str(session_id),
            'class_id': str(session.class_id),
            'start_time': session.class_started_at.isoformat(),
            'behaviors': {b: c for b, c in behavior_results}
        }
        for behavior in ['hand-raising', 'writing', 'reading']:
            stats['behaviors'].setdefault(behavior, 0)

        return jsonify({'message': stats}), 200

    except Exception as e:
        return jsonify({'message': str(e)}), 400


@bp.route('/stats/classroom/<string:class_id>/sessions', methods=['GET'])
@teacher_required
def get_classroom_sessions_stats(class_id):
    try:
        classroom = db.session.query(Classroom).filter(Classroom.id == class_id).first()
        if not classroom:
            return jsonify({'message': 'Classroom not found'}), 404
        sessions = db.session.query(Session).filter(Session.class_id == class_id).all()
        session_ids = [str(s.id) for s in sessions]
        behavior_results = (
            db.session.query(Behaviour.behaviour, func.count(Behaviour.id))
            .join(Session)
            .filter(Session.class_id == class_id)
            .group_by(Behaviour.behaviour)
            .all()
        )
        stats = {
            'class_id': str(class_id),
            'session_count': len(sessions),
            'session_ids': session_ids,
            'total_behaviors': sum(c for _, c in behavior_results),
            'behaviors': {b: c for b, c in behavior_results}
        }

        # Ensure all expected behaviors are included
        for behavior in ['hand-raising', 'writing', 'reading']:
            stats['behaviors'].setdefault(behavior, 0)

        return jsonify({'message': stats}), 200

    except Exception as e:
        return jsonify({'message': str(e)}), 400


@bp.route('/stats/classroom/<string:class_id>/session/<string:session_id>', methods=['GET'])
@teacher_required
def get_classroom_session_stats(class_id, session_id):
    try:
        # Verify session belongs to classroom
        session = (
            db.session.query(Session)
            .filter(Session.id == session_id)
            .filter(Session.class_id == class_id)
            .first()
        )

        if not session:
            return jsonify({'message': 'Session not found in this classroom'}), 404

        # Reuse the session stats function
        return get_session_stats(session_id)

    except Exception as e:
        return jsonify({'message': str(e)}), 400

@bp.route('/detect/<string:session_id>', methods=['POST'])
@teacher_required
def create_detection(session_id):
    """Endpoint for processing images and detecting behaviors."""
    print(f"\n=== STARTING DETECTION PIPELINE for session {session_id} ===")
    start_time = time.time()
    
    try:
        # Validate session_id is a valid UUID
        try:
            session_uuid = uuid.UUID(session_id)
        except ValueError:
            print("[ERROR] Invalid session ID format")
            return jsonify({"error": "Invalid session ID format"}), 400

        # 1. GPU Resource Management
        if torch.cuda.is_available():
            initial_mem = torch.cuda.memory_allocated()
            print(f"[GPU] Initial memory: {initial_mem/1024**2:.2f}MB")
            torch.cuda.empty_cache()
            print("[GPU] CUDA cache cleared")

        # 2. Validate Input
        print("\n[PHASE 1] INPUT VALIDATION")
        if 'image' not in request.files:
            print("[ERROR] No file part in request")
            return jsonify({"error": "No image provided"}), 400

        file = request.files['image']
        if not file or file.filename == '':
            print("[ERROR] Empty file submitted")
            return jsonify({"error": "No selected file"}), 400

        print(f"[SUCCESS] Received valid file: {file.filename}")

        # 3. Image Processing
        print("\n[PHASE 2] IMAGE PROCESSING")
        try:
            img_bytes = file.read()
            image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                print("[ERROR] Failed to decode image")
                return jsonify({"error": "Invalid image format"}), 400
            
            height, width = image.shape[:2]
            print(f"[SUCCESS] Image decoded - Dimensions: {width}x{height}")
        except Exception as e:
            print(f"[ERROR] Image processing failed: {str(e)}")
            print(traceback.format_exc())
            return jsonify({"error": "Image processing error"}), 400

        # 4. Behavior Detection
        print("\n[PHASE 3] BEHAVIOR DETECTION")
        try:
            detections = detect_behavior(image)
            print(f"[INFO] Found {len(detections)} potential detections")
            
            if torch.cuda.is_available():
                peak_mem = torch.cuda.max_memory_allocated()
                print(f"[GPU] Peak memory during detection: {peak_mem/1024**2:.2f}MB")
        except Exception as e:
            print(f"[ERROR] Detection failed: {str(e)}")
            print(traceback.format_exc())
            return jsonify({"error": "Behavior detection failed"}), 500

        # 5. Results Processing with Coordinate Validation
        print("\n[PHASE 4] RESULTS PROCESSING")
        # processed_detections = []
        valid_count = 0
        
        for det in detections:
            try:
                if 'class' not in det and 'label' in det:
                    det['class'] = det['label']  # Some models use 'label' instead of 'class'

                # Ensure confidence exists
                if 'confidence' not in det and 'score' in det:
                    det['confidence'] = det['score']  # Some models use 'score' instead of 'confidence'

                # Ensure bbox exists in the expected format [x1, y1, x2, y2]
                if 'bbox' not in det:
                    if all(k in det for k in ['x1', 'y1', 'x2', 'y2']):
                        det['bbox'] = [det['x1'], det['y1'], det['x2'], det['y2']]
            except (ValueError, TypeError, KeyError) as e:
                print(f"[WARNING] Invalid detection : {str(e)}")
                print(traceback.format_exc())
                continue

        print(f"[SUMMARY] Processed {valid_count}/{len(detections)} valid detections")

        # 6. Visualization and Storage with Error Handling
        print("\n[PHASE 5] VISUALIZATION & STORAGE")
        try:
            if not detections:
                print("[INFO] No valid detections to visualize")
                vis_image = image.copy()
            else:
                print("[INFO] Drawing detections on image")
                vis_image = draw_detections(image.copy(), detections)
            
            file_ext = os.path.splitext(file.filename)[1][1:] or 'jpg'
            output_filename = f"detection_{session_id}_{int(time.time())}.{file_ext}"
            output_path = os.path.join(Config.IMAGE_PATH, output_filename)
            
            os.makedirs(Config.IMAGE_PATH, exist_ok=True)
            if not cv2.imwrite(output_path, vis_image):
                raise RuntimeError("Failed to save visualization")
            
            print(f"[SUCCESS] Visualization saved to {output_path}")
        except Exception as e:
            print(f"[ERROR] Visualization failed: {str(e)}")
            print(traceback.format_exc())
            return jsonify({"error": "Result visualization failed"}), 500

        # 7. Database Operations using Behaviour model
        print("\n[PHASE 6] DATABASE OPERATIONS")
        try:           
            for det in detections:
                bbox = det['bbox']
                behaviour_record = Behaviour(
                    id = uuid.uuid4(),
                    session_id=session_uuid,
                    behaviour=det['class'],
                    x_axis=bbox[0],
                    y_axis=bbox[1],
                    w_axis=bbox[2] - bbox[0],
                    h_axis=bbox[3] - bbox[1],
                    confidence=det['confidence'],
                    image=output_filename
                )
                db.session.add(behaviour_record)
            
            db.session.commit()
            print(f"[SUCCESS] Saved {len(detections)} behaviour records to database")
        except Exception as e:
            db.session.rollback()
            print(traceback.format_exc())
            print(f"[ERROR] Database operation failed: {str(e)}")
            return jsonify({"error": "Database operation failed"}), 500

        # 8. Prepare Response
        response = {
            "success": True,
            "session_id": session_id,
            "detections": detections,
            "visualization": url_for('static', filename=f"uploads/{output_filename}"),
            "processing_time": round(time.time() - start_time, 2),
            "detection_count": len(detections)
        }

        print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
        print(f"Total processing time: {response['processing_time']} seconds")
        print("RESP SENT---------------------------------------------------------------------")
        return jsonify(response), 200

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Pipeline failed: {str(e)}")
        print(traceback.format_exc())
        db.session.rollback()
        return jsonify({"error": "Processing pipeline failed"}), 500

    # finally:
    #     # Resource Cleanup
    #     if torch.cuda.is_available():
    #         final_mem = torch.cuda.memory_allocated()
    #         torch.cuda.empty_cache()
    #         print(f"\n[GPU] Final memory usage: {final_mem/1024**2:.2f}MB")
    #         print(f"[GPU] Memory freed: {(torch.cuda.memory_allocated() - final_mem)/1024**2:.2f}MB")
    #
    #     if 'image' in locals():
    #         del image
    #     if 'vis_image' in locals():
    #         del vis_image
    #     print("=== RESOURCES CLEANED UP ===")
