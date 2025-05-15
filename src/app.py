from flask import Flask
from flask_cors import CORS
from .config import Config
from .db import db
from .blueprints.session.rotues import bp as bp_session
from .blueprints.student.routes import bp as bp_student
from .blueprints.teacher.routes import bp as bp_teacher
from .blueprints.dashboard.routes import bp as bp_dashboard
from .models.predict import initialize_model
from .blueprints.admin.routes import bp as bp_admin

def create_app(config: Config=Config):
    app = Flask(__name__)
    CORS(app)
    print("Initializing YOLOv8 model...")
    initialize_model(Config.MODEL_PATH)
    print("Model initialization complete!")
    app.config.from_object(config)
    app.register_blueprint(bp_session)
    app.register_blueprint(bp_student)
    app.register_blueprint(bp_teacher)
    app.register_blueprint(bp_dashboard)
    app.register_blueprint(bp_admin)
    db.init_app(app)
    with app.app_context():
        db.create_all()
    return app
