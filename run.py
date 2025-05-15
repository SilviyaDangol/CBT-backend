from src.app import create_app
from src.config import Config

cfg = Config()
app = create_app(cfg)


if __name__ == '__main__':
    app.run(host=cfg.FLASK_HOST, port=cfg.FLASK_PORT, debug=cfg.FLASK_DEBUG)