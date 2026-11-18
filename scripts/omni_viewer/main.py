"""
全方位画像ビューワー - メインスクリプト (main.py)

MVC構成（model.py, view.py, controller.py）を組み立ててアプリケーションを起動する。
"""

import sys
import os

# どこから実行されても正しくインポートできるよう sys.path を設定
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
src_dir = os.path.join(project_root, "src")

for p in [src_dir, current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from PyQt6.QtWidgets import QApplication

from model import OmniViewerModel
from view import OmniViewerWindow
from controller import OmniViewerController


def main():
    """アプリケーションの起動"""
    app = QApplication(sys.argv)
    model = OmniViewerModel()
    window = OmniViewerWindow()
    _controller = OmniViewerController(model, window)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
