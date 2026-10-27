"""
全方位画像ビューワー (omni_viewer.py)

要件:
- PyQt6を使用した全方位画像のビューワー
- 画面上部にタブがあり、全方位画像（.jpg, .png）のパスを選択可能
- 透視投影画像の表示（生成部分はモック関数として実装し、現状は読み込んだ全方位画像を表示）
- 左クリックホールド移動で生成角度（Yaw, Pitch）を変更
- マウスホイールで生成領域（FOV / 画角）を変更
"""

import sys
import os
import numpy as np
import cv2

from PyQt6 import QtCore
from PyQt6.QtCore import Qt, pyqtSignal, QRect
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QLineEdit, QTabWidget,
    QStatusBar
)

from omni_directional_img_utils.e2p import E2P



# ==============================================================================
# 画像入出力・変換ヘルパー
# ==============================================================================
def load_image(file_path: str) -> np.ndarray | None:
    """
    全方位画像を読み込む関数。
    Windows環境で日本語や特殊文字を含むパスでも安全に読み込めるよう np.fromfile を使用。
    """
    if not os.path.exists(file_path):
        return None
    try:
        data = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"画像読み込みエラー ({file_path}): {e}")
        return None


def cv_img_to_qimage(cv_img: np.ndarray) -> QImage:
    """OpenCV画像 (BGR) を QImage に変換"""
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()


# ==============================================================================
# 透視投影画像生成（モック関数）
# ==============================================================================
def generate_perspective_image(
    src_img: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float
) -> np.ndarray:

    e2p = E2P(src_img.shape[1], src_img.shape[0])
    e2p.generate_map(fov_deg, fov_deg, yaw_deg, pitch_deg, 0)
    dst_img = e2p.generate_img(src_img)
        
    return dst_img


# ==============================================================================
# ビューワー描画ウィジェット (ViewerWidget)
# ==============================================================================
class ViewerWidget(QWidget):
    """
    画像表示およびマウス操作（ドラッグによる角度変更、ホイールによるFOV変更）を処理するウィジェット
    """
    # 視角パラメータ変更シグナル: (yaw, pitch, fov)
    viewChanged = pyqtSignal(float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # 全方位原画 (BGR)
        self.src_img: np.ndarray | None = None
        # 表示用QPixmap
        self.display_pixmap: QPixmap | None = None

        # 透視投影パラメータ
        self.yaw_deg: float = 0.0     # 水平回転角
        self.pitch_deg: float = 0.0   # 垂直回転角
        self.fov_deg: float = 60.0    # 画角 (生成領域)

        # マウスドラッグ管理
        self.is_dragging: bool = False
        self.last_mouse_x: float = 0.0
        self.last_mouse_y: float = 0.0

        # マウス感度
        self.rotate_sensitivity: float = 0.3  # 度 / ピクセル
        self.fov_step: float = 3.0            # ホイール1ノッチあたりのFOV変化量

        # 最小・最大FOV
        self.min_fov: float = 10.0
        self.max_fov: float = 140.0

        # 背景色設定
        self.setStyleSheet("background-color: #1e1e1e;")

    def set_source_image(self, img: np.ndarray | None):
        """全方位画像をセットして表示を更新"""
        self.src_img = img
        self.update_perspective_view()

    def set_parameters(self, yaw: float, pitch: float, fov: float):
        """パラメータを手動設定"""
        self.yaw_deg = yaw % 360.0
        self.pitch_deg = float(np.clip(pitch, -89.0, 89.0))
        self.fov_deg = float(np.clip(fov, self.min_fov, self.max_fov))
        self.update_perspective_view()
        self.viewChanged.emit(self.yaw_deg, self.pitch_deg, self.fov_deg)

    def reset_view(self):
        """視点と画角を初期値にリセット"""
        self.set_parameters(0.0, 0.0, 60.0)

    def update_perspective_view(self):
        """モック関数を呼び出して画像を生成し、再描画"""
        if self.src_img is None:
            self.display_pixmap = None
            self.update()
            return

        # 透視投影画像の生成
        rendered = generate_perspective_image(
            self.src_img, self.yaw_deg, self.pitch_deg, self.fov_deg
        )

        if rendered is not None:
            qimg = cv_img_to_qimage(rendered)
            self.display_pixmap = QPixmap.fromImage(qimg)
        else:
            self.display_pixmap = None

        self.update()

    # --------------------------------------------------------------------------
    # マウスイベント処理
    # --------------------------------------------------------------------------
    def mousePressEvent(self, event):
        """左クリック押下でドラッグ開始"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = True
            pos = event.position()
            self.last_mouse_x = pos.x()
            self.last_mouse_y = pos.y()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """左クリックホールド移動で角度 (Yaw / Pitch) を変更"""
        if self.is_dragging and (event.buttons() & Qt.MouseButton.LeftButton):
            pos = event.position()
            dx = pos.x() - self.last_mouse_x
            dy = pos.y() - self.last_mouse_y

            # 移動量に応じて角度を更新
            # 水平方向: ドラッグでYawを回転（横方向を反転）
            self.yaw_deg = (self.yaw_deg - dx * self.rotate_sensitivity) % 360.0
            # 垂直方向: ドラッグでPitchを変更 (-89度 〜 89度に制限)
            self.pitch_deg = float(np.clip(
                self.pitch_deg - dy * self.rotate_sensitivity,
                -89.0, 89.0
            ))

            self.last_mouse_x = pos.x()
            self.last_mouse_y = pos.y()

            # 描画更新とシグナル通知
            self.update_perspective_view()
            self.viewChanged.emit(self.yaw_deg, self.pitch_deg, self.fov_deg)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """ドラッグ終了"""
        if event.button() == Qt.MouseButton.LeftButton and self.is_dragging:
            self.is_dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        """マウスホイールで生成領域 (FOV) を変更"""
        delta = event.angleDelta().y()
        if delta != 0:
            # ホイール奥（拡大/領域狭める）でFOV減少、ホイール手前（縮小/領域広げる）でFOV増加
            delta_fov = -self.fov_step if delta > 0 else self.fov_step
            self.fov_deg = float(np.clip(
                self.fov_deg + delta_fov,
                self.min_fov, self.max_fov
            ))

            # 描画更新とシグナル通知
            self.update_perspective_view()
            self.viewChanged.emit(self.yaw_deg, self.pitch_deg, self.fov_deg)
            event.accept()
        else:
            super().wheelEvent(event)

    # --------------------------------------------------------------------------
    # 描画処理
    # --------------------------------------------------------------------------
    def paintEvent(self, event):
        """画像およびステータス情報の描画"""
        painter = QPainter(self)
        rect = self.rect()

        # 背景塗りつぶし
        painter.fillRect(rect, QColor("#1e1e1e"))

        if self.display_pixmap and not self.display_pixmap.isNull():
            # アスペクト比を維持して中央にスケーリング描画
            pm = self.display_pixmap
            scaled = pm.scaled(
                rect.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            draw_x = (rect.width() - scaled.width()) // 2
            draw_y = (rect.height() - scaled.height()) // 2
            painter.drawPixmap(draw_x, draw_y, scaled)
        else:
            # 未読み込み時のプレースホルダーメッセージ
            painter.setPen(QColor("#888888"))
            font = QFont()
            font.setPointSize(13)
            painter.setFont(font)
            text = "上部タブの「画像選択」から全方位画像 (.jpg / .png) を指定してください"
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

        # 画面左上に現在の操作パラメータをオーバーレイ表示（HUD）
        if self.src_img is not None:
            painter.setPen(QColor(255, 255, 255, 200))
            painter.setBrush(QColor(0, 0, 0, 140))
            badge_rect = QRect(15, 15, 230, 75)
            painter.drawRoundedRect(badge_rect, 6, 6)

            font = QFont()
            font.setPointSize(9)
            painter.setFont(font)
            painter.drawText(25, 35, f"Yaw (水平角)   : {self.yaw_deg:6.1f}°")
            painter.drawText(25, 55, f"Pitch (垂直角) : {self.pitch_deg:6.1f}°")
            painter.drawText(25, 75, f"FOV (生成領域) : {self.fov_deg:6.1f}°")

        painter.end()


# ==============================================================================
# メインウィンドウ (OmniViewerWindow)
# ==============================================================================
class OmniViewerWindow(QMainWindow):
    """
    全方位画像ビューワーのメインウィンドウ
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("全方位画像ビューワー (Omni-directional Image Viewer)")
        self.resize(1000, 700)

        self.current_image_path: str = ""

        self._init_ui()

    def _init_ui(self):
        # メインコンテナ
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # ----------------------------------------------------------------------
        # 画面上部：タブウィジェット
        # ----------------------------------------------------------------------
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # タブ1: 画像選択
        tab_file = QWidget()
        tab_file_layout = QHBoxLayout(tab_file)
        tab_file_layout.setContentsMargins(6, 6, 6, 6)
        tab_file_layout.setSpacing(8)

        lbl_path = QLabel("画像パス:")
        self.line_path = QLineEdit()
        self.line_path.setPlaceholderText("全方位画像ファイル (.jpg, .jpeg, .png) のパスを選択または入力...")
        self.line_path.returnPressed.connect(self._on_load_button_clicked)

        btn_browse = QPushButton("📁 参照...")
        btn_browse.clicked.connect(self._on_browse_clicked)

        btn_load = QPushButton("読み込み")
        btn_load.clicked.connect(self._on_load_button_clicked)

        tab_file_layout.addWidget(lbl_path)
        tab_file_layout.addWidget(self.line_path, stretch=1)
        tab_file_layout.addWidget(btn_browse)
        tab_file_layout.addWidget(btn_load)
        self.tab_widget.addTab(tab_file, "画像選択")

        # タブ2: 操作 & パラメータ設定
        tab_controls = QWidget()
        tab_controls_layout = QHBoxLayout(tab_controls)
        tab_controls_layout.setContentsMargins(6, 6, 6, 6)
        tab_controls_layout.setSpacing(12)

        # 操作説明ラベル
        lbl_help = QLabel("操作: [左ドラッグ] 視線角度変更 (Yaw/Pitch)  |  [ホイール] 生成領域変更 (FOV)")
        lbl_help.setStyleSheet("color: #555555; font-weight: bold;")
        tab_controls_layout.addWidget(lbl_help)

        tab_controls_layout.addStretch(1)

        # リセットボタン
        btn_reset = QPushButton("視点・領域リセット")
        btn_reset.clicked.connect(self._on_reset_clicked)
        tab_controls_layout.addWidget(btn_reset)

        self.tab_widget.addTab(tab_controls, "操作・設定")

        # ----------------------------------------------------------------------
        # 中央：ビューワーウィジェット
        # ----------------------------------------------------------------------
        self.viewer = ViewerWidget()
        self.viewer.viewChanged.connect(self._on_view_changed)
        main_layout.addWidget(self.viewer, stretch=1)

        # ----------------------------------------------------------------------
        # 下部：ステータスバー
        # ----------------------------------------------------------------------
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("画像が選択されていません")

    # --------------------------------------------------------------------------
    # イベントハンドラ
    # --------------------------------------------------------------------------
    def _on_browse_clicked(self):
        """ファイルダイアログを開いて画像を選択"""
        file_filter = "全方位画像 (*.jpg *.jpeg *.png);;すべてのファイル (*.*)"
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "全方位画像ファイルを選択",
            "",
            file_filter
        )
        if file_path:
            self.line_path.setText(file_path)
            self._load_image_from_path(file_path)

    def _on_load_button_clicked(self):
        """パス入力欄のパスから画像を読み込み"""
        file_path = self.line_path.text().strip()
        if file_path:
            self._load_image_from_path(file_path)

    def _load_image_from_path(self, file_path: str):
        """指定パスから画像を読み込んでビューワーに反映"""
        if not os.path.isfile(file_path):
            self.status_bar.showMessage(f"エラー: ファイルが見つかりません ({file_path})")
            return

        # 拡張子チェック (jpg, jpeg, png)
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png"]:
            self.status_bar.showMessage(f"エラー: 対応していない拡張子です ({ext})")
            return

        img = load_image(file_path)
        if img is None:
            self.status_bar.showMessage(f"エラー: 画像の読み込みに失敗しました ({file_path})")
            return

        self.current_image_path = file_path
        h, w, _ = img.shape
        self.viewer.set_source_image(img)
        self.status_bar.showMessage(
            f"読み込み完了: {os.path.basename(file_path)} ({w}x{h}) | "
            f"Yaw: {self.viewer.yaw_deg:.1f}° | Pitch: {self.viewer.pitch_deg:.1f}° | FOV: {self.viewer.fov_deg:.1f}°"
        )

    def _on_view_changed(self, yaw: float, pitch: float, fov: float):
        """視点・FOV変更時にステータスバーを更新"""
        if self.viewer.src_img is not None:
            filename = os.path.basename(self.current_image_path)
            self.status_bar.showMessage(
                f"ファイル: {filename} | "
                f"Yaw: {yaw:5.1f}° | Pitch: {pitch:5.1f}° | FOV: {fov:5.1f}°"
            )

    def _on_reset_clicked(self):
        """視点と画角をリセット"""
        self.viewer.reset_view()


# ==============================================================================
# メインエントリーポイント
# ==============================================================================
def main():
    app = QApplication(sys.argv)
    window = OmniViewerWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
