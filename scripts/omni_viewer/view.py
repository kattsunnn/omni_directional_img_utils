"""
全方位画像ビューワー - View (view.py)

PyQt6によるユーザーインターフェース表示および入力イベントの検知を担当する。
- ViewerWidget: 透視投影画像およびHUDの描画、マウス入力イベントのシグナル化
- OmniViewerWindow: メインウィンドウ（タブUI、コントロール、ステータスバー）
"""

import cv2
import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal, QRect
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QFont
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTabWidget,
    QStatusBar, QDoubleSpinBox
)


def cv_img_to_qimage(cv_img: np.ndarray) -> QImage:
    """OpenCV画像 (BGR) を QImage に変換"""
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()


# ==============================================================================
# ビューワーキャンバス (ViewerWidget)
# ==============================================================================
class ViewerWidget(QWidget):
    """
    透視投影画像の描画領域
    マウス操作イベントをキャッチし、コントローラー／モデルへシグナルを発行する
    - 通常ホイール: 縦FOV変更シグナル (wheelVertical)
    - Shift + ホイール: 横FOV変更シグナル (wheelHorizontal)
    - 左ドラッグ: 視線変更シグナル (dragged)
    """
    dragged = pyqtSignal(float, float)      # dx, dy
    wheelVertical = pyqtSignal(int)         # delta (通常ホイール: 縦FOV)
    wheelHorizontal = pyqtSignal(int)       # delta (Shift+ホイール: 横FOV)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # 描画データ
        self.display_pixmap: QPixmap | None = None
        self.yaw_deg: float = 0.0
        self.pitch_deg: float = 0.0
        self.fov_w_deg: float = 60.0
        self.fov_h_deg: float = 60.0
        self.dst_w: int = 0
        self.dst_h: int = 0
        self.has_image: bool = False

        # マウスドラッグ管理
        self.is_dragging: bool = False
        self.last_mouse_x: float = 0.0
        self.last_mouse_y: float = 0.0

        self.setStyleSheet("background-color: #1e1e1e;")

    def set_display_data(
        self,
        yaw: float,
        pitch: float,
        fov_w: float,
        fov_h: float,
        rendered_img: np.ndarray | None
    ):
        """Modelから通知された最新のレンダリング結果とパラメータを反映"""
        self.yaw_deg = yaw
        self.pitch_deg = pitch
        self.fov_w_deg = fov_w
        self.fov_h_deg = fov_h

        if rendered_img is not None:
            self.has_image = True
            self.dst_h, self.dst_w = rendered_img.shape[:2]
            qimg = cv_img_to_qimage(rendered_img)
            self.display_pixmap = QPixmap.fromImage(qimg)
        else:
            self.has_image = False
            self.display_pixmap = None
            self.dst_w = 0
            self.dst_h = 0

        self.update()

    # --------------------------------------------------------------------------
    # マウス入力イベント
    # --------------------------------------------------------------------------
    def mousePressEvent(self, event):
        self.setFocus()
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = True
            pos = event.position()
            self.last_mouse_x = pos.x()
            self.last_mouse_y = pos.y()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.is_dragging and (event.buttons() & Qt.MouseButton.LeftButton):
            pos = event.position()
            dx = pos.x() - self.last_mouse_x
            dy = pos.y() - self.last_mouse_y
            self.last_mouse_x = pos.x()
            self.last_mouse_y = pos.y()

            # 移動量をシグナル通知
            self.dragged.emit(dx, dy)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.is_dragging:
            self.is_dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        """
        ホイール入力の分岐:
        - Shift + ホイール: 横FOVのみ変更
        - 通常ホイール: 縦FOVのみ変更
        """
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return

        mods = event.modifiers()
        if mods & Qt.KeyboardModifier.ShiftModifier:
            # Shift押下: 横FOV変更
            self.wheelHorizontal.emit(delta)
        else:
            # 通常: 縦FOV変更
            self.wheelVertical.emit(delta)

        event.accept()

    # --------------------------------------------------------------------------
    # 描画処理
    # --------------------------------------------------------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect()

        # 背景塗りつぶし
        painter.fillRect(rect, QColor("#1e1e1e"))

        if self.display_pixmap and not self.display_pixmap.isNull():
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
            painter.setPen(QColor("#888888"))
            font = QFont()
            font.setPointSize(13)
            painter.setFont(font)
            text = "上部タブの「画像選択」から全方位画像 (.jpg / .png) を指定してください"
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

        # HUDオーバーレイ表示 (画面左上)
        if self.has_image:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(QColor(255, 255, 255, 200))
            painter.setBrush(QColor(0, 0, 0, 160))

            badge_rect = QRect(15, 15, 250, 115)
            painter.drawRoundedRect(badge_rect, 6, 6)

            font = QFont()
            font.setPointSize(9)
            painter.setFont(font)
            painter.drawText(25, 35, f"Yaw (水平角)   : {self.yaw_deg:6.1f}°")
            painter.drawText(25, 55, f"Pitch (垂直角) : {self.pitch_deg:6.1f}°")
            painter.drawText(25, 75, f"横FOV (水平)   : {self.fov_w_deg:6.1f}°")
            painter.drawText(25, 95, f"縦FOV (垂直)   : {self.fov_h_deg:6.1f}°")
            painter.drawText(25, 115, f"出力サイズ     : {self.dst_w} x {self.dst_h} px")

        painter.end()


# ==============================================================================
# メインウィンドウ (OmniViewerWindow)
# ==============================================================================
class OmniViewerWindow(QMainWindow):
    """
    全方位画像ビューワーのメインウィンドウ
    ユーザー操作によるUIイベントをシグナル化してControllerへ通知する
    """
    browseClicked = pyqtSignal()
    loadRequested = pyqtSignal(str)
    fovWChanged = pyqtSignal(float)
    fovHChanged = pyqtSignal(float)
    resetClicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("全方位画像ビューワー (Omni-directional Image Viewer)")
        self.resize(1000, 700)

        self._updating_ui: bool = False
        self._init_ui()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # ----------------------------------------------------------------------
        # 上部：タブウィジェット
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
        btn_browse.clicked.connect(self.browseClicked.emit)

        btn_load = QPushButton("読み込み")
        btn_load.clicked.connect(self._on_load_button_clicked)

        tab_file_layout.addWidget(lbl_path)
        tab_file_layout.addWidget(self.line_path, stretch=1)
        tab_file_layout.addWidget(btn_browse)
        tab_file_layout.addWidget(btn_load)
        self.tab_widget.addTab(tab_file, "画像選択")

        # タブ2: 操作 & 設定
        tab_controls = QWidget()
        tab_controls_layout = QHBoxLayout(tab_controls)
        tab_controls_layout.setContentsMargins(6, 6, 6, 6)
        tab_controls_layout.setSpacing(10)

        # 横FOVコントロール
        lbl_fov_w = QLabel("横FOV (水平):")
        self.spin_fov_w = QDoubleSpinBox()
        self.spin_fov_w.setRange(10.0, 140.0)
        self.spin_fov_w.setSingleStep(2.0)
        self.spin_fov_w.setValue(60.0)
        self.spin_fov_w.setSuffix("°")
        self.spin_fov_w.valueChanged.connect(self._on_spin_fov_w_changed)

        # 縦FOVコントロール
        lbl_fov_h = QLabel("縦FOV (垂直):")
        self.spin_fov_h = QDoubleSpinBox()
        self.spin_fov_h.setRange(10.0, 140.0)
        self.spin_fov_h.setSingleStep(2.0)
        self.spin_fov_h.setValue(60.0)
        self.spin_fov_h.setSuffix("°")
        self.spin_fov_h.valueChanged.connect(self._on_spin_fov_h_changed)

        # 操作説明ラベル
        lbl_help = QLabel("操作: [左ドラッグ] 視線移動  |  [ホイール] 縦FOV  |  [Shift+ホイール] 横FOV")
        lbl_help.setStyleSheet("color: #666666; font-weight: bold; margin-left: 8px;")

        # リセットボタン
        btn_reset = QPushButton("視点リセット")
        btn_reset.clicked.connect(self.resetClicked.emit)

        tab_controls_layout.addWidget(lbl_fov_w)
        tab_controls_layout.addWidget(self.spin_fov_w)
        tab_controls_layout.addWidget(lbl_fov_h)
        tab_controls_layout.addWidget(self.spin_fov_h)
        tab_controls_layout.addWidget(lbl_help)
        tab_controls_layout.addStretch(1)
        tab_controls_layout.addWidget(btn_reset)

        self.tab_widget.addTab(tab_controls, "操作・設定")

        # ----------------------------------------------------------------------
        # 中央：ビューワーキャンバス
        # ----------------------------------------------------------------------
        self.viewer = ViewerWidget()
        main_layout.addWidget(self.viewer, stretch=1)

        # ----------------------------------------------------------------------
        # 下部：ステータスバー
        # ----------------------------------------------------------------------
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("画像が選択されていません")

    # --------------------------------------------------------------------------
    # 内部イベントハンドラ (シグナル再送出)
    # --------------------------------------------------------------------------
    def _on_load_button_clicked(self):
        path = self.line_path.text().strip()
        self.loadRequested.emit(path)

    def _on_spin_fov_w_changed(self, value: float):
        if not self._updating_ui:
            self.fovWChanged.emit(value)

    def _on_spin_fov_h_changed(self, value: float):
        if not self._updating_ui:
            self.fovHChanged.emit(value)

    # --------------------------------------------------------------------------
    # Controllerから呼び出される更新メソッド
    # --------------------------------------------------------------------------
    def set_path_text(self, path: str):
        self.line_path.setText(path)

    def set_status_message(self, message: str):
        self.status_bar.showMessage(message)

    def update_controls(self, fov_w: float, fov_h: float):
        """スピンボックスの値を更新（シグナル無限ループ防止ガード付き）"""
        self._updating_ui = True
        self.spin_fov_w.setValue(fov_w)
        self.spin_fov_h.setValue(fov_h)
        self._updating_ui = False

