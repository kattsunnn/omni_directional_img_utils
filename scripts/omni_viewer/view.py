"""
全方位画像ビューワー - View (view.py)

PyQt6による洗練されたダークテーマUI表示および入力イベントの検知を担当する。
- ViewerWidget: 透視投影画像およびHUDの描画、ドラッグ＆ドロップ画像読み込み、
  ロック中の点・線選択クリック検知、Dキーによる削除検知、マーカー・線分の描画
- OmniViewerWindow: メインウィンドウ（タブUI、コントロール、ステータスバー、ロックボタン、モード切替、保存ボタン）
"""

import os
import cv2
import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal, QRect, QPointF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QPen, QGuiApplication
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTabWidget,
    QStatusBar, QDoubleSpinBox
)

# モダンダークUIスタイルシート
MODERN_STYLE = """
QWidget {
    font-family: 'Segoe UI', 'Yu Gothic UI', Meiryo, sans-serif;
    font-size: 12px;
    color: #e4e4e7;
}
QMainWindow {
    background-color: #121214;
}
QTabWidget::pane {
    border: 1px solid #27272a;
    background-color: #18181b;
    border-radius: 6px;
    margin-top: -1px;
}
QTabBar::tab {
    background-color: #18181b;
    color: #a1a1aa;
    padding: 6px 16px;
    border: 1px solid #27272a;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #27272a;
    color: #ffffff;
    font-weight: bold;
}
QPushButton {
    background-color: #27272a;
    color: #f4f4f5;
    border: 1px solid #3f3f46;
    border-radius: 4px;
    padding: 5px 12px;
}
QPushButton:enabled:hover {
    background-color: #3f3f46;
}
QPushButton:pressed {
    background-color: #18181b;
}
QPushButton:disabled {
    background-color: #18181b;
    color: #52525b;
    border: 1px solid #27272a;
}
QLineEdit, QDoubleSpinBox {
    background-color: #09090b;
    border: 1px solid #27272a;
    border-radius: 4px;
    padding: 4px 8px;
    color: #f4f4f5;
}
QLineEdit:focus, QDoubleSpinBox:focus {
    border: 1px solid #00e5ff;
}
QStatusBar {
    background-color: #18181b;
    color: #a1a1aa;
    border-top: 1px solid #27272a;
}
"""


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
    - 通常時:
      - 左ドラッグ: 視線回転 (Yaw, Pitch)
      - 通常ホイール: 縦横両方のFOVを変更（同じ数ずつ加算/減算）
      - Shift + ホイール: 縦FOVのみ変更
      - Ctrl + ホイール: 横FOVのみ変更
    - ロック中:
      - 点選択モード: クリックした位置に点を配置
      - 線選択モード: 始点・端点の2点クリックで線を引く
      - D キー: 直前の要素を削除
      - M キー: 点/線モード切替
    - ロック解除時: 配置されたすべての点・線は自動クリア
    """
    dragged = pyqtSignal(float, float)      # dx, dy
    wheelBoth = pyqtSignal(int)             # delta (縦横両方)
    wheelVertical = pyqtSignal(int)         # delta (縦のみ)
    wheelHorizontal = pyqtSignal(int)       # delta (横のみ)
    wheelModeChanged = pyqtSignal(str)      # ホイールモード変更通知 ("both"/"v"/"h")
    lockToggled = pyqtSignal()              # L / Space キーによるロック切り替え
    modeToggled = pyqtSignal()              # M キーによる点/線モード切り替え
    pointClicked = pyqtSignal(float, float) # img_x, img_y (画像内クリック座標)
    undoRequested = pyqtSignal()            # D キーによる直前要素削除
    fileDropped = pyqtSignal(str)           # ドラッグ＆ドロップされた画像パス

    # ホイールモードのループ順
    WHEEL_MODES = ("both", "v", "h")
    WHEEL_MODE_LABELS = {"both": "縦横", "v": "縦", "h": "横"}

    # 点・線マーカー用のカラーパレット
    ANNOTATION_COLORS = [
        QColor("#00e5ff"),  # シアン
        QColor("#ffd600"),  # イエロー
        QColor("#76ff03"),  # ライム
        QColor("#ff4081"),  # ピンク
        QColor("#b388ff"),  # パープル
        QColor("#ff9100"),  # オレンジ
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAcceptDrops(True)

        # 描画データ
        self.display_pixmap: QPixmap | None = None
        self.yaw_deg: float = 0.0
        self.pitch_deg: float = 0.0
        self.fov_w_deg: float = 60.0
        self.fov_h_deg: float = 60.0
        self.dst_w: int = 0
        self.dst_h: int = 0
        self.has_image: bool = False

        # ロック状態および注釈データ
        self.is_locked: bool = False
        self.select_mode: str = "point"  # "point" or "line"
        self.points: list[dict] = []
        self.lines: list[dict] = []
        self.pending_line_start: dict | None = None

        # ホイールモード: "both" / "v" / "h"
        self.wheel_mode: str = "both"

        # ウィジェット内の画像描画領域: (draw_x, draw_y, scaled_w, scaled_h)
        self.render_rect: tuple[float, float, float, float] | None = None

        # マウスドラッグ管理
        self.is_dragging: bool = False
        self.last_mouse_x: float = 0.0
        self.last_mouse_y: float = 0.0

        self.setStyleSheet("background-color: #121214;")

    def set_locked(self, locked: bool):
        """ロック状態を設定してキャンバスを再描画"""
        self.is_locked = locked
        if self.is_locked:
            self.setCursor(Qt.CursorShape.CrossCursor)
            if self.is_dragging:
                self.is_dragging = False
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.points.clear()
            self.lines.clear()
            self.pending_line_start = None
        self.update()

    def set_select_mode(self, mode: str):
        """選択モード ("point" or "line") を更新"""
        self.select_mode = mode
        self.update()

    def set_points(self, points: list[dict]):
        """Modelから通知された配置点リストを保持して再描画"""
        self.points = points
        self.update()

    def set_lines(self, lines: list[dict], pending_start: dict | None):
        """Modelから通知された線リストを保持して再描画"""
        self.lines = lines
        self.pending_line_start = pending_start
        self.update()

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
    # ドラッグ＆ドロップイベント
    # --------------------------------------------------------------------------
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                ext = os.path.splitext(url.toLocalFile())[1].lower()
                if ext in [".jpg", ".jpeg", ".png"]:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                ext = os.path.splitext(path)[1].lower()
                if ext in [".jpg", ".jpeg", ".png"]:
                    self.fileDropped.emit(path)
                    event.acceptProposedAction()
                    return
        event.ignore()

    # --------------------------------------------------------------------------
    # マウス・キー入力イベント
    # --------------------------------------------------------------------------
    def keyPressEvent(self, event):
        """キーショートカット検知"""
        if event.key() in (Qt.Key.Key_L, Qt.Key.Key_Space):
            self.lockToggled.emit()
            event.accept()
        elif event.key() == Qt.Key.Key_M:
            self.modeToggled.emit()
            event.accept()
        elif event.key() == Qt.Key.Key_D:
            self.undoRequested.emit()
            event.accept()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event):
        self.setFocus()

        # ロック中の場合: 左クリックで点または線の座標を指定
        # 右クリック: ホイールモードをループ切り替え（ロック状態に関わらず）
        if event.button() == Qt.MouseButton.RightButton:
            self.cycle_wheel_mode()
            event.accept()
            return

        if self.is_locked:
            if event.button() == Qt.MouseButton.LeftButton:
                self._handle_point_click(event.position())
            super().mousePressEvent(event)
            return

        # 通常時: 左クリックでドラッグ開始
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = True
            pos = event.position()
            self.last_mouse_x = pos.x()
            self.last_mouse_y = pos.y()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def _handle_point_click(self, pos: QPointF):
        """クリック位置が画像領域内の場合、画像内ピクセル座標を計算してシグナル発行"""
        if self.render_rect is None or self.dst_w <= 0 or self.dst_h <= 0:
            return

        draw_x, draw_y, scaled_w, scaled_h = self.render_rect
        px = pos.x()
        py = pos.y()

        if draw_x <= px <= draw_x + scaled_w and draw_y <= py <= draw_y + scaled_h:
            img_x = (px - draw_x) * self.dst_w / scaled_w
            img_y = (py - draw_y) * self.dst_h / scaled_h
            self.pointClicked.emit(img_x, img_y)

    def mouseMoveEvent(self, event):
        if self.is_locked:
            super().mouseMoveEvent(event)
            return

        if self.is_dragging and (event.buttons() & Qt.MouseButton.LeftButton):
            pos = event.position()
            dx = pos.x() - self.last_mouse_x
            dy = pos.y() - self.last_mouse_y
            self.last_mouse_x = pos.x()
            self.last_mouse_y = pos.y()

            self.dragged.emit(dx, dy)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.is_dragging:
            self.is_dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(event)

    def cycle_wheel_mode(self):
        """ホイールモードをループ切り替えして通知"""
        modes = self.WHEEL_MODES
        idx = modes.index(self.wheel_mode)
        self.wheel_mode = modes[(idx + 1) % len(modes)]
        self.wheelModeChanged.emit(self.wheel_mode)

    def wheelEvent(self, event):
        """
        ホイール入力の分岐 (キー修飾なし):
        self.wheel_mode に従って縦横両方 / 縦のみ / 横のみ を切り替える
        """
        if self.is_locked:
            event.accept()
            return

        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return

        if self.wheel_mode == "v":
            self.wheelVertical.emit(delta)
        elif self.wheel_mode == "h":
            self.wheelHorizontal.emit(delta)
        else:
            self.wheelBoth.emit(delta)

        event.accept()

    def _extend_line_to_rect(self, sx, sy, ex, ey, rx0, ry0, rx1, ry1):
        """
        直線 (sx,sy)-(ex,ey) を矩形 (rx0,ry0)-(rx1,ry1) の端まで延長した2点を返す。
        矩形内に交点が2点取れない場合は None を返す。
        """
        dx = ex - sx
        dy = ey - sy
        if abs(dx) < 0.001 and abs(dy) < 0.001:
            return None

        # 4辺との t パラメータ（直線: P = (sx,sy) + t*(dx,dy)）
        ts = []
        if abs(dx) > 0.001:
            ts.append((rx0 - sx) / dx)
            ts.append((rx1 - sx) / dx)
        if abs(dy) > 0.001:
            ts.append((ry0 - sy) / dy)
            ts.append((ry1 - sy) / dy)

        pts = []
        for t in ts:
            x = sx + t * dx
            y = sy + t * dy
            if rx0 - 0.5 <= x <= rx1 + 0.5 and ry0 - 0.5 <= y <= ry1 + 0.5:
                pts.append((x, y))

        if len(pts) < 2:
            return None
        # 最も遠い2点を選択
        pts.sort(key=lambda p: p[0] * dx + p[1] * dy)
        return pts[0], pts[-1]

    # --------------------------------------------------------------------------
    # 描画処理
    # --------------------------------------------------------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect()

        # 背景塗りつぶし
        painter.fillRect(rect, QColor("#121214"))

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

            self.render_rect = (float(draw_x), float(draw_y), float(scaled.width()), float(scaled.height()))

            # ------------------------------------------------------------------
            # ロック中: 点・線の描画
            # ------------------------------------------------------------------
            if self.is_locked:
                painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                scale_x = scaled.width() / pm.width()
                scale_y = scaled.height() / pm.height()

                # 1. 線の描画 (確定済みの線)
                for line in self.lines:
                    idx = line["index"]
                    color = self.ANNOTATION_COLORS[idx % len(self.ANNOTATION_COLORS)]

                    sx = draw_x + line["start"]["img_x"] * scale_x
                    sy = draw_y + line["start"]["img_y"] * scale_y
                    ex = draw_x + line["end"]["img_x"] * scale_x
                    ey = draw_y + line["end"]["img_y"] * scale_y

                    # 線を矩形端まで延長して描画
                    extended = self._extend_line_to_rect(
                        sx, sy, ex, ey,
                        draw_x, draw_y, draw_x + scaled.width(), draw_y + scaled.height()
                    )
                    if extended:
                        p1_ext = QPointF(*extended[0])
                        p2_ext = QPointF(*extended[1])
                        painter.setPen(QPen(color, 1.5, Qt.PenStyle.SolidLine))
                        painter.drawLine(p1_ext, p2_ext)

                    # 始点・端点マーカー (半径4px)
                    for px_c, py_c in [(sx, sy), (ex, ey)]:
                        center = QPointF(px_c, py_c)
                        painter.setPen(QPen(QColor(255, 255, 255), 1.5))
                        painter.setBrush(color)
                        painter.drawEllipse(center, 4.0, 4.0)

                    # 線ラベル
                    mid_x = (sx + ex) / 2.0
                    mid_y = (sy + ey) / 2.0
                    font_lbl = QFont("Segoe UI", 8)
                    font_lbl.setBold(True)
                    painter.setFont(font_lbl)
                    painter.setPen(color)
                    painter.drawText(int(mid_x + 5), int(mid_y - 5), f"L{idx}")

                # 始点仮配置中のマーカー（端点待ち）
                if self.pending_line_start is not None:
                    sx = draw_x + self.pending_line_start["img_x"] * scale_x
                    sy = draw_y + self.pending_line_start["img_y"] * scale_y
                    center = QPointF(sx, sy)
                    painter.setPen(QPen(QColor(255, 255, 255), 2))
                    painter.setBrush(QColor("#ffd600"))
                    painter.drawEllipse(center, 6.0, 6.0)
                    font_lbl = QFont("Segoe UI", 8)
                    font_lbl.setBold(True)
                    painter.setFont(font_lbl)
                    painter.setPen(QColor("#ffd600"))
                    painter.drawText(int(sx + 8), int(sy + 3), "始点 (端点をクリック)")

                # 2. 点の描画 (点選択モード)
                for pt in self.points:
                    color = self.ANNOTATION_COLORS[pt["index"] % len(self.ANNOTATION_COLORS)]
                    sx = draw_x + pt["img_x"] * scale_x
                    sy = draw_y + pt["img_y"] * scale_y
                    center = QPointF(sx, sy)

                    # 外側円（半径5px）
                    painter.setPen(QPen(QColor(255, 255, 255), 1.5))
                    painter.setBrush(color)
                    painter.drawEllipse(center, 5.0, 5.0)

                    # 内側の中心黒点 (半径1.5px)
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QColor(0, 0, 0))
                    painter.drawEllipse(center, 1.5, 1.5)
        else:
            self.render_rect = None
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(QColor("#71717a"))
            font_title = QFont("Segoe UI", 13)
            font_title.setBold(True)
            painter.setFont(font_title)
            painter.drawText(
                rect.adjusted(0, -15, 0, -15),
                Qt.AlignmentFlag.AlignCenter,
                "📁 ここに全方位画像 (.jpg / .png) をドラッグ＆ドロップ"
            )

            font_sub = QFont("Segoe UI", 10)
            painter.setFont(font_sub)
            painter.setPen(QColor("#52525b"))
            painter.drawText(
                rect.adjusted(0, 25, 0, 25),
                Qt.AlignmentFlag.AlignCenter,
                "または上部タブの「画像選択」から指定してください"
            )

        # ----------------------------------------------------------------------
        # HUDオーバーレイ表示 (画面左上)
        # ----------------------------------------------------------------------
        if self.has_image:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(QColor(255, 255, 255, 40))
            painter.setBrush(QColor(18, 18, 20, 200))

            badge_h = 135 if self.is_locked else 115
            badge_rect = QRect(15, 15, 280, badge_h)
            painter.drawRoundedRect(badge_rect, 6, 6)

            font_hud = QFont("Segoe UI", 9)
            painter.setFont(font_hud)
            painter.setPen(QColor("#e4e4e7"))
            painter.drawText(25, 35, f"Yaw (水平角)   : {self.yaw_deg:6.1f}°")
            painter.drawText(25, 55, f"Pitch (垂直角) : {self.pitch_deg:6.1f}°")
            painter.drawText(25, 75, f"横FOV (水平)   : {self.fov_w_deg:6.1f}°")
            painter.drawText(25, 95, f"縦FOV (垂直)   : {self.fov_h_deg:6.1f}°")
            painter.drawText(25, 115, f"出力サイズ     : {self.dst_w} x {self.dst_h} px")

            # ロック中インジケータ
            if self.is_locked:
                font_lock = QFont("Segoe UI", 9)
                font_lock.setBold(True)
                painter.setFont(font_lock)
                painter.setPen(QColor("#ff5252"))
                mode_str = "点選択" if self.select_mode == "point" else "線選択"
                count_str = f"{len(self.points)}点" if self.select_mode == "point" else f"{len(self.lines)}本"
                painter.drawText(25, 135, f"🔒 ロック中 [{mode_str}: {count_str} | M:切替]")

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
    lockToggled = pyqtSignal()
    modeToggled = pyqtSignal()
    undoRequested = pyqtSignal()
    savePointsClicked = pyqtSignal()
    saveLinesClicked = pyqtSignal()
    saveIntrinsicsClicked = pyqtSignal()
    saveViewParamsClicked = pyqtSignal()
    fileDropped = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("全方位画像ビューワー (Omni-directional Image Viewer)")
        self.resize(1080, 720)
        self.setStyleSheet(MODERN_STYLE)

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
        self.line_path.setPlaceholderText("全方位画像ファイル (.jpg, .jpeg, .png) のパスを選択またはここにドロップ...")
        self.line_path.returnPressed.connect(self._on_load_button_clicked)

        btn_browse = QPushButton("参照...")
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
        tab_controls_layout.setSpacing(8)

        # 横FOVコントロール
        lbl_fov_w = QLabel("横FOV:")
        self.spin_fov_w = QDoubleSpinBox()
        self.spin_fov_w.setRange(10.0, 140.0)
        self.spin_fov_w.setSingleStep(2.0)
        self.spin_fov_w.setValue(60.0)
        self.spin_fov_w.setSuffix("°")
        self.spin_fov_w.valueChanged.connect(self._on_spin_fov_w_changed)

        # 縦FOVコントロール
        lbl_fov_h = QLabel("縦FOV:")
        self.spin_fov_h = QDoubleSpinBox()
        self.spin_fov_h.setRange(10.0, 140.0)
        self.spin_fov_h.setSingleStep(2.0)
        self.spin_fov_h.setValue(60.0)
        self.spin_fov_h.setSuffix("°")
        self.spin_fov_h.valueChanged.connect(self._on_spin_fov_h_changed)

        # ロックボタン
        self.btn_lock = QPushButton("視点ロック (L)")
        self.btn_lock.setCheckable(True)
        self.btn_lock.setToolTip("視点・画角を固定し、点・線の配置を可能にします (L / Space)")
        self.btn_lock.clicked.connect(self.lockToggled.emit)

        # ホイールモード切り替えボタン
        self.btn_wheel_mode = QPushButton("ホイール: 縦横")
        self.btn_wheel_mode.setToolTip("ホイール操作で変更するFOVを切り替えます（縦横→縦→横→縦横…）")
        self.btn_wheel_mode.clicked.connect(self._on_wheel_mode_clicked)

        # モード切り替えボタン
        self.btn_mode = QPushButton("モード: 点 (M)")
        self.btn_mode.setEnabled(False)
        self.btn_mode.setToolTip("点選択モードと線選択モードを切り替えます (Mキー)")
        self.btn_mode.clicked.connect(self.modeToggled.emit)

        # 1つ戻すボタン (Undo)
        self.btn_undo = QPushButton("戻す (D)")
        self.btn_undo.setEnabled(False)
        self.btn_undo.setToolTip("直前に配置した点または線を削除します (Dキー)")
        self.btn_undo.clicked.connect(self.undoRequested.emit)

        # 保存ボタン
        self.btn_save_pts = QPushButton("点保存")
        self.btn_save_pts.setEnabled(False)
        self.btn_save_pts.setToolTip("選択した点座標（x y 焦点距離）を.dat形式で保存")
        self.btn_save_pts.clicked.connect(self.savePointsClicked.emit)

        self.btn_save_lines = QPushButton("線保存")
        self.btn_save_lines.setEnabled(False)
        self.btn_save_lines.setToolTip("選択した線の始点・端点（x y 焦点距離）を交互にスタックして.dat形式で保存")
        self.btn_save_lines.clicked.connect(self.saveLinesClicked.emit)

        self.btn_save_intrinsics = QPushButton("内部パラメータ保存")
        self.btn_save_intrinsics.setEnabled(False)
        self.btn_save_intrinsics.setToolTip("ロック中の内部パラメータ行列 (3x3) を半角スペース区切りの.dat形式で保存")
        self.btn_save_intrinsics.clicked.connect(self.saveIntrinsicsClicked.emit)

        self.btn_save_view_params = QPushButton("視点情報保存")
        self.btn_save_view_params.setEnabled(False)
        self.btn_save_view_params.setToolTip("ロック中の水平角 垂直角 横FOV 縦FOV 出力幅 出力高を半角スペース区切りで保存")
        self.btn_save_view_params.clicked.connect(self.saveViewParamsClicked.emit)

        # リセットボタン
        self.btn_reset = QPushButton("視点リセット")
        self.btn_reset.clicked.connect(self.resetClicked.emit)

        # 操作説明ラベル
        self.lbl_help = QLabel("操作: [ドラッグ] 視線 | [ホイール] FOV（モードボタンで縦横選択） | [L] ロック")
        self.lbl_help.setStyleSheet("color: #71717a; font-weight: bold; margin-left: 6px;")

        tab_controls_layout.addWidget(lbl_fov_w)
        tab_controls_layout.addWidget(self.spin_fov_w)
        tab_controls_layout.addWidget(lbl_fov_h)
        tab_controls_layout.addWidget(self.spin_fov_h)
        tab_controls_layout.addWidget(self.btn_lock)
        tab_controls_layout.addWidget(self.btn_wheel_mode)
        tab_controls_layout.addWidget(self.btn_mode)
        tab_controls_layout.addWidget(self.btn_undo)
        tab_controls_layout.addWidget(self.btn_save_pts)
        tab_controls_layout.addWidget(self.btn_save_lines)
        tab_controls_layout.addWidget(self.btn_save_intrinsics)
        tab_controls_layout.addWidget(self.btn_save_view_params)
        tab_controls_layout.addWidget(self.btn_reset)
        tab_controls_layout.addWidget(self.lbl_help)
        tab_controls_layout.addStretch(1)

        self.tab_widget.addTab(tab_controls, "操作・設定")

        # ----------------------------------------------------------------------
        # 中央：ビューワーキャンバス
        # ----------------------------------------------------------------------
        self.viewer = ViewerWidget()
        self.viewer.fileDropped.connect(self.fileDropped.emit)
        self.viewer.wheelModeChanged.connect(self._update_wheel_mode_button)
        main_layout.addWidget(self.viewer, stretch=1)

        # ----------------------------------------------------------------------
        # 下部：ステータスバー
        # ----------------------------------------------------------------------
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("画像が選択されていません（画像をドラッグ＆ドロップして開くことができます）")

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

    def _on_wheel_mode_clicked(self):
        """ホイールモードをループ切り替えしてボタン表示を更新"""
        self.viewer.cycle_wheel_mode()
        self._update_wheel_mode_button(self.viewer.wheel_mode)

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

    def set_mode_display(self, mode: str):
        """モード表示（点 / 線）を更新"""
        self.viewer.set_select_mode(mode)
        if mode == "point":
            self.btn_mode.setText("モード: 点 (M)")
        else:
            self.btn_mode.setText("モード: 線 (M)")

    def _update_wheel_mode_button(self, wheel_mode: str):
        """ホイールモードボタンのラベルを更新"""
        labels = {"both": "縦横", "v": "縦のみ", "h": "横のみ"}
        self.btn_wheel_mode.setText(f"ホイール: {labels.get(wheel_mode, wheel_mode)}")

    def set_lock_state(self, is_locked: bool):
        """UIコントロールをロック状態に合わせて切り替え"""
        self._updating_ui = True
        self.btn_lock.setChecked(is_locked)
        self.btn_mode.setEnabled(is_locked)
        self.btn_undo.setEnabled(is_locked)
        self.btn_save_pts.setEnabled(is_locked)
        self.btn_save_lines.setEnabled(is_locked)
        self.btn_save_intrinsics.setEnabled(is_locked)
        self.btn_save_view_params.setEnabled(is_locked)

        if is_locked:
            self.btn_lock.setText("ロック解除 (L)")
            self.btn_lock.setStyleSheet("background-color: #b91c1c; color: white; font-weight: bold;")
            self.lbl_help.setText("操作: [クリック] 点/線指定  |  [M] モード切替  |  [D] 1つ戻す  |  [L] 解除")
            self.lbl_help.setStyleSheet("color: #f87171; font-weight: bold; margin-left: 6px;")
        else:
            self.btn_lock.setText("視点ロック (L)")
            self.btn_lock.setStyleSheet("")
            self.lbl_help.setText("操作: [ドラッグ] 視線 | [ホイール] FOV（モードボタンで縦横切替） | [L] ロック")
            self.lbl_help.setStyleSheet("color: #71717a; font-weight: bold; margin-left: 6px;")

        # ロック時はパラメータ変更ボタン・スピンボックスを無効化
        self.spin_fov_w.setEnabled(not is_locked)
        self.spin_fov_h.setEnabled(not is_locked)
        self.btn_reset.setEnabled(not is_locked)
        self._updating_ui = False
