"""
全方位画像ビューワー (omni_viewer.py)

要件:
- PyQt6を使用した全方位画像のビューワー
- 画面上部にタブがあり、全方位画像（.jpg, .png）のパスを選択可能
- 透視投影画像の表示（E2Pによる透視投影変換）
- 左クリックホールド移動で生成角度（Yaw, Pitch）を変更
- マウスホイールで生成領域（FOV / 画角）を変更
- 右クリックで1〜4点を順番に選択・配置でき、視点移動や画角変更をしても画像上の同じ位置に留まる
- 4点選択したときに選択領域が四角く（ポリゴン＋半透明塗りつぶし）表示される
- 削除（Undo）機能により、4 -> 3 -> 2 -> 1 と戻りながら再選択できる
"""

import sys
import os
import numpy as np
import cv2

from PyQt6 import QtCore, QtGui
from PyQt6.QtCore import Qt, pyqtSignal, QRect, QPointF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QPen, QPolygonF
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
# 順投影ヘルパー（全方位座標 -> 透視投影画面座標）
# ==============================================================================
def project_omni_point_to_perspective(
    omni_u: float,
    omni_v: float,
    src_w: int,
    src_h: int,
    fov_w_deg: float,
    fov_h_deg: float,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float = 0.0,
    scale: float = 1.0
) -> tuple[float, float] | None:
    """
    全方位画像上の座標 (omni_u, omni_v) を、指定画角・回転の透視投影画像上の画素座標 (dst_u, dst_v) に投影する。
    カメラの背面にある場合や視野外の場合は None を返す。
    """
    # 1. 全方位画像上の座標から球面角度 (theta, phi) を計算
    theta = (omni_u / src_w) * (2.0 * np.pi) - np.pi
    phi = (omni_v / src_h) * np.pi - (np.pi / 2.0)

    # 2. ワールド座標系での3次元方向単位ベクトル
    # E2P.generate_map の theta = atan2(Xx, Xz), phi = atan2(Xy, sqrt(Xx^2+Xz^2)) に対応
    Xx = np.cos(phi) * np.sin(theta)
    Xy = np.sin(phi)
    Xz = np.cos(phi) * np.cos(theta)
    p_world = np.array([Xx, Xy, Xz], dtype=np.float64)

    # 3. 回転行列の計算 (E2P.generate_map と同一の合成回転)
    yaw_rad = np.deg2rad(yaw_deg)
    pitch_rad = np.deg2rad(pitch_deg)
    roll_rad = np.deg2rad(roll_deg)
    R = np.dot(E2P.rotation_y(yaw_rad), E2P.rotation_x(pitch_rad))
    if roll_deg != 0.0:
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        Ri = E2P.rotation_by_axis(np.dot(R, axis), roll_rad)
        R = np.dot(Ri, R)

    # 4. カメラ座標系への逆変換 (回転行列 R の転置行列 R.T)
    p_cam = np.dot(R.T, p_world)
    xc, yc, zc = p_cam[0], p_cam[1], p_cam[2]

    # カメラの後方にある場合は視界外
    if zc <= 1e-6:
        return None

    # 5. 透視投影（ピンホールカメラモデル）
    fov_w_rad = np.deg2rad(fov_w_deg)
    fov_h_rad = np.deg2rad(fov_h_deg)
    dst_w = int((src_w / np.pi) * np.tan(fov_w_rad / 2.0))
    dst_h = int(2.0 * np.tan(fov_h_rad / 2.0) * (src_h / np.pi))
    f = src_w / (2.0 * np.pi)
    f_z = f * scale

    dst_u = (xc / zc) * f_z + dst_w * 0.5
    dst_v = (yc / zc) * f_z + dst_h * 0.5

    # 透視投影画像の境界内にあるか判定
    if 0.0 <= dst_u < dst_w and 0.0 <= dst_v < dst_h:
        return float(dst_u), float(dst_v)
    return None


# ==============================================================================
# ビューワー描画ウィジェット (ViewerWidget)
# ==============================================================================
class ViewerWidget(QWidget):
    """
    画像表示およびマウス操作（ドラッグによる角度変更、ホイールによるFOV変更、右クリックによる4点選択・追従）を処理するウィジェット
    """
    # 視角パラメータ変更シグナル: (yaw, pitch, fov)
    viewChanged = pyqtSignal(float, float, float)
    # 点リスト変更シグナル: list[tuple[float, float]]
    pointsChanged = pyqtSignal(list)

    # 各点（1〜4）の識別カラー
    POINT_COLORS = [
        QColor("#00e5ff"),  # 1: シアン
        QColor("#76ff03"),  # 2: ライムグリーン
        QColor("#ffd600"),  # 3: イエロー
        QColor("#ff1744"),  # 4: レッド
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # 全方位原画 (BGR)
        self.src_img: np.ndarray | None = None
        # E2P インスタンス
        self.e2p: E2P | None = None
        # 表示用QPixmap
        self.display_pixmap: QPixmap | None = None

        # 透視投影パラメータ
        self.yaw_deg: float = 0.0     # 水平回転角
        self.pitch_deg: float = 0.0   # 垂直回転角
        self.fov_deg: float = 60.0    # 画角 (生成領域)

        # 選択された点リスト (最大4点): 各要素は全方位画像上の (omni_u, omni_v)
        self.selected_points: list[tuple[float, float]] = []

        # ウィジェット内の画像描画矩形情報: (draw_x, draw_y, scaled_w, scaled_h)
        self.render_rect: tuple[float, float, float, float] | None = None

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
        self.selected_points.clear()
        if img is not None:
            self.e2p = E2P(img.shape[1], img.shape[0])
        else:
            self.e2p = None
        self.update_perspective_view()
        self.pointsChanged.emit(self.selected_points)

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

    def undo_point(self):
        """直前に選択した点を1つ削除して1ステップ戻る"""
        if self.selected_points:
            self.selected_points.pop()
            self.update()
            self.pointsChanged.emit(self.selected_points)
            self.viewChanged.emit(self.yaw_deg, self.pitch_deg, self.fov_deg)

    def clear_points(self):
        """選択されたすべての点をクリア"""
        if self.selected_points:
            self.selected_points.clear()
            self.update()
            self.pointsChanged.emit(self.selected_points)
            self.viewChanged.emit(self.yaw_deg, self.pitch_deg, self.fov_deg)

    def update_perspective_view(self):
        """透視投影画像を生成して再描画"""
        if self.src_img is None or self.e2p is None:
            self.display_pixmap = None
            self.update()
            return

        # 透視投影画像の生成
        self.e2p.generate_map(
            self.fov_deg, self.fov_deg,
            self.yaw_deg, self.pitch_deg, 0
        )
        rendered = self.e2p.generate_img(self.src_img)

        if rendered is not None:
            qimg = cv_img_to_qimage(rendered)
            self.display_pixmap = QPixmap.fromImage(qimg)
        else:
            self.display_pixmap = None

        self.update()

    # --------------------------------------------------------------------------
    # キーボードショートカット
    # --------------------------------------------------------------------------
    def keyPressEvent(self, event):
        """Backspace / Delete / Ctrl+Z で1点戻す"""
        if event.key() in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            self.undo_point()
            event.accept()
        elif (event.modifiers() & Qt.KeyboardModifier.ControlModifier) and event.key() == Qt.Key.Key_Z:
            self.undo_point()
            event.accept()
        else:
            super().keyPressEvent(event)

    # --------------------------------------------------------------------------
    # マウスイベント処理
    # --------------------------------------------------------------------------
    def mousePressEvent(self, event):
        """クリック処理（左クリック：視点回転ドラッグ開始、右クリック：1〜4点の選択）"""
        self.setFocus()
        # 左クリック: 視線ドラッグ開始
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = True
            pos = event.position()
            self.last_mouse_x = pos.x()
            self.last_mouse_y = pos.y()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

        # 右クリック: 点の選択・配置 (最大4点)
        elif event.button() == Qt.MouseButton.RightButton:
            self._handle_right_click(event)

        super().mousePressEvent(event)

    def _handle_right_click(self, event):
        """右クリック位置から全方位画像上の座標を取得して点を追加・更新"""
        if self.src_img is None or self.e2p is None or self.render_rect is None:
            return
        if self.e2p.map_u is None or self.e2p.map_v is None:
            return

        pos = event.position()
        draw_x, draw_y, scaled_w, scaled_h = self.render_rect

        # クリック位置が画像描画領域内にあるか判定
        if not (draw_x <= pos.x() <= draw_x + scaled_w and draw_y <= pos.y() <= draw_y + scaled_h):
            return

        # 透視投影画像 (dst_img) 内の画素インデックスを計算
        dst_h, dst_w = self.e2p.map_u.shape
        img_x = int((pos.x() - draw_x) * dst_w / scaled_w)
        img_y = int((pos.y() - draw_y) * dst_h / scaled_h)

        img_x = int(np.clip(img_x, 0, dst_w - 1))
        img_y = int(np.clip(img_y, 0, dst_h - 1))

        # map_u, map_v から全方位画像上の対応座標 (U, V) を直接取得
        omni_u = float(self.e2p.map_u[img_y, img_x])
        omni_v = float(self.e2p.map_v[img_y, img_x])

        if len(self.selected_points) < 4:
            # 1〜4点目を順に追加
            self.selected_points.append((omni_u, omni_v))
        else:
            # 既に4点ある場合は4点目の位置を更新（再調整）
            self.selected_points[-1] = (omni_u, omni_v)

        self.update()
        self.pointsChanged.emit(self.selected_points)
        self.viewChanged.emit(self.yaw_deg, self.pitch_deg, self.fov_deg)

    def mouseMoveEvent(self, event):
        """左クリックホールド移動で角度 (Yaw / Pitch) を変更"""
        if self.is_dragging and (event.buttons() & Qt.MouseButton.LeftButton):
            pos = event.position()
            dx = pos.x() - self.last_mouse_x
            dy = pos.y() - self.last_mouse_y

            # 移動量に応じて角度を更新
            # 水平方向: ドラッグでYawを回転
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
        """画像、選択された4点と四角形領域、およびHUDの描画"""
        painter = QPainter(self)
        rect = self.rect()

        # 背景塗りつぶし
        painter.fillRect(rect, QColor("#1e1e1e"))

        visible_points_info: list[tuple[int, float, float, bool]] = []

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

            # 画像描画領域を記録（クリック判定用）
            self.render_rect = (float(draw_x), float(draw_y), float(scaled.width()), float(scaled.height()))

            # ------------------------------------------------------------------
            # 選択された点 (1〜4点) の追従投影座標を計算
            # ------------------------------------------------------------------
            if self.selected_points and self.src_img is not None:
                src_h, src_w = self.src_img.shape[:2]
                scale_x = scaled.width() / pm.width()
                scale_y = scaled.height() / pm.height()

                # 各点の画面上座標リスト: QPointF または None (視野外)
                widget_pts: list[QPointF | None] = []

                for idx, (omni_u, omni_v) in enumerate(self.selected_points):
                    proj = project_omni_point_to_perspective(
                        omni_u, omni_v,
                        src_w, src_h,
                        self.fov_deg, self.fov_deg,
                        self.yaw_deg, self.pitch_deg,
                        0.0
                    )
                    if proj is not None:
                        dst_u, dst_v = proj
                        wx = draw_x + dst_u * scale_x
                        wy = draw_y + dst_v * scale_y
                        widget_pts.append(QPointF(wx, wy))
                        visible_points_info.append((idx + 1, omni_u, omni_v, True))
                    else:
                        widget_pts.append(None)
                        visible_points_info.append((idx + 1, omni_u, omni_v, False))

                painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

                # --------------------------------------------------------------
                # 4点選択時の四角形領域（塗りつぶし＋枠線）または線分を描画
                # --------------------------------------------------------------
                num_pts = len(self.selected_points)

                # 4点すべてが視野内の場合、半透明の四角形領域を塗りつぶし描画
                if num_pts == 4 and all(pt is not None for pt in widget_pts):
                    poly = QPolygonF([widget_pts[0], widget_pts[1], widget_pts[2], widget_pts[3]])
                    # 半透明シアンの塗りつぶし
                    painter.setBrush(QColor(0, 229, 255, 45))
                    painter.setPen(QPen(QColor(0, 229, 255, 220), 2, Qt.PenStyle.SolidLine))
                    painter.drawPolygon(poly)
                else:
                    # 4点未満、または一部視野外の場合は見える線分を描画
                    painter.setPen(QPen(QColor(0, 229, 255, 200), 2, Qt.PenStyle.DashLine))
                    for i in range(num_pts - 1):
                        p1, p2 = widget_pts[i], widget_pts[i + 1]
                        if p1 is not None and p2 is not None:
                            painter.drawLine(p1, p2)
                    # 4点選択時は4点目と1点目も結ぶ
                    if num_pts == 4 and widget_pts[3] is not None and widget_pts[0] is not None:
                        painter.drawLine(widget_pts[3], widget_pts[0])

                # --------------------------------------------------------------
                # 各マーカー（番号バッジ付き）の描画
                # --------------------------------------------------------------
                for idx, pt in enumerate(widget_pts):
                    if pt is None:
                        continue
                    color = self.POINT_COLORS[idx % len(self.POINT_COLORS)]
                    omni_u, omni_v = self.selected_points[idx]

                    # 外側円（各色・白枠）
                    painter.setPen(QPen(QColor(255, 255, 255), 2))
                    painter.setBrush(color)
                    painter.drawEllipse(pt, 9.0, 9.0)

                    # 番号テキスト (1〜4)
                    painter.setPen(QColor(0, 0, 0))
                    font_num = QFont()
                    font_num.setPointSize(8)
                    font_num.setBold(True)
                    painter.setFont(font_num)
                    text_num = str(idx + 1)
                    # 文字を円の中央に配置
                    painter.drawText(
                        QRect(int(pt.x() - 9), int(pt.y() - 9), 18, 18),
                        Qt.AlignmentFlag.AlignCenter,
                        text_num
                    )

                    # 座標ラベル
                    painter.setPen(QColor(255, 255, 255))
                    font_lbl = QFont()
                    font_lbl.setPointSize(8)
                    font_lbl.setBold(True)
                    painter.setFont(font_lbl)
                    painter.drawText(
                        int(pt.x() + 12),
                        int(pt.y() - 4),
                        f"P{idx + 1} ({omni_u:.1f}, {omni_v:.1f})"
                    )

        else:
            self.render_rect = None
            # 未読み込み時のプレースホルダーメッセージ
            painter.setPen(QColor("#888888"))
            font = QFont()
            font.setPointSize(13)
            painter.setFont(font)
            text = "上部タブの「画像選択」から全方位画像 (.jpg / .png) を指定してください"
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

        # ----------------------------------------------------------------------
        # 画面左上 HUD オーバーレイ表示
        # ----------------------------------------------------------------------
        if self.src_img is not None:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(QColor(255, 255, 255, 200))
            painter.setBrush(QColor(0, 0, 0, 160))

            num_pts = len(self.selected_points)
            badge_h = 80 + (20 * num_pts)
            badge_rect = QRect(15, 15, 270, badge_h)
            painter.drawRoundedRect(badge_rect, 6, 6)

            font = QFont()
            font.setPointSize(9)
            painter.setFont(font)
            painter.drawText(25, 35, f"Yaw (水平角)   : {self.yaw_deg:6.1f}°")
            painter.drawText(25, 55, f"Pitch (垂直角) : {self.pitch_deg:6.1f}°")
            painter.drawText(25, 75, f"FOV (生成領域) : {self.fov_deg:6.1f}°  | 点: {num_pts}/4")

            # 各点情報の表示
            for i, (pt_num, u, v, is_vis) in enumerate(visible_points_info):
                y_pos = 95 + i * 20
                status_txt = "表示中" if is_vis else "視野外"
                status_col = self.POINT_COLORS[i % len(self.POINT_COLORS)] if is_vis else QColor(160, 160, 160)
                painter.setPen(status_col)
                painter.drawText(25, y_pos, f"点{pt_num}: (U:{u:6.1f}, V:{v:6.1f}) [{status_txt}]")

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
        tab_controls_layout.setSpacing(10)

        # 操作説明ラベル
        lbl_help = QLabel(
            "操作: [左ドラッグ] 視線変更  |  [ホイール] 画角  |  [右クリック] 点配置 (1〜4点)  |  [Backspace / 戻す] 1点削除"
        )
        lbl_help.setStyleSheet("color: #555555; font-weight: bold;")
        tab_controls_layout.addWidget(lbl_help)

        tab_controls_layout.addStretch(1)

        # 1点戻す (Undo) ボタン
        self.btn_undo_pt = QPushButton("↩ 1点戻す (削除)")
        self.btn_undo_pt.setToolTip("直前に選択した点を1つ削除して再選択できます (Backspace)")
        self.btn_undo_pt.clicked.connect(self._on_undo_point_clicked)
        tab_controls_layout.addWidget(self.btn_undo_pt)

        # 全点クリアボタン
        self.btn_clear_pt = QPushButton("全点をクリア")
        self.btn_clear_pt.clicked.connect(self._on_clear_points_clicked)
        tab_controls_layout.addWidget(self.btn_clear_pt)

        # リセットボタン
        btn_reset = QPushButton("視点リセット")
        btn_reset.clicked.connect(self._on_reset_clicked)
        tab_controls_layout.addWidget(btn_reset)

        self.tab_widget.addTab(tab_controls, "操作・設定")

        # ----------------------------------------------------------------------
        # 中央：ビューワーウィジェット
        # ----------------------------------------------------------------------
        self.viewer = ViewerWidget()
        self.viewer.viewChanged.connect(self._on_view_changed)
        self.viewer.pointsChanged.connect(self._on_points_changed)
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
            pts_cnt = len(self.viewer.selected_points)
            pts_str = f" | 選択中の点: {pts_cnt}/4"
            if pts_cnt == 4:
                pts_str += " (四角形選択完了)"
            self.status_bar.showMessage(
                f"ファイル: {filename} | "
                f"Yaw: {yaw:5.1f}° | Pitch: {pitch:5.1f}° | FOV: {fov:5.1f}°{pts_str}"
            )

    def _on_points_changed(self, points: list):
        """点リスト更新時のステータス表示"""
        filename = os.path.basename(self.current_image_path) if self.current_image_path else ""
        cnt = len(points)
        if cnt == 0:
            self.status_bar.showMessage(f"ファイル: {filename} | 点がクリアされました")
        elif cnt < 4:
            last_u, last_v = points[-1]
            self.status_bar.showMessage(
                f"ファイル: {filename} | 点{cnt} を配置: (U={last_u:.1f}, V={last_v:.1f}) [あと {4 - cnt} 点選択してください]"
            )
        else:
            self.status_bar.showMessage(
                f"ファイル: {filename} | 4点の選択が完了しました（四角形表示中）。[1点戻す] で直前の点を再選択できます。"
            )

    def _on_reset_clicked(self):
        """視点と画角をリセット"""
        self.viewer.reset_view()

    def _on_undo_point_clicked(self):
        """直前の点を1つ削除"""
        self.viewer.undo_point()

    def _on_clear_points_clicked(self):
        """全点をクリア"""
        self.viewer.clear_points()


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
