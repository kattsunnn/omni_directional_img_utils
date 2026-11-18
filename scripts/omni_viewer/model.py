"""
全方位画像ビューワー - Model (model.py)

全方位画像データ、変換パラメータ（Yaw, Pitch, 横FOV, 縦FOV）、
およびE2Pによる透視投影変換ビジネスロジックを管理する。
"""

import os
import cv2
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from omni_directional_img_utils.e2p import E2P


class OmniViewerModel(QObject):
    """
    全方位画像ビューワーのデータモデル

    シグナル:
    - imageLoaded(file_path, width, height): 画像読み込み成功時
    - viewChanged(yaw, pitch, fov_w, fov_h, rendered_img): パラメータ更新・再レンダリング時
    - errorMessage(message): エラー発生時
    """
    imageLoaded = pyqtSignal(str, int, int)
    viewChanged = pyqtSignal(float, float, float, float, object)
    errorMessage = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.file_path: str = ""
        self.src_img: np.ndarray | None = None
        self.e2p: E2P | None = None
        self.rendered_img: np.ndarray | None = None

        # 視角パラメータ (度)
        self.yaw_deg: float = 0.0       # 水平回転角 (0〜360°)
        self.pitch_deg: float = 0.0     # 垂直回転角 (-89〜+89°)
        self.fov_w_deg: float = 60.0    # 横FOV (水平画角)
        self.fov_h_deg: float = 60.0    # 縦FOV (垂直画角)

        # パラメータ制限値
        self.min_fov: float = 10.0
        self.max_fov: float = 140.0
        self.min_pitch: float = -89.0
        self.max_pitch: float = 89.0

        # マウス感度
        self.rotate_sensitivity: float = 0.3  # 度 / ピクセル
        self.fov_step: float = 3.0            # 度 / ホイールステップ

    def load_image(self, file_path: str) -> bool:
        """
        全方位画像ファイルを読み込む
        Windowsの日本語・特殊文字パスに対応するため np.fromfile を使用
        """
        if not os.path.isfile(file_path):
            self.errorMessage.emit(f"エラー: ファイルが見つかりません ({file_path})")
            return False

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png"]:
            self.errorMessage.emit(f"エラー: 対応していない拡張子です ({ext})")
            return False

        try:
            data = np.fromfile(file_path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception as e:
            self.errorMessage.emit(f"エラー: 画像読み込みに失敗しました ({e})")
            return False

        if img is None:
            self.errorMessage.emit(f"エラー: 画像をデコードできませんでした ({file_path})")
            return False

        self.file_path = file_path
        self.src_img = img
        h, w = img.shape[:2]
        self.e2p = E2P(w, h)

        self.update_perspective()
        self.imageLoaded.emit(file_path, w, h)
        return True

    def rotate_view(self, dx: float, dy: float):
        """マウス左ドラッグ移動量 (dx, dy) による視線角度 (Yaw, Pitch) の更新"""
        self.yaw_deg = (self.yaw_deg - dx * self.rotate_sensitivity) % 360.0
        self.pitch_deg = float(np.clip(
            self.pitch_deg - dy * self.rotate_sensitivity,
            self.min_pitch, self.max_pitch
        ))
        self.update_perspective()

    def change_fov_h_by_wheel(self, delta: int):
        """通常ホイール: 縦FOV（垂直画角）のみを独立して変更"""
        step = -self.fov_step if delta > 0 else self.fov_step
        self.set_fov_h(self.fov_h_deg + step)

    def change_fov_w_by_wheel(self, delta: int):
        """Shift + ホイール: 横FOV（水平画角）のみを独立して変更"""
        step = -self.fov_step if delta > 0 else self.fov_step
        self.set_fov_w(self.fov_w_deg + step)

    def set_fov_w(self, fov_w: float):
        """横FOV（水平画角）を設定"""
        self.fov_w_deg = float(np.clip(fov_w, self.min_fov, self.max_fov))
        self.update_perspective()

    def set_fov_h(self, fov_h: float):
        """縦FOV（垂直画角）を設定"""
        self.fov_h_deg = float(np.clip(fov_h, self.min_fov, self.max_fov))
        self.update_perspective()

    def reset_view(self):
        """視点と画角を初期値にリセット (Yaw: 0°, Pitch: 0°, 横FOV: 60°, 縦FOV: 60°)"""
        self.yaw_deg = 0.0
        self.pitch_deg = 0.0
        self.fov_w_deg = 60.0
        self.fov_h_deg = 60.0
        self.update_perspective()

    def update_perspective(self):
        """E2Pによる透視投影変換を実行し、viewChanged シグナルを発行"""
        if self.src_img is None or self.e2p is None:
            self.rendered_img = None
        else:
            self.e2p.generate_map(
                self.fov_w_deg, self.fov_h_deg,
                self.yaw_deg, self.pitch_deg, 0
            )
            self.rendered_img = self.e2p.generate_img(self.src_img)

        self.viewChanged.emit(
            self.yaw_deg, self.pitch_deg,
            self.fov_w_deg, self.fov_h_deg,
            self.rendered_img
        )

