"""
全方位画像ビューワー - Model (model.py)

全方位画像データ、変換パラメータ（Yaw, Pitch, 横FOV, 縦FOV）、
ロック状態、点・線データ管理、E2Pによる透視投影変換ビジネスロジック、および.dat保存を担当する。
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
    - lockStateChanged(is_locked: bool): ロック状態変更時
    - modeChanged(mode: str): モード変更時 ("point" or "line")
    - pointsChanged(points: list): 配置点リスト更新時
    - linesChanged(lines: list, pending_start: dict | None): 線リスト更新時
    - errorMessage(message): エラー発生時
    - infoMessage(message): 情報メッセージ通知時
    """
    imageLoaded = pyqtSignal(str, int, int)
    viewChanged = pyqtSignal(float, float, float, float, object)
    lockStateChanged = pyqtSignal(bool)
    modeChanged = pyqtSignal(str)
    pointsChanged = pyqtSignal(list)
    linesChanged = pyqtSignal(list, object)
    errorMessage = pyqtSignal(str)
    infoMessage = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.file_path: str = ""
        self.src_img: np.ndarray | None = None
        self.src_img_w: int = 0
        self.src_img_h: int = 0
        self.e2p: E2P | None = None
        self.rendered_img: np.ndarray | None = None

        # 視角パラメータ (度)
        self.yaw_deg: float = 0.0       # 水平回転角 (0〜360°)
        self.pitch_deg: float = 0.0     # 垂直回転角 (-89〜+89°)
        self.fov_w_deg: float = 60.0    # 横FOV (水平画角 / Width)
        self.fov_h_deg: float = 60.0    # 縦FOV (垂直画角 / Height)

        # 操作ロックフラグ
        self.is_locked: bool = False

        # 選択モード: "point" (点選択) または "line" (線選択)
        self.select_mode: str = "point"

        # ロック中に配置された点リスト
        # 各要素: {"img_x": float, "img_y": float, "omni_u": float, "omni_v": float, "index": int}
        self.points: list[dict] = []

        # ロック中に配置された線リスト
        # 各要素: {"start": dict, "end": dict, "index": int}
        self.lines: list[dict] = []
        # 線選択中の始点仮配置データ (端点クリック待ち)
        self.pending_line_start: dict | None = None

        # パラメータ制限値
        self.min_fov: float = 10.0
        self.max_fov: float = 140.0
        self.min_pitch: float = -89.0
        self.max_pitch: float = 89.0

        # マウス感度
        self.rotate_sensitivity: float = 0.3  # 度 / ピクセル
        self.fov_step: float = 3.0            # 度 / ホイールステップ

    @property
    def focal_length(self) -> float:
        """全方位画像の横幅から焦点距離を計算 (ppi.py:L24と同様)"""
        if self.src_img_w > 0:
            return float(self.src_img_w / (2.0 * np.pi))
        return 0.0

    # --------------------------------------------------------------------------
    # ロック & モード管理
    # --------------------------------------------------------------------------
    def set_locked(self, locked: bool):
        """
        ロック状態を設定。
        ロック解除（locked=False）時は配置されていたすべての点および線を自動削除する。
        """
        if self.is_locked != locked:
            self.is_locked = locked
            if not self.is_locked:
                self.clear_all_annotations()
            self.lockStateChanged.emit(self.is_locked)

    def toggle_lock(self):
        """ロック状態をトグル反転"""
        self.set_locked(not self.is_locked)

    def toggle_mode(self):
        """Mキー押下時: 点選択モードと線選択モードを切り替え"""
        if self.select_mode == "point":
            self.select_mode = "line"
        else:
            self.select_mode = "point"
            # 線選択途中の始点があればキャンセル
            self.pending_line_start = None
            self.linesChanged.emit(self.lines, self.pending_line_start)

        self.modeChanged.emit(self.select_mode)

    # --------------------------------------------------------------------------
    # 点・線配置 & 削除 (Undo)
    # --------------------------------------------------------------------------
    def _create_point_dict(self, img_x: float, img_y: float) -> dict | None:
        """透視投影画像座標から全方位座標を求め辞書データを生成"""
        if self.e2p is None or self.e2p.map_u is None or self.e2p.map_v is None:
            return None

        dst_h, dst_w = self.e2p.map_u.shape
        ix = int(np.clip(round(img_x), 0, dst_w - 1))
        iy = int(np.clip(round(img_y), 0, dst_h - 1))

        omni_u = float(self.e2p.map_u[iy, ix])
        omni_v = float(self.e2p.map_v[iy, ix])

        return {
            "img_x": float(img_x),
            "img_y": float(img_y),
            "omni_u": omni_u,
            "omni_v": omni_v,
        }

    def handle_click_annotation(self, img_x: float, img_y: float):
        """
        ロック中のクリック処理:
        - 点モード: 点を1個追加
        - 線モード: 1回目で始点、2回目で端点を指定して線分を確定
        """
        if not self.is_locked:
            return

        pt_data = self._create_point_dict(img_x, img_y)
        if pt_data is None:
            return

        if self.select_mode == "point":
            pt_data["index"] = len(self.points) + 1
            self.points.append(pt_data)
            self.pointsChanged.emit(self.points)
        else:
            # 線モード
            if self.pending_line_start is None:
                # 始点仮配置
                pt_data["index"] = len(self.lines) + 1
                self.pending_line_start = pt_data
                self.linesChanged.emit(self.lines, self.pending_line_start)
            else:
                # 端点確定 -> 線を作成
                line_data = {
                    "start": self.pending_line_start,
                    "end": pt_data,
                    "index": len(self.lines) + 1
                }
                self.lines.append(line_data)
                self.pending_line_start = None
                self.linesChanged.emit(self.lines, self.pending_line_start)

    def undo_annotation(self):
        """
        Dキー押下時: 直前の要素を削除
        - 点モード: 直前の点を削除 (3 -> 2 -> 1)
        - 線モード: 始点仮配置中ならそれを取消、線確定後なら最後の線を削除
        """
        if not self.is_locked:
            return

        if self.select_mode == "point":
            if self.points:
                self.points.pop()
                self.pointsChanged.emit(self.points)
        else:
            # 線モード
            if self.pending_line_start is not None:
                self.pending_line_start = None
                self.linesChanged.emit(self.lines, self.pending_line_start)
            elif self.lines:
                self.lines.pop()
                self.linesChanged.emit(self.lines, self.pending_line_start)

    def clear_all_annotations(self):
        """すべての点・線を削除（ロック解除時等）"""
        self.points.clear()
        self.lines.clear()
        self.pending_line_start = None
        self.pointsChanged.emit(self.points)
        self.linesChanged.emit(self.lines, self.pending_line_start)

    # --------------------------------------------------------------------------
    # データ保存 (.dat形式)
    # --------------------------------------------------------------------------
    def save_points_dat(self, file_path: str) -> bool:
        """
        点データを.dat形式で保存:
        x半角スペースy半角スペース焦点距離
        点ごとに改行
        """
        if not self.points:
            self.errorMessage.emit("保存エラー: 保存する点がありません")
            return False

        fl = self.focal_length
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                for pt in self.points:
                    f.write(f"{pt['img_x']:.2f} {pt['img_y']:.2f} {fl:.2f}\n")
            self.infoMessage.emit(f"点を保存しました: {os.path.basename(file_path)} ({len(self.points)}点)")
            return True
        except Exception as e:
            self.errorMessage.emit(f"ファイル保存エラー: {e}")
            return False

    def save_lines_dat(self, file_path: str) -> bool:
        """
        線データを.dat形式で保存:
        始点と端点が交互にスタックされる形:
        始点x 始点y 焦点距離
        端点x 端点y 焦点距離
        """
        if not self.lines:
            self.errorMessage.emit("保存エラー: 保存する線がありません")
            return False

        fl = self.focal_length
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                for line in self.lines:
                    s = line["start"]
                    e = line["end"]
                    f.write(f"{s['img_x']:.2f} {s['img_y']:.2f} {fl:.2f}\n")
                    f.write(f"{e['img_x']:.2f} {e['img_y']:.2f} {fl:.2f}\n")
            self.infoMessage.emit(f"線を保存しました: {os.path.basename(file_path)} ({len(self.lines)}本)")
            return True
        except Exception as e:
            self.errorMessage.emit(f"ファイル保存エラー: {e}")
            return False

    def save_view_params_dat(self, file_path: str) -> bool:
        """
        現在の視点情報を.dat形式で保存:
        yaw pitch fov_w fov_h dst_w dst_h
        1行に数値を半角スペース区切りで書き出す。
        """
        dst_w = 0
        dst_h = 0
        if self.rendered_img is not None:
            dst_h, dst_w = self.rendered_img.shape[:2]

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(
                    f"{self.yaw_deg:.2f} {self.pitch_deg:.2f} "
                    f"{self.fov_w_deg:.2f} {self.fov_h_deg:.2f} "
                    f"{dst_w} {dst_h}\n"
                )
            self.infoMessage.emit(f"視点情報を保存しました: {os.path.basename(file_path)}")
            return True
        except Exception as e:
            self.errorMessage.emit(f"ファイル保存エラー: {e}")
            return False

    # --------------------------------------------------------------------------
    # 画像読み込み & 視角操作
    # --------------------------------------------------------------------------
    def load_image(self, file_path: str) -> bool:
        """全方位画像ファイルを読み込む"""
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
        self.src_img_w = w
        self.src_img_h = h
        self.e2p = E2P(w, h)
        self.clear_all_annotations()

        self.update_perspective()
        self.imageLoaded.emit(file_path, w, h)
        return True

    def rotate_view(self, dx: float, dy: float):
        """マウス左ドラッグ移動量 (dx, dy) による視線角度 (Yaw, Pitch) の更新"""
        if self.is_locked:
            return

        self.yaw_deg = (self.yaw_deg - dx * self.rotate_sensitivity) % 360.0
        self.pitch_deg = float(np.clip(
            self.pitch_deg - dy * self.rotate_sensitivity,
            self.min_pitch, self.max_pitch
        ))
        self.update_perspective()

    def change_fov_both_by_wheel(self, delta: int):
        """通常ホイール: 縦横両方のFOVに同じ数ずつ加算/減算する"""
        if self.is_locked:
            return

        step = -self.fov_step if delta > 0 else self.fov_step
        self.fov_w_deg = float(np.clip(self.fov_w_deg + step, self.min_fov, self.max_fov))
        self.fov_h_deg = float(np.clip(self.fov_h_deg + step, self.min_fov, self.max_fov))
        self.update_perspective()

    def change_fov_height_by_wheel(self, delta: int):
        """Shift + ホイール: 縦FOV (垂直画角 / Height) のみを変更"""
        if self.is_locked:
            return

        step = -self.fov_step if delta > 0 else self.fov_step
        self.set_fov_h(self.fov_h_deg + step)

    def change_fov_width_by_wheel(self, delta: int):
        """Ctrl + ホイール: 横FOV (水平画角 / Width) のみを変更"""
        if self.is_locked:
            return

        step = -self.fov_step if delta > 0 else self.fov_step
        self.set_fov_w(self.fov_w_deg + step)

    def set_fov_w(self, fov_w: float):
        """横FOV（水平画角 / Width）を設定"""
        if self.is_locked:
            return

        self.fov_w_deg = float(np.clip(fov_w, self.min_fov, self.max_fov))
        self.update_perspective()

    def set_fov_h(self, fov_h: float):
        """縦FOV（垂直画角 / Height）を設定"""
        if self.is_locked:
            return

        self.fov_h_deg = float(np.clip(fov_h, self.min_fov, self.max_fov))
        self.update_perspective()

    def reset_view(self):
        """視点と画角を初期値にリセット (Yaw: 0°, Pitch: 0°, 横FOV: 60°, 縦FOV: 60°)"""
        if self.is_locked:
            return

        self.yaw_deg = 0.0
        self.pitch_deg = 0.0
        self.fov_w_deg = 60.0
        self.fov_h_deg = 60.0
        self.clear_all_annotations()
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
