"""
全方位画像ビューワー - Controller (controller.py)

ModelとViewの仲介を行い、ユーザー操作イベントに応じたModelの更新、
およびModelの状態変化に応じたViewの描画更新を制御する。
"""

import os
from PyQt6.QtWidgets import QFileDialog

from model import OmniViewerModel
from view import OmniViewerWindow


class OmniViewerController:
    """
    全方位画像ビューワーのコントローラー
    """
    def __init__(self, model: OmniViewerModel, window: OmniViewerWindow):
        self.model = model
        self.window = window

        self._connect_view_signals()
        self._connect_model_signals()

    def _connect_view_signals(self):
        """Viewからのユーザー入力シグナルをハンドラへ接続"""
        # メインウィンドウからのシグナル
        self.window.browseClicked.connect(self._on_browse_clicked)
        self.window.loadRequested.connect(self._on_load_requested)
        self.window.fovWChanged.connect(self.model.set_fov_w)
        self.window.fovHChanged.connect(self.model.set_fov_h)
        self.window.resetClicked.connect(self.model.reset_view)

        # ViewerCanvas（マウス操作）からのシグナル
        self.window.viewer.dragged.connect(self.model.rotate_view)
        self.window.viewer.wheelVertical.connect(self.model.change_fov_h_by_wheel)
        self.window.viewer.wheelHorizontal.connect(self.model.change_fov_w_by_wheel)

    def _connect_model_signals(self):
        """Modelからのデータ更新シグナルをViewへ反映"""
        self.model.imageLoaded.connect(self._on_image_loaded)
        self.model.viewChanged.connect(self._on_view_changed)
        self.model.errorMessage.connect(self._on_error_message)

    # --------------------------------------------------------------------------
    # View -> Model の制御ハンドラ
    # --------------------------------------------------------------------------
    def _on_browse_clicked(self):
        """ファイルダイアログを開き、画像選択時に読み込みを実行"""
        file_filter = "全方位画像 (*.jpg *.jpeg *.png);;すべてのファイル (*.*)"
        file_path, _ = QFileDialog.getOpenFileName(
            self.window,
            "全方位画像ファイルを選択",
            "",
            file_filter
        )
        if file_path:
            self.window.set_path_text(file_path)
            self.model.load_image(file_path)

    def _on_load_requested(self, path: str):
        """パス入力欄からの読み込み要求"""
        if path:
            self.model.load_image(path)

    # --------------------------------------------------------------------------
    # Model -> View の反映ハンドラ
    # --------------------------------------------------------------------------
    def _on_image_loaded(self, path: str, w: int, h: int):
        """画像読み込み成功時のステータス表示"""
        filename = os.path.basename(path)
        self.window.set_status_message(
            f"読み込み完了: {filename} ({w}x{h}) | "
            f"Yaw: {self.model.yaw_deg:.1f}° | Pitch: {self.model.pitch_deg:.1f}° | "
            f"横FOV: {self.model.fov_w_deg:.1f}°, 縦FOV: {self.model.fov_h_deg:.1f}°"
        )

    def _on_view_changed(
        self,
        yaw: float,
        pitch: float,
        fov_w: float,
        fov_h: float,
        rendered_img
    ):
        """パラメータやレンダリング結果の変更時、Viewの各部を同期更新"""
        # 1. ビューワーキャンバスの再描画
        self.window.viewer.set_display_data(yaw, pitch, fov_w, fov_h, rendered_img)

        # 2. 操作タブ内のスピンボックス更新
        self.window.update_controls(fov_w, fov_h)

        # 3. ステータスバーの更新
        if self.model.file_path:
            filename = os.path.basename(self.model.file_path)
            dst_h, dst_w = rendered_img.shape[:2] if rendered_img is not None else (0, 0)
            self.window.set_status_message(
                f"ファイル: {filename} | "
                f"Yaw: {yaw:5.1f}° | Pitch: {pitch:5.1f}° | "
                f"横FOV: {fov_w:5.1f}°, 縦FOV: {fov_h:5.1f}° | "
                f"出力サイズ: {dst_w}x{dst_h}px"
            )

    def _on_error_message(self, msg: str):
        """エラーメッセージをステータスバーに表示"""
        self.window.set_status_message(msg)
