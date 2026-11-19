"""
全方位画像ビューワー - Controller (controller.py)

ModelとViewの仲介を行い、ユーザー操作イベントに応じたModelの更新、
およびModelの状態変化に応じたViewの描画更新、データ保存処理を制御する。
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
        self.window.lockToggled.connect(self.model.toggle_lock)
        self.window.modeToggled.connect(self.model.toggle_mode)
        self.window.undoRequested.connect(self.model.undo_annotation)
        self.window.savePointsClicked.connect(self._on_save_points_clicked)
        self.window.saveLinesClicked.connect(self._on_save_lines_clicked)
        self.window.saveViewParamsClicked.connect(self._on_save_view_params_clicked)
        self.window.fileDropped.connect(self._on_file_dropped)

        # ViewerCanvas（マウス・キー操作）からのシグナル
        self.window.viewer.dragged.connect(self.model.rotate_view)
        self.window.viewer.wheelBoth.connect(self.model.change_fov_both_by_wheel)
        self.window.viewer.wheelVertical.connect(self.model.change_fov_height_by_wheel)
        self.window.viewer.wheelHorizontal.connect(self.model.change_fov_width_by_wheel)
        self.window.viewer.lockToggled.connect(self.model.toggle_lock)
        self.window.viewer.modeToggled.connect(self.model.toggle_mode)
        self.window.viewer.pointClicked.connect(self.model.handle_click_annotation)
        self.window.viewer.undoRequested.connect(self.model.undo_annotation)

    def _connect_model_signals(self):
        """Modelからのデータ更新シグナルをViewへ反映"""
        self.model.imageLoaded.connect(self._on_image_loaded)
        self.model.viewChanged.connect(self._on_view_changed)
        self.model.lockStateChanged.connect(self._on_lock_state_changed)
        self.model.modeChanged.connect(self._on_mode_changed)
        self.model.pointsChanged.connect(self.window.viewer.set_points)
        self.model.pointsChanged.connect(self._on_points_changed)
        self.model.linesChanged.connect(self.window.viewer.set_lines)
        self.model.linesChanged.connect(self._on_lines_changed)
        self.model.errorMessage.connect(self._on_error_message)
        self.model.infoMessage.connect(self._on_info_message)

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

    def _on_file_dropped(self, path: str):
        """ドラッグ＆ドロップによる画像読み込み要求"""
        self.window.set_path_text(path)
        self.model.load_image(path)

    def _on_save_points_clicked(self):
        """点データの.dat保存ダイアログ"""
        if not self.model.points:
            self.window.set_status_message("⚠️ 保存する点がありません")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self.window,
            "点データを保存 (.dat)",
            "points.dat",
            "DATファイル (*.dat);;すべてのファイル (*.*)"
        )
        if file_path:
            self.model.save_points_dat(file_path)

    def _on_save_lines_clicked(self):
        """線データの.dat保存ダイアログ"""
        if not self.model.lines:
            self.window.set_status_message("⚠️ 保存する線がありません")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self.window,
            "線データを保存 (.dat)",
            "lines.dat",
            "DATファイル (*.dat);;すべてのファイル (*.*)"
        )
        if file_path:
            self.model.save_lines_dat(file_path)

    def _on_save_view_params_clicked(self):
        """ロック中の視点情報を.dat保存する"""
        if not self.model.is_locked:
            self.window.set_status_message("⚠️ 視点情報の保存はロック中のみ可能です")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self.window,
            "視点情報を保存 (.dat)",
            "view_params.dat",
            "DATファイル (*.dat);;すべてのファイル (*.*)"
        )
        if file_path:
            self.model.save_view_params_dat(file_path)

    # --------------------------------------------------------------------------
    # Model -> View の反映ハンドラ
    # --------------------------------------------------------------------------
    def _on_image_loaded(self, path: str, w: int, h: int):
        """画像読み込み成功時のステータス表示"""
        filename = os.path.basename(path)
        fl = self.model.focal_length
        self.window.set_status_message(
            f"読み込み完了: {filename} ({w}x{h}) | 焦点距離 f: {fl:.2f}px | "
            f"Yaw: {self.model.yaw_deg:.1f}° | Pitch: {self.model.pitch_deg:.1f}° | "
            f"横FOV: {self.model.fov_w_deg:.1f}°, 縦FOV: {self.model.fov_h_deg:.1f}°"
        )

    def _on_lock_state_changed(self, is_locked: bool):
        """ロック状態変更時のUIおよびキャンバスの同期"""
        self.window.set_lock_state(is_locked)
        self.window.viewer.set_locked(is_locked)

        if is_locked:
            self.window.set_status_message("🔒 ロック中: クリックで配置、Mキーで点/線切替、Dキーで削除 (Lキーで解除・全削除)")
        else:
            self.window.set_status_message("🔓 ロックを解除しました（配置されたすべての点・線を全削除しました）")

    def _on_mode_changed(self, mode: str):
        """モード変更時のUI更新"""
        self.window.set_mode_display(mode)
        mode_ja = "点選択モード" if mode == "point" else "線選択モード (2点クリックで線指定)"
        self.window.set_status_message(f"📍 {mode_ja} に切り替えました [Mキーで切替]")

    def _on_points_changed(self, points: list):
        """点配置・削除時のステータスバー更新"""
        if not self.model.is_locked or self.model.select_mode != "point":
            return

        cnt = len(points)
        if cnt > 0:
            last_pt = points[-1]
            self.window.set_status_message(
                f"点P{last_pt['index']} (U: {last_pt['omni_u']:.1f}, V: {last_pt['omni_v']:.1f}) | 合計 {cnt} 点配置中 [Dキーで削除]"
            )
        else:
            self.window.set_status_message("🔒 ロック中 (点モード): 画像をクリックして点を配置できます")

    def _on_lines_changed(self, lines: list, pending_start: dict | None):
        """線配置・削除時のステータスバー更新"""
        if not self.model.is_locked or self.model.select_mode != "line":
            return

        if pending_start is not None:
            self.window.set_status_message(
                f"始点を指定しました (U: {pending_start['omni_u']:.1f}, V: {pending_start['omni_v']:.1f}) → 端点をクリックしてください [Dキーで取消]"
            )
        else:
            cnt = len(lines)
            if cnt > 0:
                self.window.set_status_message(f"線L{cnt} を確定しました | 合計 {cnt} 本の線を配置中 [Dキーで削除]")
            else:
                self.window.set_status_message("🔒 ロック中 (線モード): 1回目のクリックで始点、2回目のクリックで端点を指定")

    def _on_view_changed(
        self,
        yaw: float,
        pitch: float,
        fov_w: float,
        fov_h: float,
        rendered_img
    ):
        """パラメータやレンダリング結果の変更時、Viewの各部を同期更新"""
        self.window.viewer.set_display_data(yaw, pitch, fov_w, fov_h, rendered_img)
        self.window.update_controls(fov_w, fov_h)

        if self.model.file_path:
            filename = os.path.basename(self.model.file_path)
            dst_h, dst_w = rendered_img.shape[:2] if rendered_img is not None else (0, 0)
            lock_label = " [🔒 LOCKED]" if self.model.is_locked else ""
            self.window.set_status_message(
                f"ファイル: {filename} | "
                f"Yaw: {yaw:5.1f}° | Pitch: {pitch:5.1f}° | "
                f"横FOV: {fov_w:5.1f}°, 縦FOV: {fov_h:5.1f}° | "
                f"出力サイズ: {dst_w}x{dst_h}px{lock_label}"
            )

    def _on_error_message(self, msg: str):
        """エラーメッセージをステータスバーに表示"""
        self.window.set_status_message(f"⚠️ {msg}")

    def _on_info_message(self, msg: str):
        """情報メッセージをステータスバーに表示"""
        self.window.set_status_message(f"ℹ️ {msg}")
