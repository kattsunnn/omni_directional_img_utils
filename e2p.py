# 全方位画像のサイズを元に、任意の透視投影画像のマップを生成するクラス
# Todo: 画像サイズから適切な画角を計算する関数も追加
# Todo: 引数にValidateを追加

import math
import numpy as np
import cv2

class E2P:
    def __init__(self, src_img_w, src_img_h):

        # 全方位画像の大きさ
        self.src_img_w = src_img_w
        self.src_img_h = src_img_h
        # 焦点距離
        self.f = self.src_img_w / (2*np.pi)

        self.map_u = None
        self.map_v = None

    #画角から適切な出力画像の横サイズを計算
    def calc_optimal_width(self, fov_deg):
        fov_rad = np.deg2rad(fov_deg)
        optimal_w = int(2.0 * np.tan(fov_rad / 2.0) * self.src_img_w / (2*np.pi))
        return optimal_w
    # 画角から適切な出力画像の縦サイズを計算
    def calc_optimal_height(self, fov_deg):
        fov_rad = np.deg2rad(fov_deg)
        optimal_h = int(2.0 * np.tan(fov_rad / 2.0) * self.src_img_h / np.pi)
        return optimal_h

    # X軸周りの回転行列
    @staticmethod
    def rotation_x(angle):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        R = np.array([[1.0, 0.0, 0.0],
                    [0.0, cos_a, -sin_a],
                      [0.0, sin_a, cos_a]], dtype=np.float64)
        return R

    # Y軸周りの回転行列
    @staticmethod
    def rotation_y(angle):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        R = np.array([[cos_a, 0.0, sin_a],
                      [0.0, 1.0, 0.0],
                      [-sin_a, 0.0, cos_a]], dtype=np.float64)
        return R

    # 任意の軸周りの回転行列
    @staticmethod
    def rotation_by_axis(n, angle):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        l_cos_a = 1.0 - cos_a
        R = np.array([[cos_a + n[0] * n[0] * l_cos_a,
                       n[0] * n[1] * l_cos_a - n[2] * sin_a,
                       n[2] * n[0] * l_cos_a + n[1] * sin_a],
                      [n[0] * n[1] * l_cos_a + n[2] * sin_a,
                       cos_a + n[1] * n[1] * l_cos_a,
                       n[1] * n[2] * l_cos_a - n[0] * sin_a],
                      [n[2] * n[0] * l_cos_a - n[1] * sin_a,
                       n[1] * n[2] * l_cos_a + n[0] * sin_a,
                       cos_a + n[2] * n[2] * l_cos_a]], dtype=np.float64)
        return R
    
    # 画像生成用のマップ作成関数
    def generate_map(self,
                     dst_w,
                     dst_h,
                     angle_u_deg,
                     angle_v_deg,
                     angle_z_deg,
                     scale=1):
        angle_u_rad = np.deg2rad(angle_u_deg)
        angle_v_rad = np.deg2rad(angle_v_deg)
        angle_z_rad = np.deg2rad(angle_z_deg)

        # 回転行列の計算
        R = np.dot(self.rotation_y(angle_u_rad), self.rotation_x(angle_v_rad))
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        Ri = self.rotation_by_axis(np.dot(R, axis), angle_z_rad)
        R = np.dot(Ri, R)
        
        # 一括計算用の出力画像の画素座標データ
        u = np.arange(0, dst_w, 1)
        v = np.arange(0, dst_h, 1)
        dst_u, dst_v = np.meshgrid(u, v)

        # 視線ベクトルを計算
        x = dst_u - dst_w * 0.5
        y = dst_v - dst_h * 0.5
        z = self.f * scale * np.ones((dst_h, dst_w))
        
        # 回転行列で回転
        Xx = R[0][0] * x + R[0][1] * y + R[0][2] * z
        Xy = R[1][0] * x + R[1][1] * y + R[1][2] * z
        Xz = R[2][0] * x + R[2][1] * y + R[2][2] * z
        
        # 視線ベクトルから角度を計算
        theta = np.arctan2(Xx, Xz)
        phi = np.arctan2(np.sqrt(Xx**2 + Xz**2), Xy)

        # 角度から入力画像の座標を計算
        self.map_u = (0.5 * (theta + np.pi) * self.src_img_w / np.pi - 0.5).astype(np.float32)
        self.map_v = ((np.pi - phi) * self.src_img_h / np.pi - 0.5).astype(np.float32)

    # 画像生成
    def generate_img(self, src_img):
        if self.map_u is None or self.map_v is None:
            raise Exception("Error: Map is not generated yet. Please call generate_map() first.")
        return cv2.remap(src_img, self.map_u, self.map_v, cv2.INTER_LINEAR)

if __name__ == '__main__':

    import sys

    src_img = cv2.imread(sys.argv[1]) # 全方位画像
    fov_w_deg = float(sys.argv[2])
    fov_h_deg = float(sys.argv[3])
    angle_u_deg = float(sys.argv[4])
    angle_v_deg = float(sys.argv[5])
    angle_z_deg = float(sys.argv[6])
    e2p = E2P(src_img.shape[1], src_img.shape[0])

    optimal_width = e2p.calc_optimal_width(fov_w_deg)
    optimal_height = e2p.calc_optimal_height(fov_h_deg)

    e2p.generate_map( optimal_width, optimal_height, 
                      angle_u_deg, angle_v_deg, angle_z_deg,
                      scale=1.0)

    dst_img = e2p.generate_img(src_img)

    # 画像表示
    cv2.imshow("dst", dst_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()