import img_utils.img_utils as iu
from e2p import E2P
import sys

# コマンドライン引数
img_path = sys.argv[1]
u = float(sys.argv[2])
v = float(sys.argv[3])
fov_w_deg = float(sys.argv[4])
fov_h_deg = float(sys.argv[5])

# 画像読み込み
src_img = iu.load_imgs(img_path)

# インスタンス化
e2p = E2P(src_img.shape[1], src_img.shape[0])

# 画像座標を角度座標に変換（generate_mapの引数用）
angle_u_deg, angle_v_deg = e2p.uv_to_angle(u, v)

# 透視投影画像のマップを作成
e2p.generate_map( fov_w_deg, fov_h_deg, angle_u_deg, angle_v_deg, 0, scale=1.0)

# 透視投影画像の作成
dst_img = e2p.generate_img(src_img)

# 画像表示
iu.show_imgs(dst_img)

