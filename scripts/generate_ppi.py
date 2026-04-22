# def generate_ppi_from_uv(src_img, fov_w_deg, fov_h_deg, u, v):
#     e2p = E2P(src_img.shape[1], src_img.shape[0])
#     angle_u_deg, angle_v_deg = e2p.uv_to_angle(u, v)
#     e2p.generate_map( fov_w_deg, fov_h_deg, angle_u_deg, angle_v_deg, 0, scale=1.0)
#     ppi = e2p.generate_img(src_img)
#     r_mat = E2P.calc_r_xy(angle_u_deg, angle_v_deg)
#     return ppi, r_mat

import sys
import cv2

from omni_directional_img_utils.e2p import E2P
import img_utils as iu

src_img = cv2.imread(sys.argv[1]) # 全方位画像
fov_w_deg = float(sys.argv[2])
fov_h_deg = float(sys.argv[3])
angle_u_deg = float(sys.argv[4])
angle_v_deg = float(sys.argv[5])
angle_z_deg = float(sys.argv[6])

e2p = E2P(src_img.shape[1], src_img.shape[0])
e2p.generate_map( fov_w_deg, fov_h_deg, 
                    angle_u_deg, angle_v_deg, angle_z_deg,
                    scale=1.0)
dst_img = e2p.generate_img(src_img)
iu.show_imgs(dst_img)

if len(sys.argv) > 7:
    output_path = sys.argv[7]
    cv2.imwrite(output_path, dst_img)
    print(f"Image saved to: {output_path}")

# uv run .\generate_ppi.py .\sample\sample.jpg 60 60 30 0 0
# foreach ($img in Get-Item "C:\Users\naoki\Prog\output\frame_division\surrounding_frames\tennis_center\camera_00\*.jpg") { python scripts/generate_ppi.py $img.FullName 30 30 -27 1 0 }