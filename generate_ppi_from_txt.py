from generate_ppi import generate_ppi_from_uv
import img_utils.img_utils as iu
import sys
 
img_path = sys.argv[1]
fov_w_deg = float(sys.argv[2])
fov_h_deg = float(sys.argv[3])
coordinates_txt_path = sys.argv[4]

# 画像読み込み
src_img = iu.load_imgs(img_path)
# 座標読み込み
coordinates = iu.load_coodinates_from_txt(coordinates_txt_path)

for u, v in coordinates:
    ppi = generate_ppi_from_uv(src_img, fov_w_deg, fov_h_deg, u, v)
    iu.show_imgs(ppi)

# 実行例：python .\generate_ppi_from_txt.py .\sample\sample.jpg 30 30 .\sample\sample.txt