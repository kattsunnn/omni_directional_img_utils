from generate_ppi import generate_ppi_with_gui
import img_utils.img_utils as iu
import sys

input_path = sys.argv[1]
scale = float(sys.argv[2])
fov_w_deg = float(sys.argv[3])
fov_h_deg = float(sys.argv[4])

src_img = iu.load_imgs(input_path)

ppi = generate_ppi_with_gui(src_img, scale, fov_w_deg, fov_h_deg)
iu.show_imgs(ppi)

# 実行例：python .\generate_ppi_with_gui.py ..\sample\sample.jpg 0.5 60 30  