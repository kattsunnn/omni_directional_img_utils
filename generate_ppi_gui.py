import img_utils.img_utils as iu
from e2p import E2P

def generate_ppi_gui(src_img, scale=1, fov_w_deg=30, fov_v_deg=30):
    point, _ = iu.get_single_point_with_gui(src_img, scale)
    u = point[0]
    v = point[1]
    ppi_generator = E2P(src_img.shape[1], src_img.shape[0])
    angle_u_deg, angle_v_deg = ppi_generator.uv_to_angle(u, v)
    ppi_generator.generate_map(fov_w_deg, fov_v_deg, angle_u_deg, angle_v_deg, 0)
    ppi = ppi_generator.generate_img(src_img)
    return ppi

if __name__ == "__main__":

    import sys

    input_path = sys.argv[1]
    scale = float(sys.argv[2])
    fov_w_deg = float(sys.argv[3])
    fov_h_deg = float(sys.argv[4])

    src_img = iu.load_imgs(input_path)

    ppi = generate_ppi_gui(src_img, scale, fov_w_deg, fov_h_deg)
    iu.show_imgs(ppi)

    # 実行例：
    # python .\generate_ppi_gui.py .\sample_img\sample.jpg 0.5 60 30  