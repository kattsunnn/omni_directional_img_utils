from pathlib import Path

import click
import numpy as np

import img_utils as iu
from omni_directional_img_utils.e2p import E2P

def generate_ppi_with_gui(src_img, scale=1, fov_w_deg=30, fov_v_deg=30):
    point, _ = iu.get_single_point_with_gui(src_img, scale)
    u = point[0]
    v = point[1]
    ppi_generator = E2P(src_img.shape[1], src_img.shape[0])
    angle_u_deg, angle_v_deg = ppi_generator.uv_to_angle(u, v)
    ppi_generator.generate_map(fov_w_deg, fov_v_deg, angle_u_deg, angle_v_deg, 0)
    ppi = ppi_generator.generate_img(src_img)
    r_mat = E2P.angle_to_r_mat(angle_u_deg, angle_v_deg)
    return ppi, r_mat

@click.command
@iu.prepare_io_path
@click.option('-ws', '--window_scale', required=True, type=float, help='スケール（必須）')
@click.option('-fwd', '--fov-w_deg', type=float, required=True, help='画角（横）')
@click.option('-fhd', '--fov-h_deg', type=float, required=True, help='画角（縦）')
def main(input_path, output_path, window_scale, fov_w_deg, fov_h_deg):
    src_img = iu.load_imgs(input_path)
    ppi, r_mat = generate_ppi_with_gui(src_img, window_scale, fov_w_deg, fov_h_deg)
    # print(E2P.r_mat_to_angle(r_mat))
    iu.show_imgs(ppi)
    file_name = Path(input_path).stem + "_ppi"
    iu.save_imgs(ppi, output_path, file_name=file_name)
    file_name = Path(input_path).stem + "_ppi_r"
    np.savetxt(Path(output_path)/Path(file_name).with_suffix(".dat"), r_mat)

main()