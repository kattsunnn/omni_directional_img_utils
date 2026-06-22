# def generate_ppi_from_uv(src_img, fov_w_deg, fov_h_deg, u, v):
#     e2p = E2P(src_img.shape[1], src_img.shape[0])
#     angle_u_deg, angle_v_deg = e2p.uv_to_angle(u, v)
#     e2p.generate_map( fov_w_deg, fov_h_deg, angle_u_deg, angle_v_deg, 0, scale=1.0)
#     ppi = e2p.generate_img(src_img)
#     r_mat = E2P.calc_r_xy(angle_u_deg, angle_v_deg)
#     return ppi, r_mat

import sys
import cv2
import argparse # オプション解析用に追加

from omni_directional_img_utils.e2p import E2P
import img_utils as iu

def main():
    parser = argparse.ArgumentParser(description="Omni-directional image to Perspective image")

    parser.add_argument("-i", "--input_path", help="Path to source omni-directional image")
    parser.add_argument("-o", "--output_path", help="Path to save the output image")
    parser.add_argument("-fw", "--fov_w", type=float, required=True, help="FOV width in degrees")
    parser.add_argument("-fh", "--fov_h", type=float, required=True, help="FOV height in degrees")
    parser.add_argument("-u", "--angle_u", type=float, required=True, help="Angle U")
    parser.add_argument("-v", "--angle_v", type=float, required=True, help="Angle V")
    parser.add_argument("-z", "--angle_z", type=float, required=True, help="Angle Z")
    
    # モード選択
    parser.add_argument("-m", "--mode", choices=["view", "save", "both"], default="both",
                        help="Execution mode: view, save, both (default: both)")

    args = parser.parse_args()

    src_img = cv2.imread(args.input_path)
    if src_img is None:
        print(f"Error: Could not load image {args.src_img}")
        return

    e2p = E2P(src_img.shape[1], src_img.shape[0])
    e2p.generate_map(args.fov_w, args.fov_h, 
                    args.angle_u, args.angle_v, args.angle_z,
                    scale=1.0)
    dst_img = e2p.generate_img(src_img)

    if args.mode in ["view", "both"]:
        print("Showing image... (Press any key to close)")
        # img_utilsにshow_imgsがある場合はそちらを使用
        cv2.imshow("Output Image", dst_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if args.mode in ["save", "both"]:
        cv2.imwrite(args.output_path, dst_img)
        print(f"Image saved to: {args.output_path}")

if __name__ == "__main__":
    main()

# uv run .\generate_ppi.py .\sample\sample.jpg 60 60 30 0 0
# foreach ($img in Get-Item "C:\Users\naoki\Prog\output\frame_division\surrounding_frames\2026-04-23\*.jpg") {
#     # Using $img.Name to get "filename.jpg"
#     python scripts/generate_ppi.py $img.FullName 30 30 -27 1 0 "C:\Users\naoki\Prog\output\omni_directional_img_utils\2026-04-23\$($img.Name)"
# }