# omni_directional_img_utils
## e2p.py

## ppi.py

## generate_ppi_gui.py
GUIで全方位画像の座標を指定して、透視投影画像を作成する

### generate_ppi_gui(src_img, scale=1, fov_w_deg=30, fov_v_deg=30)
**引数：**
- **src_img:** cv2.imreadで読み込んだ画像のnp配列
- **scale:** GUIで画像を表示する際のスケール
- **fov_w_deg:** 作成する透視投影画像の画角（横）
- **fov_h_deg:** 作成する透視投影画像の画角（縦）

**戻り値：** 
- ** ppi:** 透視投影画像のnp配列