import open3d as o3d
import numpy as np
from pathlib import Path

import context
from vis_tac_sim.o3d_utils import select_points

view_params = {	
    "front" : [ 0.95291722914164678, 0.040399747718346389, 0.30052722803316911 ],
    "lookat" : [ -0.22423458151238204, 0.0088581721001014992, 0.0054539871171858682 ],
    "up" : [ -0.30187067376456861, 0.032633014217487287, 0.952790209177239 ],
    "zoom" : 0.2999999999999996
}

crop_view_params = {
    "front" : [ 0.97000986682365598, 0.038385063499544743, 0.24001551025900231 ],
    "lookat" : [ 0.21073045402493243, 0.013304188767706953, 0.17243558734608105 ],
    "up" : [ -0.24015699736271925, -0.00094990925186575967, 0.97073359594181985 ],
    "zoom" : 0.80000000000000004
}

if __name__ == '__main__':

    obj_id = 'eucalyptus_leaf_01'
    out_dir = f'out_data/plant_assets/{obj_id}'

    b_min = np.array([0.0, -0.15, 0.01])
    b_max = np.array([0.2,  0.15,  0.5])
    bbox = o3d.geometry.AxisAlignedBoundingBox(b_min, b_max)

    H_mat = np.load(f'{out_dir}/H_mat.npy')

    pcd_fn_lst = Path(f'/home/planck/plant-model/out_data/{obj_id}').glob('step*.pcd')
    pcd_fn_lst = sorted(pcd_fn_lst)

    for i, pcd_fn in enumerate(pcd_fn_lst):
        print('pcd fn:', pcd_fn)

        pcd = o3d.io.read_point_cloud(str(pcd_fn))
        pcd.transform(H_mat)
        
        # manually get bounding box
        # pts_idx = select_points(pcd)
        # print('select points:', np.array(pcd.points)[pts_idx, :])

        cropped_pcd = pcd.crop(bbox)
        in_bound_idx = bbox.get_point_indices_within_bounding_box(pcd.points)
        crop_pcd = pcd.select_by_index(in_bound_idx)

        print('number of in bound points:', len(crop_pcd.points))

        o3d.visualization.draw_geometries([crop_pcd, bbox], **crop_view_params)

        o3d.io.write_point_cloud(f'{out_dir}/step_{i:03d}.pcd', crop_pcd)