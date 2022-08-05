import os.path as osp
import warnings
import numpy as np
from functools import reduce

import pycocotools.mask as maskUtils

from pathlib import Path
from copy import deepcopy
from det3d import torchie
from det3d.core import box_np_ops
import pickle 
import os 
from ..registry import PIPELINES
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from nuscenes.map_expansion.map_api import NuScenesMap
import torch
from shapely.validation import make_valid
from shapely.geometry.polygon import Polygon 
import cv2
from nuscenes import NuScenes

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

    return dx, bx, nx

def get_rot(h):
    return np.array([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])

def get_nusc_maps(map_folder):
    nusc_maps = {map_name: NuScenesMap(dataroot=map_folder,
                map_name=map_name) for map_name in [
                    "singapore-hollandvillage", 
                    "singapore-queenstown",
                    "boston-seaport",
                    "singapore-onenorth",
                ]}
    return nusc_maps


def get_local_map(nmap, center, stretch, layer_names, line_names):
    # need to get the map here...
    box_coords = (
        center[0] - stretch,
        center[1] - stretch,
        center[0] + stretch,
        center[1] + stretch,
    )

    polys = {}

    # polygons
    records_in_patch = nmap.get_records_in_patch(box_coords,
                                                 layer_names=layer_names,
                                                 mode='intersect')
    for layer_name in layer_names:
        polys[layer_name] = []
        for token in records_in_patch[layer_name]:
            poly_record = nmap.get(layer_name, token)
            if layer_name == 'drivable_area':
                polygon_tokens = poly_record['polygon_tokens']
            else:
                polygon_tokens = [poly_record['polygon_token']]

            for polygon_token in polygon_tokens:
                polygon = nmap.extract_polygon(polygon_token)
                polys[layer_name].append(np.array(polygon.exterior.xy).T)

    # lines
    for layer_name in line_names:
        polys[layer_name] = []
        for record in getattr(nmap, layer_name):
            token = record['token']

            line = nmap.extract_line(record['line_token'])
            if line.is_empty:  # Skip lines without nodes
                continue
            xs, ys = line.xy

            polys[layer_name].append(
                np.array([xs, ys]).T
                )

    # convert to local coordinates in place
    rot = get_rot(np.arctan2(center[3], center[2])).T
    for layer_name in polys:
        for rowi in range(len(polys[layer_name])):
            polys[layer_name][rowi] -= center[:2]
            polys[layer_name][rowi] = np.dot(polys[layer_name][rowi], rot)

    return polys
    
def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]

def read_file(path, tries=2, num_point_feature=4, virtual=False):
    if virtual:
        # WARNING: hard coded for nuScenes 
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :num_point_feature]
        tokens = path.split('/')
        seg_path = os.path.join(*tokens[:-2], tokens[-2]+"_VIRTUAL", tokens[-1]+'.pkl.npy')
        data_dict = np.load(seg_path, allow_pickle=True).item()

        # remove reflectance as other virtual points don't have this value  
        virtual_points1 = data_dict['real_points'][:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]] 
        virtual_points2 = data_dict['virtual_points']

        points = np.concatenate([points, np.ones([points.shape[0], 15-num_point_feature])], axis=1)
        virtual_points1 = np.concatenate([virtual_points1, np.zeros([virtual_points1.shape[0], 1])], axis=1)
        virtual_points2 = np.concatenate([virtual_points2, -1 * np.ones([virtual_points2.shape[0], 1])], axis=1)
        points = np.concatenate([points, virtual_points1, virtual_points2], axis=0).astype(np.float32)
    else:
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :num_point_feature]

    return points


def remove_close(points, radius: float) -> None:
    """
    Removes point too close within a certain radius from origin.
    :param radius: Radius below which points are removed.
    """
    x_filt = np.abs(points[0, :]) < radius
    y_filt = np.abs(points[1, :]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    points = points[:, not_close]
    return points


def read_sweep(sweep, virtual=False):
    min_distance = 1.0
    points_sweep = read_file(str(sweep["lidar_path"]), virtual=virtual).T
    points_sweep = remove_close(points_sweep, min_distance)

    nbr_points = points_sweep.shape[1]
    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot(
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]
    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))

    return points_sweep.T, curr_times.T

def read_single_waymo(obj):
    points_xyz = obj["lidars"]["points_xyz"]
    points_feature = obj["lidars"]["points_feature"]

    # normalize intensity 
    points_feature[:, 0] = np.tanh(points_feature[:, 0])

    points = np.concatenate([points_xyz, points_feature], axis=-1)
    
    return points 

def read_single_waymo_sweep(sweep):
    obj = get_obj(sweep['path'])

    points_xyz = obj["lidars"]["points_xyz"]
    points_feature = obj["lidars"]["points_feature"]

    # normalize intensity 
    points_feature[:, 0] = np.tanh(points_feature[:, 0])
    points_sweep = np.concatenate([points_xyz, points_feature], axis=-1).T # 5 x N

    nbr_points = points_sweep.shape[1]

    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot( 
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]

    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))
    
    return points_sweep.T, curr_times.T


def get_obj(path):
    with open(path, 'rb') as f:
            obj = pickle.load(f)
    return obj 


@PIPELINES.register_module
class LoadPointCloudFromFile(object):
    def __init__(self, dataset="KittiDataset", **kwargs):
        self.type = dataset
        self.random_select = kwargs.get("random_select", False)
        self.npoints = kwargs.get("npoints", 16834)

    def __call__(self, res, info):

        res["type"] = self.type

        if self.type == "NuScenesDataset":

            nsweeps = res["lidar"]["nsweeps"]

            lidar_path = Path(info["lidar_path"])
            points = read_file(str(lidar_path), virtual=res["virtual"])

            sweep_points_list = [points]
            sweep_times_list = [np.zeros((points.shape[0], 1))]

            assert (nsweeps - 1) == len(
                info["sweeps"]
            ), "nsweeps {} should equal to list length {}.".format(
                nsweeps, len(info["sweeps"])
            )

            for i in np.random.choice(len(info["sweeps"]), nsweeps - 1, replace=False):
                sweep = info["sweeps"][i]
                points_sweep, times_sweep = read_sweep(sweep, virtual=res["virtual"])
                sweep_points_list.append(points_sweep)
                sweep_times_list.append(times_sweep)

            points = np.concatenate(sweep_points_list, axis=0)
            times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

            res["lidar"]["points"] = points
            res["lidar"]["times"] = times
            res["lidar"]["combined"] = np.hstack([points, times])
        
        elif self.type == "WaymoDataset":
            path = info['path']
            nsweeps = res["lidar"]["nsweeps"]
            obj = get_obj(path)
            points = read_single_waymo(obj)
            res["lidar"]["points"] = points

            if nsweeps > 1: 
                sweep_points_list = [points]
                sweep_times_list = [np.zeros((points.shape[0], 1))]

                assert (nsweeps - 1) == len(
                    info["sweeps"]
                ), "nsweeps {} should be equal to the list length {}.".format(
                    nsweeps, len(info["sweeps"])
                )

                for i in range(nsweeps - 1):
                    sweep = info["sweeps"][i]
                    points_sweep, times_sweep = read_single_waymo_sweep(sweep)
                    sweep_points_list.append(points_sweep)
                    sweep_times_list.append(times_sweep)

                points = np.concatenate(sweep_points_list, axis=0)
                times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

                res["lidar"]["points"] = points
                res["lidar"]["times"] = times
                res["lidar"]["combined"] = np.hstack([points, times])
        else:
            raise NotImplementedError

        return res, info


@PIPELINES.register_module
class LoadPointCloudAnnotations(object):
    def __init__(self, with_bbox=True, **kwargs):
        self._root_path = "data/nuScenes"
        self.nusc =  NuScenes(version='v1.0-trainval', dataroot=self._root_path, verbose=True)
        ## this will generate a gt with the same size of the mid spatial features
        self.x_bound,self.y_bound,self.z_bound = [-51.2, 51.2, 0.2 ], [-51.2, 51.2, 0.2], [-10.0, 10.0, 20.0] # 0.25->0.403225 for voxel net
        dx, bx, nx = gen_dx_bx(self.x_bound,self.y_bound,self.z_bound)
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()
        # pass

    def get_binimg_map(self, rec):
        nusc = self.nusc
        egopose = nusc.get('ego_pose', nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        scene2map = {}
        for rec_ in self.nusc.scene:
            log = self.nusc.get('log', rec_['log_token'])
            scene2map[rec_['name']] = log['location']
        map_name = scene2map[nusc.get('scene', rec['scene_token'])['name']]

        rot = Quaternion(egopose['rotation']).rotation_matrix
        rot = np.arctan2(rot[1, 0], rot[0, 0])
        center = np.array([egopose['translation'][0], egopose['translation'][1], np.cos(rot), np.sin(rot)])

        poly_names = ['road_segment', 'lane','drivable_area','ped_crossing','walkway','stop_line','carpark_area']
        line_names = ['road_divider', 'lane_divider']
        map_folder = self._root_path
        nusc_maps = get_nusc_maps(map_folder)
        lmap = get_local_map(nusc_maps[map_name], center,
                            50.0, poly_names, line_names)
        # print(lmap)
        my_im_size = 512
        total_poly = Polygon([(0,0),(my_im_size,0),(my_im_size,my_im_size),(0,my_im_size)]) ## 400 -> 248
        
        imgs = np.zeros((len(poly_names),my_im_size,my_im_size))
        dx, bx = self.dx[:2], self.bx[:2]
        for i,name in enumerate(poly_names):
            for la in lmap[name]:
                pts = (la - bx) / dx
                pts_reversed = [(pts[indd_,1],pts[indd_,0]) for indd_ in range(len(pts))]
                pts_poly = Polygon(pts_reversed)
                if total_poly.disjoint(pts_poly):
                    continue
                # print()
                pts_poly = make_valid(pts_poly)
                # print(pts_poly)
                poly_res = total_poly.intersection(pts_poly)
                
                # try:
                #     poly_res = total_poly.intersection(pts_poly)
                # except:
                #     print("!!!!!")
                #     print(poly_res)
                #     pts_poly.buffer(0)
                #     print(poly_res)
                #     poly_res = total_poly.intersection(pts_poly)
                if type(poly_res) == type(total_poly):
                    x_res, y_res = poly_res.exterior.coords.xy
                    res_pts = [(x_res[ind_],y_res[ind_]) for ind_ in range(len(x_res))]
                    # print(res_pts)
                    cv2.fillPoly(imgs[i],np.int32([res_pts]),1)
                else:
                    for polys in poly_res:
                        x_res, y_res = polys.exterior.coords.xy
                        res_pts = [(x_res[ind_],y_res[ind_]) for ind_ in range(len(x_res))]
                        # print(res_pts)
                        cv2.fillPoly(imgs[i],np.int32([res_pts]),1)
                
        return torch.Tensor(imgs)

    def get_binimg_ve(self, rec):
        egopose = self.nusc.get('ego_pose',
                                self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        img = np.zeros((self.nx[0], self.nx[1]))
        for tok in rec['anns']:
            inst = self.nusc.get('sample_annotation', tok)
            # add category for lyft
            if not inst['category_name'].split('.')[0] == 'vehicle':
                # print(inst['category_name']): human, static_object,movable_object
                # print(inst['category_name'].split('.')[0])
                continue
            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
            box.translate(trans)
            box.rotate(rot)

            pts = box.bottom_corners()[:2].T
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(img, [pts], 1.0)
        return torch.Tensor(img)

    def get_binimg_human(self, rec):
        egopose = self.nusc.get('ego_pose',
                                self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        img = np.zeros((self.nx[0], self.nx[1]))
        for tok in rec['anns']:
            inst = self.nusc.get('sample_annotation', tok)
            # add category for lyft
            if not inst['category_name'].split('.')[0] == 'human':
                # print(inst['category_name']): human, static_object,movable_object
                continue
            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
            box.translate(trans)
            box.rotate(rot)

            pts = box.bottom_corners()[:2].T
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(img, [pts], 1.0)

        return torch.Tensor(img)


    def __call__(self, res, info):

        if res["type"] in ["NuScenesDataset"] and "gt_boxes" in info:
            gt_boxes = info["gt_boxes"].astype(np.float32)
            gt_boxes[np.isnan(gt_boxes)] = 0
            info_tocken = info["token"]
            rec = self.nusc.get('sample', info_tocken)
            binimg_map = self.get_binimg_map(rec)
            res["lidar"]["annotations"] = {
                "boxes": gt_boxes,
                "names": info["gt_names"],
                "tokens": info["gt_boxes_token"],
                "velocities": info["gt_boxes_velocity"].astype(np.float32),
                "bin_map":binimg_map
            }

            

        elif res["type"] == 'WaymoDataset' and "gt_boxes" in info:
            res["lidar"]["annotations"] = {
                "boxes": info["gt_boxes"].astype(np.float32),
                "names": info["gt_names"],
            }
        else:
            pass 

        return res, info
