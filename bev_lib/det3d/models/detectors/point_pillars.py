from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from copy import deepcopy 
from ..seg.seg_heads import *   

@DETECTORS.register_module
class PointPillars(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(PointPillars, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )
        self.seg_head = Unet_res50()
    def extract_feat(self, data):
        input_features = self.reader(
            data["features"], data["num_voxels"], data["coors"]
        )
        x = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        # print(x.size())
        if self.with_neck:
            neck_x = self.neck(x)
        return x,neck_x

    def forward(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        before_neck_x,x = self.extract_feat(data)
        # print(x.size())
        # print(before_neck_x.size())
        preds, _ = self.bbox_head(x)
        seg_preds = self.seg_head(before_neck_x)
        seg_loss_fn = nn.BCELoss()
        
        # for debug
        # np.save('gt.npy',gt.cpu().detach().numpy())
        # np.save('before_neck_x.npy',before_neck_x.cpu().detach().numpy())
        # np.save('seg_preds.npy',seg_preds.cpu().detach().numpy())


        # print(gt.size())
        # print(seg_preds.size())
        # print('inside forward!')
        if return_loss:
            gt = example["bin_map"]
            seg_loss = 0
            for i in range(len(gt[0])):
                seg_loss += seg_loss_fn(seg_preds[:,i,:,:],gt[:,i,:,:])
            print('seg loss',seg_loss)
            return self.bbox_head.loss(example, preds, self.test_cfg), 3*seg_loss
        else:
            # print('!!!')
            return self.bbox_head.predict(example, preds, self.test_cfg),seg_preds

    def forward_two_stage(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x = self.extract_feat(data)
        bev_feature = x 
        preds, _ = self.bbox_head(x)

        # manual deepcopy ...
        new_preds = []
        for pred in preds:
            new_pred = {} 
            for k, v in pred.items():
                new_pred[k] = v.detach()

            new_preds.append(new_pred)

        boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

        if return_loss:
            return boxes, bev_feature, self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            return boxes, bev_feature, None 
