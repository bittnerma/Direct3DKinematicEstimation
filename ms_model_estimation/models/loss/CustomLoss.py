import torch


class CustomLoss:

    @staticmethod
    def opensim_rotation_mat_loss(lossType=1, beta=0.04):

        if lossType == 3:
            loss = torch.nn.SmoothL1Loss(reduction='none', beta=beta)

        def inner(pred, target, evaluation=False):

            pred = pred[:, :3, :3]
            target = target[:, :3, :3]
            projection_pred = torch.sum(pred, dim=2) / (3 ** 0.5)
            projection_target = torch.sum(target, dim=2) / (3 ** 0.5)

            if lossType == 3:
                count = loss(projection_pred, projection_target)
                count = torch.sum(count, dim=-1)
            elif lossType == 1 or lossType == 2:
                count = torch.linalg.norm(projection_pred - projection_target, ord=lossType, dim=- 1)
            else:
                assert False

            if evaluation:
                return torch.sum(count)
            else:
                # averaged by batch size
                return torch.mean(count)

        return inner

    @staticmethod
    def opensim_coordinate_projection_loss():
        loss = torch.nn.L1Loss(reduction="none")

        def inner(pred, target, coordinateMask, eval=True):
            count = loss(pred, target)
            count = torch.sum(count, dim=-1)
            count = count * coordinateMask

            return torch.sum(count) / torch.sum(coordinateMask)

        return inner

    @staticmethod
    def opensim_coordinate_angle_loss(coef=None, coordinateWeights=None, lossType=1, beta=1.57):

        if lossType == 3:
            loss = torch.nn.SmoothL1Loss(reduction='none', beta=beta)

        def inner(pred, target, mask=None, evaluation=False):

            if lossType == 1:
                count = torch.abs(pred - target)
            elif lossType == 2:
                count = (pred - target) ** 2
            elif lossType == 3:
                # diff = pred - target
                # diff = torch.einsum('bj,j->bj', diff, coef)
                pred = torch.einsum('bj,j->bj', pred, coef)
                target = torch.einsum('bj,j->bj', target, coef)
                count = loss(pred, target)
                count = torch.einsum('bj,j->bj', count, torch.div(1.0, coef))
            else:
                assert False

            if coordinateWeights is not None:
                count = torch.einsum('bj,j->bj', count, coordinateWeights)

            if mask is not None:
                assert mask.shape == count.shape
                count = count * mask
                maskSum = torch.sum(mask, dim=-1)
                maskSum = (maskSum == 0) * 1.0 + (maskSum != 0) * maskSum
                count = torch.div(torch.sum(count, dim=-1), maskSum)
            else:
                # averaged by the number of key joints
                count = torch.mean(count, dim=-1)

            if evaluation:
                return torch.sum(count)
            else:
                # averaged by batch size
                return torch.mean(count)

        return inner

    @staticmethod
    def opensim_sequence_coordinate_angle_loss(coef=None, coordinateWeights=None, lossType=1, beta=1.57):

        if lossType == 3:
            loss = torch.nn.SmoothL1Loss(reduction='none', beta=beta)

        def inner(pred, target, mask=None, evaluation=False):

            assert len(pred.shape) == 3
            assert len(target.shape) == 3

            if lossType == 1:
                count = torch.abs(pred - target)
            elif lossType == 2:
                count = (pred - target) ** 2
            elif lossType == 3:
                # diff = pred - target
                # diff = torch.einsum('bj,j->bj', diff, coef)
                pred = torch.einsum('bsj,j->bsj', pred, coef)
                target = torch.einsum('bsj,j->bsj', target, coef)
                count = loss(pred, target)
                count = torch.einsum('bsj,j->bsj', count, torch.div(1.0, coef))
            else:
                assert False

            if coordinateWeights is not None:
                count = torch.einsum('bsj,j->bsj', count, coordinateWeights)

            if mask is not None:
                assert mask.shape == count.shape
                count = count * mask
                maskSum = torch.sum(mask, dim=-1)
                maskSum = (maskSum == 0) * 1.0 + (maskSum != 0) * maskSum
                count = torch.div(torch.sum(count, dim=-1), maskSum)
            else:
                # averaged by the number of angles
                count = torch.mean(count, dim=-1)

            # averaged by the sequence length
            count = torch.mean(count, dim=-1)

            if evaluation:
                return torch.sum(count)
            else:
                # averaged by batch size
                return torch.mean(count)

        return inner

    @staticmethod
    def bone_scale_loss(L, weights=None, coef=None, beta=0.5):

        if L == 2:
            loss = torch.nn.MSELoss(reduction='none')
        elif L == 3:
            loss = torch.nn.SmoothL1Loss(reduction='none', beta=beta)

        def inner(pred, target, evaluation=False):

            assert len(pred.shape) == 3 and pred.shape[-1] == 3

            if L == 1:
                count = torch.abs(pred - target)
            elif L == 3:
                # diff = pred - target
                pred = torch.einsum('bjk,jk->bjk', pred, coef)
                target = torch.einsum('bjk,jk->bjk', target, coef)
                count = loss(pred, target)
                count = torch.einsum('bjk,jk->bjk', count, torch.div(1.0, coef))
            elif L == 2:
                count = loss(pred, target)
            else:
                assert False

            if weights is not None:
                count = torch.einsum('bij,ij->bij', count, weights)
            count = torch.sum(count, dim=-1)

            # averaged by the number of bones
            count = torch.mean(count, dim=-1)

            if evaluation:
                return torch.sum(count)
            else:
                # averaged by batch size
                return torch.mean(count)

        return inner

    @staticmethod
    def sequence_bone_scale_loss(L, coef=None, beta=0.5):

        if L == 2:
            loss = torch.nn.MSELoss(reduction='none')
        elif L == 3:
            loss = torch.nn.SmoothL1Loss(reduction='none', beta=beta)

        def inner(pred, target, evaluation=False):

            assert len(pred.shape) == 4 and pred.shape[-1] == 3
            assert len(target.shape) == 4 and target.shape[-1] == 3

            if L == 1:
                count = torch.linalg.norm(pred - target, ord=1, dim=len(target.shape) - 1)
            elif L == 3:
                # diff = pred - target
                pred = torch.einsum('bsjk,jk->bsjk', pred, coef)
                target = torch.einsum('bsjk,jk->bsjk', target, coef)
                count = loss(pred, target)
                count = torch.einsum('bsjk,jk->bsjk', count, torch.div(1.0, coef))
                count = torch.sum(count, dim=-1)
            elif L == 2:
                count = loss(pred, target)
                count = torch.sum(count, dim=-1)
            else:
                assert False

            # averaged by the number of bodies
            count = torch.mean(count, dim=-1)

            # averaged by the sequence length
            count = torch.mean(count, dim=-1)

            if evaluation:
                return torch.sum(count)
            else:
                # averaged by batch size
                return torch.mean(count)

        return inner

    @staticmethod
    def pose3d_mpjpe(root=True, L=2, weights=None, beta=0.04):

        if L == 3:
            loss = torch.nn.SmoothL1Loss(reduction='none', beta=beta)

        def inner(pred, target, mask=None, evaluation=False):

            assert (len(pred.shape) == 3 or len(pred.shape) == 4) and (pred.shape[-1] == 3 or pred.shape[-1] == 2)

            if root:
                # relative root distance
                pred = pred - pred[..., :1, :]
                target = target - target[..., :1, :]

            if L == 3:
                count = loss(pred, target)
                count = torch.sum(count, dim=-1)
            elif L == 1 or L == 2:
                count = torch.linalg.norm(pred - target, ord=L, dim=len(target.shape) - 1)
            else:
                assert False

            if weights is not None:
                count = torch.einsum('bj,j->bj', count, weights)
                # count = count * weightRatio

            if mask is not None:
                assert mask.shape == count.shape
                count = count * mask
                maskSum = torch.sum(mask, dim=-1)
                maskSum = (maskSum == 0) * 1.0 + (maskSum != 0) * maskSum
                count = torch.div(torch.sum(count, dim=-1), maskSum)
            else:
                # averaged by the number of key joints
                count = torch.mean(count, dim=-1)

            if evaluation:
                return torch.sum(count)
            else:
                # averaged by batch size
                return torch.mean(count)

        return inner

    @staticmethod
    def sequence_pose3d_mpjpe(root=True, L=2, weights=None, beta=0.04):

        if L == 3:
            loss = torch.nn.SmoothL1Loss(reduction='none', beta=beta)

        def inner(pred, target, mask=None, evaluation=False):

            assert len(pred.shape) == 4 and (pred.shape[-1] == 3 or pred.shape[-1] == 2)

            if root:
                # relative root distance
                pred = pred - pred[..., :1, :]
                target = target - target[..., :1, :]

            if L == 3:
                count = loss(pred, target)
                count = torch.sum(count, dim=-1)
            elif L == 1 or L == 2:
                count = torch.linalg.norm(pred - target, ord=L, dim=len(target.shape) - 1)
            else:
                assert False

            if weights is not None:
                count = torch.einsum('bsj,j->bsj', count, weights)
                # count = count * weightRatio

            if mask is not None:
                count = count * mask
                maskSum = torch.sum(mask, dim=-1)
                maskSum = (maskSum == 0) * 1.0 + (maskSum != 0) * maskSum
                count = torch.div(torch.sum(count, dim=-1), maskSum)
            else:
                # averaged by the number of key joints
                count = torch.mean(count, dim=-1)

            # averaged by the number of sequence length
            count = torch.mean(count, dim=-1)

            if evaluation:
                return torch.sum(count)
            else:
                # averaged by batch size
                return torch.mean(count)

        return inner

    @staticmethod
    def pos2d_align():

        def inner(pred, target):
            assert len(pred.shape) == 3 and len(target.shape) == 3

            mean_pred = CustomLoss.calculate_mean(pred)
            mean_target = CustomLoss.calculate_mean(target)
            pred = pred - mean_pred
            target = target - mean_target
            # = pred - pred[:, :1, :]
            # target = target - target[:, :1, :]

            # pred = pred - root_pred
            # target = target - root_target

            std_pred = CustomLoss.calcualte_std(pred)
            std_target = CustomLoss.calcualte_std(target)

            # pred = pred - mean_pred
            std_pred = torch.div(1, std_pred) + 10 ** -10
            pred = torch.einsum('bij , b -> bij', pred, std_pred)
            pred = torch.einsum('bij , b -> bij', pred, std_target)
            pred = pred + mean_target
            target = target + mean_target

            return [pred, target]

        return inner

    @staticmethod
    def calculate_mean(tensor):
        assert len(tensor.shape) == 3
        '''
        device = tensor.device
        
        means = torch.empty(tensor.shape[0], 1, tensor.shape[2]).to(device)
        for i in range(tensor.shape[2]):
            meanValue = torch.mean(tensor[:, :, i:i + 1], dim=-2, keepdim=True)
            means[:, :, i:i + 1] = meanValue
        '''
        means = torch.mean(tensor, dim=1, keepdim=True)
        return means

    @staticmethod
    def calcualte_std(tensor):
        assert len(tensor.shape) == 3

        tensor = tensor ** 2
        tensor = torch.sqrt(torch.sum(tensor, dim=-1))
        stds = torch.std(tensor, dim=-1)

        return stds
