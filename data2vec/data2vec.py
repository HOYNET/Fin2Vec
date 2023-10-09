import torch
import torch.nn as nn

# import torch.nn.functional as F
from .ema import EMA


class Data2vec(nn.Module):
    TYPES = ["finance"]

    def __init__(self, encoder: nn.Module, cfg, **kwargs):
        super(Data2vec, self).__init__()
        cfg = cfg["d2v"]["config"]
        self.type_ = cfg["type_"]
        self.embed_dim = cfg["embed_dim"]  # regression head에서 맞춰줄 embed dimension
        self.encoder = encoder
        self.__dict__.update(kwargs)

        self.ema = EMA(self.encoder, cfg)  # EMA acts as the teacher
        self.regression_head = self._build_regression_head()  # 여기에다가 전 embedding과정 넣어야함

        self.cfg = cfg
        self.ema_decay = self.cfg["ema_decay"]  # 논문 기준 시작하는 T값(0.999)
        self.ema_end_decay = self.cfg["ema_end_decay"]  # 논문 기준 시작하는 T값(0.9999)
        self.ema_anneal_end_step = self.cfg[
            "ema_anneal_end_step"
        ]  # 얼마에 걸쳐서 decay값을 변경할지

    def _build_regression_head(self):
        """
        여기는 regression 진행하기 전에 encoder의 embed 값이 나오면 regression을 위해
        처리하는 부분임. forward의 return 값을 직접 만져도 된다고 생각해서 지워둠.
        """
        if self.type_ == "finance":
            return
        else:
            return

    def ema_step(self):
        """
        One EMA step for the teacher model until the ending decay value is reached

        그니까 이건 그 T값 0.999에서 0.9999로 갱신하는 부분.

        그때 말한 대로 이건 써도되고 안써도 되는데,

        학습시에 loss 계산하고
        바로 다음에 self.model.ema_step() 이런식으로 쓰는 걸로 보여짐.

        있으니까 쓰는게 좋을듯?
        아니 써야만 함
        """
        if self.ema_decay != self.ema_end_decay:
            if self.ema.num_updates >= self.ema_anneal_end_step:
                decay = self.ema_end_decay
            else:
                decay = self.ema.get_annealed_rate(
                    self.ema_decay,
                    self.ema_end_decay,
                    self.ema.num_updates,
                    self.ema_anneal_end_step,
                )
            self.ema.decay = decay
        if self.ema.decay < 1:
            self.ema.step(self.encoder)

    def forward(self, src, idx, end, **kwargs):
        """
        Data2Vec forward method.

        Args:
            src: src tokens 즉, 마스킹한 student에 들어갈 data(masked inputs for training)
            trg: trg tokens 즉, 마스킹하지 않은 teacher에 들어갈 data (unmasked inputs for training but left as `None` otherwise)
            mask: bool masked indices, Note: if a modality requires the inputs to be masked before forward this param
            has no effect. (see the Encoder for each modality to see if it uses mask or not)

        Returns:
            x = student의 encoder mask 씌운부분 outputs
            y = teacher의 encoder mask 씌우지 않은 부분 outputs

        """
        # model forward in student mode
        x, mask = self.encoder(src=src, idx=idx, end=end, student=True)

        # model forward in teacher mode
        with torch.no_grad():
            self.ema.model.eval()
            y, _ = self.ema.model(src=src, idx=idx, end=end, student=False)

        # mask 씌운부분만 비교하기 위해서
        # x = x[mask]
        # y = y[mask]

        # 여기 필요 이유를 지금 embed 값에서 가용 데이터로 바꾸기 위한 부분이라고 이해
        # 근데 굳이 필요 없을듯? ㅇㅇ
        # x = self.regression_head(x)

        return x, y, mask
