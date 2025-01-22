# import math
#
# from torch import BoolTensor, Tensor, nn
# from torch.nn import Sequential
#
#
# class MaskConv(nn.Module):
#     def __init__(self, seq_module: nn.Sequential):
#         super().__init__()
#         self.seq_module = seq_module
#
#     def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
#         for module in self.seq_module:
#             x = module(x)
#             mask = BoolTensor(x.size()).fill_(0)
#             if x.is_cuda:
#                 mask = mask.cuda()
#             for i, length in enumerate(lengths.cpu().tolist()):
#                 if (mask[i].size(2) - length) > 0:
#                     mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
#             x = x.masked_fill(mask, 0)
#         return x
#
#
# class BatchRNN(nn.Module):
#     def __init__(
#         self,
#         input_size: int,
#         hidden_size: int,
#         rnn_type: nn.Module = nn.GRU,
#         bidirectional: bool = True,
#         dropout: float = 0.0,
#     ):
#         super().__init__()
#
#         self.batch_norm = nn.BatchNorm1d(input_size)
#         self.rnn = rnn_type(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             bidirectional=bidirectional,
#             dropout=dropout,
#         )
#         self.batch_norm = nn.BatchNorm1d(input_size)
#         self.bidirectional = bidirectional
#
#     def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
#         x = self.batch_norm(x.permute(1, 2, 0)).permute(2, 0, 1)  # TxBxC
#         x = nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False)
#         x, _ = self.rnn(x)
#         x, _ = nn.utils.rnn.pad_packed_sequence(x)
#         sizes = x.size()
#         if self.bidirectional:
#             x = x.view(sizes[0], sizes[1], 2, -1).sum(2).view(sizes[0], sizes[1], -1)
#
#         return x
#
#
# class DeepSpeech2Model(nn.Module):
#     def __init__(
#         self,
#         n_feats: int,
#         n_tokens: int,
#         conv_channels: int = 32,
#         n_rnn_layers: int = 3,
#         rnn_hidden_size: int = 256,
#         rnn_type: nn.Module = nn.GRU,
#         rnn_bidirectional: bool = True,
#         rnn_dropout: float = 0.0,
#         fc_hidden_size: int = 1024,
#     ):
#         """
#         Args:
#             n_feats (int): number of input features.
#             n_tokens (int): number of tokens in the vocabulary.
#             fc_hidden (int): number of hidden features.
#         """
#         super().__init__()
#
#         self.conv = MaskConv(
#             Sequential(
#                 nn.Conv2d(
#                     in_channels=1,
#                     out_channels=conv_channels,
#                     kernel_size=(41, 11),
#                     stride=(2, 2),
#                     padding=(20, 5),
#                 ),
#                 nn.BatchNorm2d(32),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(
#                     in_channels=conv_channels,
#                     out_channels=conv_channels,
#                     kernel_size=(21, 11),
#                     stride=(2, 1),
#                     padding=(10, 5),
#                 ),
#                 nn.BatchNorm2d(32),
#                 nn.ReLU(inplace=True),
#             )
#         )
#
#         self.n_feats = math.floor((n_feats + 1) / 2)
#         self.n_feats = math.floor((self.n_feats + 1) / 2)
#         self.n_feats *= conv_channels  # * 2
#
#         self.rnn = nn.Sequential(
#             BatchRNN(
#                 self.n_feats,
#                 rnn_hidden_size,
#                 rnn_type,
#                 rnn_bidirectional,
#                 dropout=rnn_dropout,
#             ),
#             *[
#                 BatchRNN(
#                     rnn_hidden_size,
#                     rnn_hidden_size,
#                     rnn_type,
#                     rnn_bidirectional,
#                     dropout=rnn_dropout,
#                 )
#                 for _ in range(n_rnn_layers - 1)
#             ],
#         )
#         self.fc_batchnorm = nn.BatchNorm1d(rnn_hidden_size)
#         self.fc = nn.Sequential(
#             nn.Linear(rnn_hidden_size, fc_hidden_size),  # turn off bias
#             nn.ReLU(inplace=True),
#             nn.Linear(fc_hidden_size, fc_hidden_size),
#             nn.ReLU(inplace=True),
#             nn.Linear(fc_hidden_size, n_tokens),
#         )
#
#     def forward(self, spectrogram: Tensor, spectrogram_length: Tensor, **batch):
#         spectrogram = spectrogram[:, None, :, :]  # Bx1xDxT
#         spectrogram_length = self.transform_input_lengths(
#             spectrogram_length.cpu().int()
#         )
#         x = self.conv(spectrogram, spectrogram_length)  # BxCxDxT
#         sizes = x.size()
#         x = x.view(sizes[0], -1, sizes[3])
#         x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxBxC
#
#         for layer in self.rnn:
#             x = layer(x, spectrogram_length)  # TxBxC
#
#         x = self.fc_batchnorm(x.permute(1, 2, 0)).permute(0, 2, 1)  # BxTxC
#         output = self.fc(x)
#
#         log_probs = nn.functional.log_softmax(output, dim=-1)
#         probs = nn.functional.softmax(output, dim=-1)
#         return {
#             "log_probs": log_probs,
#             "log_probs_length": spectrogram_length,
#             "probs": probs,
#             "probs_length": spectrogram_length,
#         }
#
#     def transform_input_lengths(self, input_lengths: Tensor):
#         seq_len = input_lengths
#         for m in self.conv.modules():
#             if type(m) == nn.modules.conv.Conv2d:
#                 seq_len = (
#                     seq_len
#                     + 2 * m.padding[1]
#                     - m.dilation[1] * (m.kernel_size[1] - 1)
#                     - 1
#                 ) // m.stride[1] + 1
#         return seq_len.int()
#
#     def __str__(self):
#         all_parameters = sum([p.numel() for p in self.parameters()])
#         trainable_parameters = sum(
#             [p.numel() for p in self.parameters() if p.requires_grad]
#         )
#
#         result_info = super().__str__()
#         result_info = result_info + f"\nAll parameters: {all_parameters}"
#         result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"
#
#         return result_info

import math
from torch import BoolTensor, Tensor, nn
from torch.nn import Sequential

class MaskConv(nn.Module):
    def __init__(self, seq_module: nn.Sequential):
        super().__init__()
        self.seq_module = seq_module

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        for module in self.seq_module:
            x = module(x)
            mask = BoolTensor(x.size()).fill_(0)
            if x.is_cuda:
                mask = mask.cuda()
            for i, length in enumerate(lengths.cpu().tolist()):
                if (mask[i].size(2) - length) > 0:
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
            x = x.masked_fill(mask, 0)
        return x

class BatchRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        rnn_type: nn.Module = nn.GRU,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_size)  # Применяем BatchNorm1d к num_features
        self.rnn = rnn_type(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.bidirectional = bidirectional

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        # x shape: (sequence_length, batch_size, num_features)
        x = x.permute(1, 2, 0)  # (batch_size, num_features, sequence_length)
        x = self.batch_norm(x)  # Применяем BatchNorm1d к num_features
        x = x.permute(2, 0, 1)  # (sequence_length, batch_size, num_features)

        # Упаковываем последовательность
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False)
        x, _ = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)

        # Если RNN двунаправленный, суммируем выходы
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2)  # Суммируем по направлениям

        return x

class DeepSpeech2Model(nn.Module):
    def __init__(
        self,
        n_feats: int,
        n_tokens: int,
        conv_channels: int = 16,
        n_rnn_layers: int = 2,
        rnn_hidden_size: int = 128,
        rnn_type: nn.Module = nn.GRU,
        rnn_bidirectional: bool = False,
        rnn_dropout: float = 0.0,
        fc_hidden_size: int = 512,
    ):
        super().__init__()
        self.conv = MaskConv(
            Sequential(
                nn.Conv2d(1, conv_channels, kernel_size=(21, 5), stride=(2, 2), padding=(10, 2)),
                nn.BatchNorm2d(conv_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(conv_channels, conv_channels, kernel_size=(11, 5), stride=(2, 1), padding=(5, 2)),
                nn.BatchNorm2d(conv_channels),
                nn.ReLU(inplace=True),
            )
        )
        self.n_feats = math.floor((n_feats + 1) / 2)
        self.n_feats = math.floor((self.n_feats + 1) / 2)
        self.n_feats *= conv_channels

        self.rnn = nn.Sequential(
            BatchRNN(
                self.n_feats,
                rnn_hidden_size,
                rnn_type,
                rnn_bidirectional,
                dropout=rnn_dropout,
            ),
            *[
                BatchRNN(
                    rnn_hidden_size,
                    rnn_hidden_size,
                    rnn_type,
                    rnn_bidirectional,
                    dropout=rnn_dropout,
                )
                for _ in range(n_rnn_layers - 1)
            ],
        )
        self.fc = nn.Sequential(
            nn.Linear(rnn_hidden_size, fc_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden_size, n_tokens),
        )

    def forward(self, spectrogram: Tensor, spectrogram_length: Tensor, **batch):
        spectrogram = spectrogram[:, None, :, :]  # Bx1xDxT
        spectrogram_length = self.transform_input_lengths(spectrogram_length.cpu().int())
        x = self.conv(spectrogram, spectrogram_length)  # BxCxDxT
        sizes = x.size()
        x = x.view(sizes[0], -1, sizes[3])
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxBxC

        for layer in self.rnn:
            x = layer(x, spectrogram_length)  # TxBxC

        x = x.permute(1, 2, 0)  # BxCxT
        output = self.fc(x.permute(0, 2, 1))  # BxTxC

        log_probs = nn.functional.log_softmax(output, dim=-1)
        probs = nn.functional.softmax(output, dim=-1)
        return {
            "log_probs": log_probs,
            "log_probs_length": spectrogram_length,
            "probs": probs,
            "probs_length": spectrogram_length,
        }

    def transform_input_lengths(self, input_lengths: Tensor):
        seq_len = input_lengths
        for m in self.conv.modules():
            if isinstance(m, nn.Conv2d):
                seq_len = (
                    seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1
                ) // m.stride[1] + 1
        return seq_len.int()

    def __str__(self):
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum([p.numel() for p in self.parameters() if p.requires_grad])
        result_info = super().__str__()
        result_info += f"\nAll parameters: {all_parameters}"
        result_info += f"\nTrainable parameters: {trainable_parameters}"
        return result_info
