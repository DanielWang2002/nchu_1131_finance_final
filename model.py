# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# 通道注意力模塊
class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)

    def forward(self, x):
        avg_out = self.fc2(F.relu(self.fc1(self.global_avg_pool(x))))
        max_out = self.fc2(F.relu(self.fc1(self.global_max_pool(x))))
        out = torch.sigmoid(avg_out + max_out)
        return x * out


# LSTM 注意力模塊
class LSTMAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(LSTMAttention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        att_weights = F.softmax(self.attention(lstm_output), dim=1)  # 注意力權重
        context = torch.sum(att_weights * lstm_output, dim=1)  # 加權和
        return context


# LSTM + CNN 模型
class LSTM_CNN_Model(nn.Module):
    def __init__(
        self,
        lstm_input_dim,
        lstm_hidden_dim,
        lstm_num_layers,
        cnn_input_channels,
        cnn_output_channels,
        cnn_kernel_size,
        cnn_fc_size,
    ):
        super(LSTM_CNN_Model, self).__init__()
        # CNN
        self.cnn_input_channels = cnn_input_channels
        self.conv1 = nn.Conv2d(cnn_input_channels, cnn_output_channels, kernel_size=cnn_kernel_size)
        self.conv2 = nn.Conv2d(
            cnn_output_channels, cnn_output_channels, kernel_size=cnn_kernel_size
        )
        self.channel_attention = ChannelAttention(cnn_output_channels)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
        )
        self.lstm_attention = LSTMAttention(lstm_hidden_dim)

        # 計算 CNN 輸出大小
        self.cnn_output_size = self.compute_cnn_output_size((64, 64))

        # 全連接層
        self.fc_cnn = nn.Linear(self.cnn_output_size, cnn_fc_size)
        self.fc_final = nn.Linear(lstm_hidden_dim + cnn_fc_size, 1)

    def compute_cnn_output_size(self, img_size):
        dummy_input = torch.randn(1, self.cnn_input_channels, img_size[0], img_size[1])
        with torch.no_grad():
            output = self.conv1(dummy_input)
            output = self.conv2(output)
            output = self.channel_attention(output)
            output = output.view(output.size(0), -1)  # Flatten
        return output.size(1)

    def forward(self, x_lstm, x_cnn):
        # CNN 分支
        cnn_out = F.relu(self.conv1(x_cnn))
        cnn_out = F.relu(self.conv2(cnn_out))
        cnn_out = self.channel_attention(cnn_out)
        cnn_out = cnn_out.view(cnn_out.size(0), -1)  # Flatten
        cnn_out = F.relu(self.fc_cnn(cnn_out))

        # LSTM 分支
        lstm_out, _ = self.lstm(x_lstm)
        lstm_out = self.lstm_attention(lstm_out)  # 應用注意力機制

        # 合併 CNN 和 LSTM 的輸出
        combined = torch.cat((cnn_out, lstm_out), dim=1)
        output = self.fc_final(combined)
        return output
