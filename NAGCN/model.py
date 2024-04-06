import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
torch.backends.cudnn.enabled = False
class GCN(nn.Module):
     def __init__(self):
         """
         :param args: Arguments object.
         """
         super(GCN, self).__init__()

         self.encoder_cir = nn.Sequential(
             nn.Linear(663, 256),
             nn.ReLU(),
             nn.Linear(256, 64),
             nn.ReLU(),
             nn.Linear(64, 30),
             nn.ReLU(),

         )

         self.decoder_cir = nn.Sequential(
             nn.Linear(30, 32),
             nn.ReLU(),
             nn.Linear(32, 64),
             nn.ReLU(),
             nn.Linear(64, 64),
             nn.Sigmoid()
         )

         self.encoder_dis = nn.Sequential(
             nn.Linear(100, 64),
             nn.ReLU(),
             nn.Linear(64, 32),
             nn.ReLU(),
             nn.Linear(32, 30),
             nn.ReLU(),
         )

         self.decoder_dis = nn.Sequential(
             nn.Linear(30, 32),
             nn.ReLU(),
             nn.Linear(32, 64),
             nn.ReLU(),
             nn.Linear(64, 64),
             nn.ReLU()
         )

         self.gcn_cir1_f = GCNConv(64, 128)
         self.gcn_cir2_f = GCNConv(128, 128)

         self.gcn_dis1_f = GCNConv(64, 128)
         self.gcn_dis2_f = GCNConv(128, 128)

         self.cnn_cir = nn.Conv1d(in_channels=2,
                                  out_channels=256,
                                  kernel_size=(128, 1),
                                  stride=1,
                                  bias=True)
         self.cnn_dis = nn.Conv1d(in_channels=2,
                                  out_channels=256,
                                  kernel_size=(128, 1),
                                  stride=1,
                                  bias=True)

     def forward(self, data):
         # encoder
         x_cir = self.encoder_cir(data['cc']['data_matrix'])
         x_dis = self.encoder_dis(data['dd']['data_matrix'])
         # decoder
         x_cir = self.decoder_cir(x_cir)
         x_dis = self.decoder_dis(x_dis)

         x_cir_f1 = torch.relu(self.gcn_cir1_f(x_cir, data['cc']['edges'], data['cc']['data_matrix'][data['cc']['edges'][0], data['cc']['edges'][1]]))
         x_cir_f2 = torch.relu(self.gcn_cir2_f(x_cir_f1, data['cc']['edges'], data['cc']['data_matrix'][data['cc']['edges'][0], data['cc']['edges'][1]]))

         x_dis_f1 = torch.relu(self.gcn_dis1_f(x_dis, data['dd']['edges'], data['dd']['data_matrix'][data['dd']['edges'][0], data['dd']['edges'][1]]))
         x_dis_f2 = torch.relu(self.gcn_dis2_f(x_dis_f1, data['dd']['edges'], data['dd']['data_matrix'][data['dd']['edges'][0], data['dd']['edges'][1]]))

         X_cir = torch.cat((x_cir_f1, x_cir_f2), 1).t()
         X_cir = X_cir.view(1, 2,128, -1)

         X_dis = torch.cat((x_dis_f1, x_dis_f2), 1).t()
         X_dis = X_dis.view(1, 2, 128, -1)

         cir_fea = self.cnn_cir(X_cir)
         cir_fea = cir_fea.view(256, 663).t()
         dis_fea = self.cnn_dis(X_dis)
         dis_fea = dis_fea.view(256, 100).t()
         return cir_fea.mm(dis_fea.t()), cir_fea, dis_fea
