import torch
import torch.nn as nn
import torch.nn.functional as F
from util.conf import OptionConf
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
import pandas as pd
import pickle
import numpy as np


class XSimGCL(GraphRecommender):

    
    def __init__(self, conf, training_set, test_set, valid_set, dislike_set):
        super(XSimGCL, self).__init__(conf, training_set, test_set, valid_set, dislike_set)
        args = self.config['XSimGCL']
        
        if isinstance(args, str):
            args_list = args.split()  
            args_dict = {}
            for i in range(0, len(args_list), 2): 
                key = args_list[i].lstrip('-') 
                value = args_list[i + 1]
                try:
                    value = float(value) if '.' in value else int(value)  
                except ValueError:
                    pass  
                args_dict[key] = value
            args = args_dict  

        if not isinstance(args, dict):  
            raise TypeError(f"Expected args to be a dictionary, but got {type(args)}: {args}")
            
        self.cl_rate = float(args['lambda'])
        self.eps = float(args['eps'])
        self.temp = float(args['tau'])
        self.n_layers = int(args['n_layer'])
        self.layer_cl = int(args['l_star'])
        self.model = XSimGCL_Encoder(self.data, self.emb_size, self.eps, self.n_layers, self.layer_cl)

        train_file = training_set[0] if isinstance(training_set, list) else str(training_set)
        test_file = test_set[0] if isinstance(test_set, list) else str(test_set)
        dislike_file = dislike_set[0] if isinstance(dislike_set, list) else str(dislike_set)

        df_train = pd.DataFrame(training_set, columns=['u', 'i', 'r'])
        df_test = pd.DataFrame(test_set, columns=['u', 'i', 'r'])
        df_dislike = pd.DataFrame(dislike_set, columns=['u', 'i', 'r'])  
        df_combined = pd.concat([df_train, df_test]).drop_duplicates()
        self.user_id_mapping = {old_id: new_id for new_id, old_id in enumerate(df_combined['u'].unique())}
        self.item_id_mapping = {old_id: new_id for new_id, old_id in enumerate(df_combined['i'].unique())}
        
        dislike_items = df_dislike['i'].unique()
        missing_items = set(dislike_items) - set(self.item_id_mapping.keys())
        
        if 1522 in df_combined['i'].values:
            if 1522 in self.item_id_mapping:
                print(f"✅ Item 1522 is now in item_id_mapping with index {self.item_id_mapping[1522]}")
            else:
                print(f"⚠️ Item 1522 is in dataset but NOT in item_id_mapping!")
        else:
            print(f"❌ Item 1522 is NOT FOUND in df_combined['i']! It might be missing from training & test sets.")
        
        if missing_items:
                    print(f"⚠️ Found {len(missing_items)} items in dislike.txt that are missing from train/test!")
                    start_index = max(self.item_id_mapping.values()) + 1  
                    for i, item in enumerate(missing_items):
                        self.item_id_mapping[item] = start_index + i
        

        
        with open('user_id_mapping.pkl', 'wb') as f:
            pickle.dump(self.user_id_mapping, f)
        with open('item_id_mapping.pkl', 'wb') as f:
            pickle.dump(self.item_id_mapping, f)
        
        
        with open('item_id_mapping-all.pkl', 'wb') as f:
            pickle.dump(self.item_id_mapping, f)
        print(f"✅ Mapping files saved! Total items (train+test+dislike): {len(self.item_id_mapping)}")

        
        n_users = len(self.user_id_mapping)
        n_items = len(self.item_id_mapping)
        rating_matrix = np.zeros((n_users, n_items))
        
        for _, row in df_combined.iterrows():
            if row['u'] in self.user_id_mapping and row['i'] in self.item_id_mapping:  
                u = self.user_id_mapping[row['u']]
                i = self.item_id_mapping[row['i']]
                rating_matrix[u, i] = row['r']

        # Save rating matrix
        with open('rating_matrix.pkl', 'wb') as f:
            pickle.dump(rating_matrix, f)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb, cl_user_emb, cl_item_emb  = model(True)
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                cl_loss = self.cl_rate * self.cal_cl_loss([user_idx,pos_idx],rec_user_emb,cl_user_emb,rec_item_emb,cl_item_emb)
                batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    

    def cal_cl_loss(self, idx, user_view1,user_view2,item_view1,item_view2):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_cl_loss = InfoNCE(user_view1[u_idx], user_view2[u_idx], self.temp)
        item_cl_loss = InfoNCE(item_view1[i_idx], item_view2[i_idx], self.temp)
        return user_cl_loss + item_cl_loss


    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()
        print("✅ Only save embeddings at the best epoch")
        self.save_embeddings()
        
    def save_embeddings(self):
        
         
        test_items = set(self.data.test_set.keys())
        for item in test_items:
            if str(item) not in self.item_id_mapping:
                print(f"🛠 Adding missing test item {item} to item_id_mapping...")
                self.item_id_mapping[str(item)] = len(self.item_id_mapping)  

       
        
        
        
        print("✅ Saving user & item embeddings after test()...")
    
        
        user_emb_array = self.user_emb.cpu().detach().numpy().astype(np.float32)
        item_emb_array = self.item_emb.cpu().detach().numpy().astype(np.float32)
    
        sorted_items = sorted(self.item_id_mapping.keys(), key=int)
        sorted_users = sorted(self.user_id_mapping.keys(), key=int)
    
        print(f"🔍 Total users in mapping: {len(sorted_users)}, Embedding shape: {user_emb_array.shape}")
        print(f"🔍 Total items in mapping: {len(sorted_items)}, Embedding shape: {item_emb_array.shape}")
    
       
        if len(sorted_items) > item_emb_array.shape[0]:
            print(f"⚠️ Expanding item_emb_array from {item_emb_array.shape[0]} to {len(sorted_items)}")
            expanded_item_emb = np.zeros((len(sorted_items), item_emb_array.shape[1]), dtype=np.float32)
            expanded_item_emb[:item_emb_array.shape[0], :] = item_emb_array
            item_emb_array = expanded_item_emb 
        
       

    
       
        print("🔄 Updating prediction matrix to match all items...")
        pred_matrix = torch.matmul(self.user_emb, torch.tensor(item_emb_array, dtype=torch.float32).cuda().T).cpu().detach().numpy()
    
        if pred_matrix.shape[1] < len(sorted_items):
            print(f"⚠️ Expanding pred_matrix from {pred_matrix.shape[1]} to {len(sorted_items)}")
            expanded_pred_matrix = np.zeros((pred_matrix.shape[0], len(sorted_items)), dtype=np.float32)
            expanded_pred_matrix[:, :pred_matrix.shape[1]] = pred_matrix
            pred_matrix = expanded_pred_matrix 
        
        print(f"✅ New pred_matrix shape: {pred_matrix.shape}")
    
        
        cm_user_dict = {str(user_id): user_emb_array[idx] for user_id, idx in self.user_id_mapping.items()}
        cm_item_dict = {str(item_id): item_emb_array[idx] for item_id, idx in self.item_id_mapping.items()}
    
      
        with open('user.pkl', 'wb') as f:
            pickle.dump(cm_user_dict, f)
        with open('item.pkl', 'wb') as f:
            pickle.dump(cm_item_dict, f)
    
        print("✅ All training & testing items embeddings saved!")
    
        
        print("🔄 Creating cm_pred_dict...")
        cm_pred_dict = {
            (str(user_id), str(item_id)): pred_matrix[user_idx, item_idx]
            for user_id, user_idx in self.user_id_mapping.items()
            for item_id, item_idx in self.item_id_mapping.items()
            if item_idx < pred_matrix.shape[1]
        }
    
        print(f"✅ cm_pred_dict created with {len(cm_pred_dict)} entries")
    
        
        with open('pred.pkl', 'wb') as f:
            pickle.dump(cm_pred_dict, f)
    
        print("✅ User & item embeddings saved!")
        
    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()
    
    def test(self):
        
        print("🔍 Running test...")
        rec_list = super().test() 
        #self.save_embeddings()  
        return rec_list


class XSimGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers, layer_cl):
        super(XSimGCL_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.layer_cl = layer_cl
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def forward(self, perturbed=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        all_embeddings_cl = ego_embeddings
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
            if k==self.layer_cl-1:
                all_embeddings_cl = ego_embeddings
        final_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(final_embeddings, [self.data.user_num, self.data.item_num])
        user_all_embeddings_cl, item_all_embeddings_cl = torch.split(all_embeddings_cl, [self.data.user_num, self.data.item_num])
        if perturbed:
            return user_all_embeddings, item_all_embeddings,user_all_embeddings_cl, item_all_embeddings_cl
        return user_all_embeddings, item_all_embeddings
