import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss

class LightGCN(GraphRecommender):
    def __init__(self, conf, training_set, test_set, valid_set, dislike_set):
        super(LightGCN, self).__init__(conf, training_set, test_set, valid_set, dislike_set)
        args = self.config['LightGCN']
        self.n_layers = 2
        self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers)
        #print(f"DEBUG: training_set type={type(training_set)}, value={training_set}")
        #print(f"DEBUG: test_set type={type(test_set)}, value={test_set}")

        # Ensure training_set and test_set are strings
        #train_file = training_set[0] if isinstance(training_set, list) else training_set
        #test_file = test_set[0] if isinstance(test_set, list) else test_set
        
        train_file = training_set[0] if isinstance(training_set, list) else str(training_set)
        test_file = test_set[0] if isinstance(test_set, list) else str(test_set)
        dislike_file = dislike_set[0] if isinstance(dislike_set, list) else str(dislike_set)

        
        # Load datasets
        df_train = pd.DataFrame(training_set, columns=['u', 'i', 'r'])
        df_test = pd.DataFrame(test_set, columns=['u', 'i', 'r'])
        df_dislike = pd.DataFrame(dislike_set, columns=['u', 'i', 'r'])  # 读取 dislike.txt
        
        # Create a combined dataset for ID mapping
        df_combined = pd.concat([df_train, df_test]).drop_duplicates()

        #sorted_item_ids = sorted(df_combined['i'].unique()) 
        
        # Create ID mappings
        self.user_id_mapping = {old_id: new_id for new_id, old_id in enumerate(df_combined['u'].unique())}
        self.item_id_mapping = {old_id: new_id for new_id, old_id in enumerate(df_combined['i'].unique())}
        #self.item_id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted_item_ids)}



        dislike_items = df_dislike['i'].unique()
        missing_items = set(dislike_items) - set(self.item_id_mapping.keys())
        if missing_items:
                    print(f"⚠️ Found {len(missing_items)} items in dislike.txt that are missing from train/test!")
                    start_index = max(self.item_id_mapping.values()) + 1  
                    for i, item in enumerate(missing_items):
                        self.item_id_mapping[item] = start_index + i
        
        
        # Save mappings
        with open('user_id_mapping.pkl', 'wb') as f:
            pickle.dump(self.user_id_mapping, f)
        with open('item_id_mapping.pkl', 'wb') as f:
            pickle.dump(self.item_id_mapping, f)
     
        with open('item_id_mapping-all.pkl', 'wb') as f:
            pickle.dump(self.item_id_mapping, f)
        print(f"✅ Mapping files saved! Total items (train+test+dislike): {len(self.item_id_mapping)}")

        # Create rating matrix
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
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, model.embedding_dict['user_emb'][user_idx],model.embedding_dict['item_emb'][pos_idx],model.embedding_dict['item_emb'][neg_idx])/self.batch_size
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0 and n > 0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = model()
            if epoch % 5 == 0:
                self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save_embeddings1(self):
        
        print("✅ Saving user & item embeddings after test()...")
       
        user_emb_array = self.user_emb.cpu().detach().numpy()
        item_emb_array = self.item_emb.cpu().detach().numpy()

        print(f"🔍 Maximum item index: {max(self.item_id_mapping.values())}")
        print(f"🔍 item_emb_array.shape: {item_emb_array.shape}")

       
        sorted_items = sorted(self.item_id_mapping.keys(), key=int)  
        sorted_users = sorted(self.user_id_mapping.keys(), key=int)
    
        print(f"🔍 Total users in mapping: {len(sorted_users)}, Embedding shape: {user_emb_array.shape}")
        print(f"🔍 Total items in mapping: {len(sorted_items)}, Embedding shape: {item_emb_array.shape}")

        
        if len(sorted_items) > item_emb_array.shape[0]:
            print(f"⚠️ Expanding item_emb_array from {item_emb_array.shape[0]} to {len(sorted_items)}")
            expanded_item_emb = np.zeros((len(sorted_items), item_emb_array.shape[1])) 
            expanded_item_emb[:item_emb_array.shape[0], :] = item_emb_array  
            item_emb_array = expanded_item_emb
        
   
        cm_user_dict = {str(user_id): user_emb_array[idx] for user_id, idx in self.user_id_mapping.items()}
        cm_item_dict = {str(item_id): item_emb_array[idx] for item_id, idx in self.item_id_mapping.items()}
        
       
        with open('user.pkl', 'wb') as f:
            pickle.dump(cm_user_dict, f) 
        with open('item.pkl', 'wb') as f:
            pickle.dump(cm_item_dict, f)  

        print("✅ Complete user & item embeddings are saved, including all data in training & test sets!")
        # Save predictions
        pred_matrix = torch.matmul(self.user_emb, self.item_emb.transpose(0, 1)).cpu().detach().numpy()
   
        cm_pred_dict = {
            (str(user_id), str(item_id)): pred_matrix[user_idx, item_idx]
            for user_id, user_idx in self.user_id_mapping.items()
            for item_id, item_idx in self.item_id_mapping.items()
        }
        with open('pred.pkl', 'wb') as f:
            pickle.dump(cm_pred_dict, f)

        print("✅ User & item embeddings saved!")
    def save_embeddings2(self):
        
        print("✅ Saving user & item embeddings after test()...")
        
      
        user_emb_array = self.user_emb.cpu().detach().numpy().astype(np.float32)
        item_emb_array = self.item_emb.cpu().detach().numpy().astype(np.float32)
    
        print(f"🔍 Maximum item index: {max(self.item_id_mapping.values())}")
        print(f"🔍 item_emb_array.shape: {item_emb_array.shape}")
    
       
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
    
        print("✅ Complete user & item embeddings are saved, including all data in training & test sets!")
    
    
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

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()
        print("✅ Only save embeddings at the best epoch")
        self.save_embeddings()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()
        
    def test(self):
        
        print("🔍 Running test...")
        rec_list = super().test()  
        self.save_embeddings()  
        return rec_list
    

class LGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings
