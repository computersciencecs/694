import sys
import random
import pandas as pd
import jsonlines
import numpy as np
import copy
import pickle
from sklearn.cluster import DBSCAN


random.seed(42)
np.random.seed(42)
dataset_names = ['ml-100k']
isHint = True
sample_method = 'db'  # importance
for dataset_name in dataset_names:
    if dataset_name == 'ml-100k':
        df_like = pd.read_csv('./train_set.txt', names=['u', 'i', 'r', 't'], sep=' ')
        df_dislike = pd.read_csv('./dislike.txt', header=None, names=['u', 'i', 'r', 't'])
        movie_info = pd.read_csv('./movie_info.csv', header=None,
                                 names=['movie_id', 'movie_name', 'url', 'genre', 'genr0', 'genre1', 'genre2', 'genre3',
                                        'genre4', 'genre5', 'genre6', 'genre7', 'genre8', 'genre9', 'genre10',
                                        'genre11', 'genre12', 'genre13', 'genre14', 'genre15', 'genre16', 'genre17',
                                        'genre18', 'genre19'], sep='|', engine='python', encoding='latin-1')
        df_like_p = pd.read_csv('./train_set.txt', sep=' ', names=['u', 'i', 'r', 't'])
        print(df_like_p.head())
        movie_id_list = movie_info['movie_id'].tolist()
        movie_name_list = movie_info['movie_name'].tolist()
        movie_name_dict = {movie_id_list[i]: movie_name_list[i] for i in range(len(movie_name_list))}
    
    mes_list_pointwise = []
    mes_list_pairwise = []
    mes_list_pairwise_inv = []
    mes_list_listwise = []
    if 'book' in dataset_name:
        kws = 'book'
    else:
        kws = 'movie'


    def sort_list_reverse_with_indices(lst):
        sorted_indices = sorted(enumerate(lst), key=lambda x: x[1], reverse=True)
        sorted_indices = [index for index, _ in sorted_indices]
        return sorted_indices

    import pickle

    with open('./ml-100k_item.pkl', "rb") as file:
        cm_item = pickle.load(file)
    with open('./ml-100k_user.pkl', "rb") as file:
        cm_user = pickle.load(file)
    with open('./ml-100k_pred.pkl', "rb") as file:
        cm_pred = pickle.load(file)    
    with open('./ml-100k_item_id_mapping.pkl', "rb") as file:
        mf_item = pickle.load(file)
    with open('./ml-100k_item_id_mapping-all.pkl', "rb") as file:
        all_item = pickle.load(file)
    with open('./ml-100k_user_id_mapping.pkl', "rb") as file:
        mf_user = pickle.load(file)
    with open('./ml-100k_rating_matrix.pkl', "rb") as file:
        mf_pred = pickle.load(file)
    with open('./ml-100k_user.pkl', "rb") as file:
        cm_user_emb = pickle.load(file)


    print(f"🔍 Sample cm_item keys: {list(cm_item.keys())[:10]}")
    print(f"🔍 Does '1522' exist in cm_item? {'1522' in cm_item}")
    print(f"🔍 Sample cm_pred keys: {list(cm_pred.keys())[:10]}")
    print(f"🔍 Does ('some_user', '1522') exist in cm_pred? {('some_user', '1522') in cm_pred}")
    print(f"🔍 Total users in mf_user: {len(mf_user)}")
    print(f"🔍 Total items in mf_item: {len(mf_item)}")
    

    missing_users = ["655", "115", "26"]
    missing_items = ["143", "79", "936"]
    
    for user in missing_users:
        if user not in mf_user:
            print(f"⚠️ User {user} NOT FOUND in user_id_mapping!")
    
    for item in missing_items:
        if item not in mf_item:
            print(f"⚠️ Item {item} NOT FOUND in item_id_mapping!")
    

    ###print("\n🔍 Sample user mappings (first 10):", list(mf_user.items())[:10])
    ###print("🔍 Sample item mappings (first 10):", list(mf_item.items())[:10])
    
    mes_list = []
    gt_list = []


    if 'ml-1M' in dataset_name:
        sample_n = 1000
    else:
        sample_n = 1000
    user_list = list(df_like['u'].unique())
    sample_list = []
    import math

    weights = [math.log(len(df_like[df_like['u'] == uni])) for uni in user_list]


    print(type(cm_user_emb)) 
    print(len(cm_user_emb)) 
    print(next(iter(cm_user_emb.items()))) 


    

    if sample_method == 'uniform':
        for i in range(sample_n):
            sample_ = random.sample(user_list, 1)[0]
            sample_list.append(sample_)
    else:
        sample_list1 = []
        sample_list2 = []

        sample_imp = int(sample_n * 0.6)
        
        #1
        user_ids = sorted(cm_user_emb.keys(), key=int)
        cm_user_emb_matrix = np.array([cm_user_emb[user] for user in user_ids])
        weights = [math.log(len(df_like[df_like['u'] == int(user)])) for user in user_ids]

        
        for i in range(sample_imp):
            sample_ = random.choices(user_list, weights, k=1)[0]
            sample_list1.append(sample_)
        
        dbscan = DBSCAN(eps=0.5, min_samples=5, metric='euclidean', n_jobs=-1)
        labels = dbscan.fit_predict(cm_user_emb_matrix)
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        valid_clusters = unique_labels[unique_labels != -1]
        valid_counts = counts[unique_labels != -1]
        
        if len(valid_clusters) > 0:
            samples_per_cluster = np.round(valid_counts / valid_counts.sum() * sample_imp).astype(int)
        else:
            samples_per_cluster = []
        sampled_ids = []

        for cluster_id, samples in zip(valid_clusters, samples_per_cluster):
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_users = [user_ids[i] for i in cluster_indices] 
            sampled_ids.extend(np.random.choice(cluster_users, samples, replace=True))

        sample_list1.extend(sampled_ids)
        from collections import Counter

        occurrences = Counter(sample_list1)
        t_occurrences = {element: 0.95 ** (count - 1) for element, count in occurrences.items()}
        sample_list2 = [t_occurrences[_] for _ in sample_list1]

        sample_list = random.choices(sample_list1, weights=sample_list2, k=sample_n)


    for uni in sample_list:
        df = df_like[df_like['u'] == uni]
        df_un = df_dislike[df_dislike['u'] == uni]

        if len(df) > 1:

            '''
            Pointwise Ranking
            '''
            dfp = df_like_p[df_like_p['u'] == uni]
            if dfp.empty:
                #print(f"⚠️ Warning: No records found for user {uni} in df_like_p!")
            my_list = dfp['i'].tolist()
            my_list_r = dfp['r'].tolist()
    
            if not my_list:
                #print(f"⚠️ Warning: No items found for user {uni}!")

            rndl = [i_ for i_ in range(len(my_list))]
            random.shuffle(rndl)

            try:
                if not 'book' in dataset_name:
                    my_list = [int(my_list[x]) for x in rndl]
                    my_list_r = [int(my_list_r[x]) for x in rndl]
                else:
                    my_list = [(my_list[x]) for x in rndl]
                    my_list_r = [(my_list_r[x]) for x in rndl]
            except Exception as e:
                #print(f"⚠️ Error converting item IDs to int: {e}")
            if len(dfp) > 50:
                topk = 50
            else:
                #topk = len(dfp) - 3
                topk = max(5, len(dfp) - 3)  
            trainlist = my_list[:topk]
            trainlist_r = my_list_r[:topk]

            testlist = my_list[-1:]
            #mf_lable = "Unknown."
            if not testlist:
                #print(f"⚠️ Warning: Testlist is empty for user {uni} - Skipping user.")
                continue  

            testlist_r = my_list_r[-1:]

            yy = mf_item.get(str(testlist[0]), None)
            uu = mf_user.get(str(uni), None)


            if yy is None:
                #print(f"⚠️ Warning: Item {testlist[0]} not found in item mapping!")
            if uu is None:
                #print(f"⚠️ Warning: User {uni} not found in user mapping!")
            
            if yy is not None and uu is not None:
                try:
                    mf_lable = mf_pred[uu][yy]
                    ###print("get-mf_label-=-mf_pred[uu][yy]-pointwise")
                except IndexError:
                    print(f"⚠️ IndexError: ({uu}, {yy}) out `mf_pred`  {mf_pred.shape}!")
                    mf_lable = 'Unknown.' 
                except Exception as e:
                    print(f"⚠️ Unexpected error in fetching mf_lable: {e}")
                    mf_lable = 'Unknown.'
            else:
                mf_lable = 'Unknown.'
                print('mf_lableerro')
            
            historical_interactions = [f'"{movie_name_dict[i]}"' for i in trainlist]
            answer_items_set = [f'"{movie_name_dict[i]}"' for i in testlist]

            historical_interactions = [historical_interactions[i_] + ': ' + str(trainlist_r[i_]) + ';' for i_ in
                                       range(len(historical_interactions))]

            historical_interactions = ' '.join(historical_interactions)
            if 'book' in dataset_name:
                #22#highest_score = 5
                highest_score = 10
            else:
                #22#highest_score = 10
                highest_score = 5
            instruct0 = f'''You are a {kws} recommender system. Your task is to predict the relevance score to a target {kws} based on the user's historical {kws} ratings. The score should be between 1 and {highest_score}, where 1 is the lowest affinity and {highest_score} is the highest. Respond only with a number between 1 to {highest_score}.\n\n'''

            instruct1 = f'''User's historical {kws} ratings: <historical interactions>. \n\nQuestion: Based on the user's historical ratings, predict the relavance score of the target {kws} <movie> with the user.\n'''
            instruct2 = '''Hint: Another recommender system suggests the answer is <mf_prediction>"'''
            instruct3 = '\n\nPlease only output the score.\n'
            if isHint:
                instruct1 = instruct1 + instruct2
            instruct1 = instruct1.replace('<historical interactions>', historical_interactions).replace('<movie>', answer_items_set[0]).replace('<mf_prediction>', str(mf_lable)[:3])

            fi = {'messages': [
                {"role": "system", "content": [{"type": "text", "content": ""}]},
                {"role": "user", "content": [{"type": "text", "content": instruct0 + instruct1 + instruct3}]},
                {"role": "assistant", "content": [{"type": "text", "content": 'Answer: ' + str(testlist_r[0])}]}
            ]}
            mes_list.append(fi)
            mes_list_pointwise.append(fi)

            '''
            Pairwise Ranking
            '''
            unlikelist = []
            if len(df_un) > 0:
                my_list = df_un['i'].tolist()
                random.shuffle(my_list)

                if not 'book' in dataset_name:
                    my_list = [int(x) for x in my_list]
                else:
                    my_list = [(x) for x in my_list]

                unlikelist = my_list[:10]

            my_list = df['i'].tolist()
            random.shuffle(my_list)

            if not 'book' in dataset_name:
                my_list = [int(x) for x in my_list]
            else:
                my_list = [(x) for x in my_list]

            if len(df) > 55:
                topk = 50
            else:
                topk = len(df) - 3
            trainlist = my_list[:topk]
            testlist = my_list[-1:]

            if not trainlist:
                print(f"⚠️ Warning: Trainlist is empty for user {uni} - Skipping user.")
                continue  
                
            neglist = []
            while len(neglist) < 1:
                rn = random.sample(movie_id_list, 1)[0]
                if not rn in my_list:
                    neglist.append(rn)

            random_n = (random.random() > 0.5)


  
            if not trainlist:
                print(f"⚠️ Warning: Trainlist is empty for user {uni}!")
            if not neglist:
                print(f"⚠️ Warning: Neglist is empty for user {uni}!")
            if not testlist:
                print(f"⚠️ Warning: Testlist is empty for user {uni}!")
            
            historical_interactions = [f'"{movie_name_dict[i]}"' for i in trainlist]

            false_items_set = [f'"{movie_name_dict[i]}"' for i in neglist]

            answer_items_set = [f'"{movie_name_dict[i]}"' for i in testlist]
  
            if not historical_interactions:
                print(f"⚠️ Warning: historical_interactions is empty for user {uni}!")
            if not false_items_set:
                print(f"⚠️ Warning: false_items_set is empty for user {uni}!")
            if not answer_items_set:
                print(f"⚠️ Warning: answer_items_set is empty for user {uni}!")

            #print(f"🔍 Type of cm_pred: {type(cm_pred)}")
            #if isinstance(cm_pred, dict):
                #print(f"🔍 Sample keys in cm_pred: {list(cm_pred.keys())[:10]}")
            #elif isinstance(cm_pred, np.ndarray):
                #print(f"🔍 Shape of cm_pred: {cm_pred.shape}")

            sample_cm_item_keys = list(cm_item.keys())[:10]
            ####print(f"🔍 Sample cm_item keys: {sample_cm_item_keys}")
            ####print(f"🔍 Type of first cm_item key: {type(sample_cm_item_keys[0])}")
            #print(f"🔍 Type of neglist[0]: {type(neglist[0])} | Value: {neglist[0]}")
            #print(f"🔍 Type of testlist[0]: {type(testlist[0])} | Value: {testlist[0]}")
            #print(f"🔍 Checking if {neglist[0]} exists in cm_item keys: {'Yes' if str(neglist[0]) in cm_item else 'No'}")
            #print(f"🔍 Checking if {testlist[0]} exists in cm_item keys: {'Yes' if str(testlist[0]) in cm_item else 'No'}")


            
            try:
                xx = cm_item[str(neglist[0])]
                yy = cm_item[str(testlist[0])]
                #xx = cm_item.get(str(neglist[0]), None)
                #yy = cm_item.get(str(testlist[0]), None)
                #uu = cm_user[str(uni)]
                uu = cm_user.get(str(uni), None)
            
                if xx is None:
                    #print(f"⚠️ Warning: Item {neglist[0]} not found in neglist cm_item!")
                if yy is None:
                    #print(f"⚠️ Warning: Item {testlist[0]} not found in testlist cm_item!")
                if uu is None:
                    #print(f"⚠️ Warning: User {uni} not found in cm_user!")

                if xx is not None and yy is not None and uu is not None:
                    #if cm_pred[uu][yy] > cm_pred[uu][xx]:
                    if cm_pred.get((str(uu), str(yy)), 0) > cm_pred.get((str(uu), str(xx)), 0):
                        cm_lable = 'Yes.'
                    else:
                        cm_lable = 'No.'
                else:
                    cm_lable = 'Unknown.'
                    print('Unknown')
            except IndexError as e:
                print(f"⚠️ IndexError: {e}")
                cm_lable = 'Unknown.'
            except Exception as e:
                ####print(f"⚠️ Unexpected error: {e} in pairwise")
                cm_lable = 'Unknown.'

            unlikelist = [x_ for x_ in unlikelist if x_ in movie_name_dict.keys()]

            user_unpre = [f'"{movie_name_dict[i]}"' for i in unlikelist]

            if len(unlikelist) < 3:
                user_unpre = 'None.'
            else:
                user_unpre = ', '.join(user_unpre)

            historical_interactions = ', '.join(historical_interactions)
            gt_list.append(movie_name_dict[testlist[0]])

            if random_n:
                first_name = answer_items_set[0]
                second_name = false_items_set[0]
                tg = 'Yes.'
            else:
                first_name = false_items_set[0]
                second_name = answer_items_set[0]
                tg = 'No.'

                if cm_lable == 'Yes.':
                    cm_lable = 'No.'
                elif cm_lable == 'No.':
                    cm_lable = 'Yes.'

            instruct0 = f'''You are a {kws} recommender system. Based on a user's likes and dislikes, determine if they would prefer one {kws} over another. Respond only with "Yes." or "No.".\n\n'''

            instruct1 = f'''User's Liked {kws}s: <historical interactions>. \nUser's Disliked {kws}s: <user_unpre>\n\nQuestion: Would the user prefer <movie1> over <movie2>?\n'''
            instruct2 = '''Hint: Another recommender system suggests the answer is "<cm_result>"'''
            instruct3 = '\n\nPlease only output "Yes." or "No.".\n'
            if isHint:
                instruct1 = instruct1 + instruct2
            instruct1 = instruct1.replace('<historical interactions>', historical_interactions).replace('<user_unpre>',
                                                                                                        user_unpre).replace(
                '<movie1>', first_name).replace('<movie2>', second_name).replace('<cm_result>', cm_lable)

            fi = {'messages': [
                {"role": "system", "content": [{"type": "text", "content": ""}]},
                {"role": "user", "content": [{"type": "text", "content": instruct0 + instruct1 + instruct3}]},
                {"role": "assistant", "content": [{"type": "text", "content": 'Answer: ' + tg}]}
            ]}
            mes_list.append(fi)
            mes_list_pairwise.append(fi)

            '''
            Listwise Ranking
            '''

            my_list = dfp['i'].tolist()
            my_list_r = dfp['r'].tolist()
            if not 'book' in dataset_name:
                my_list = [int(my_list[x]) for x in rndl]
                my_list_r = [int(my_list_r[x]) for x in rndl]
            else:
                my_list = [(my_list[x]) for x in rndl]
                my_list_r = [(my_list_r[x]) for x in rndl]

            if not 'book' in dataset_name:
                largest_rating = 5
                second_largest_rating = 4
            else:
                largest_rating = 10
                second_largest_rating = 9

            num_5 = my_list_r.count(largest_rating)
            num_4 = my_list_r.count(second_largest_rating)

            if (not (num_5 > 3 and num_4 > 2)) and 'book' in dataset_name:
                largest_rating = 9
                second_largest_rating = 8
                num_5 = my_list_r.count(largest_rating)
                num_4 = my_list_r.count(second_largest_rating)

            if num_5 > 3 and num_4 > 2:
                idx_5 = [index for index, value in enumerate(my_list) if my_list_r[index] == largest_rating]
                idx_4 = [index for index, value in enumerate(my_list) if my_list_r[index] == second_largest_rating]

                select_5 = random.sample(idx_5, 3)
                select_4 = random.sample(idx_4, 2)

                select_5_i = [my_list[__] for __ in select_5]
                select_4_i = [my_list[__] for __ in select_4]

                user_like_list = []
                user_dislike_list = []
                for i_ in range(len(my_list)):
                    if my_list_r[i_] >= 4 and not (my_list[i_] in select_5 or my_list[i_] in select_4):
                        user_like_list.append(my_list[i_])
                    elif my_list_r[i_] <= 2:
                        user_dislike_list.append(my_list[i_])

                if len(user_like_list) > 55:
                    topk = 50
                user_like_list = user_like_list[:topk]
                user_dislike_list = user_dislike_list[:10]

                neglist = []
                while len(neglist) < 5:
                    rn = random.sample(movie_id_list, 1)[0]
                    if not (rn in my_list or rn in neglist):
                        neglist.append(rn)

                total_list = select_5_i + select_4_i + neglist

                total_list_mf = []
                for j_ in total_list:
                    try:
                        yy = cm_item[str(j_)]
                        ##yy = cm_item.get(str(j_), None)  # 确保 key 是 str
                        #uu = cm_user[str(uni)]
                        uu = cm_user.get(str(uni), None)
                        if yy is None:
                            print(f"⚠️ Warning: Item {j_} not found in cm_item!")
                        if uu is None:
                            print(f"⚠️ Warning: User {uni} not found in cm_user!")

                        #str_yy = str(yy)
                        int_yy = int(j_)
                        #str_uu = str(uu)
                        int_uu = int(uni)
                
                     
                        if int_uu < mf_pred.shape[0] and int_yy < mf_pred.shape[1]:
                        #if str_yy in mf_pred and str_uu in mf_pred[str_yy]:
                            mf_label = mf_pred[int_uu][int_yy]
                            ######print(f"✅ Found prediction for (User: {uni} -> {int_uu}, Item: {j_} -> {int_yy}) -> {mf_label}")
                        else:
                            mf_label = 1.5
                            print(f"⚠️ Warning: Prediction not found for (User: {uni}, Item: {j_}) in mf_pred!")
                
                    except KeyError:
                        print(f"❌ KeyError: Item {j_} or User {uni} not found in cm_item/cm_user!")
                        mf_label = 1.5
                    except IndexError as e:
                        print(f"❌ IndexError: {e} - Possible out-of-bounds in `mf_pred`!")
                        mf_label = 1.5
                    except ValueError as e:
                        print(f"❌ ValueError: {e} - `yy` or `uu` conversion issue!")
                        mf_label = 1.5
                    except Exception as e:
                        print(f"❌ Unexpected error: {e}")
                        mf_label = 1.5
                
                    total_list_mf.append(mf_label)                       
                        
                        
                        
                        #mf_label = mf_pred[uu][yy]
                        #print('找到')
                    #except Exception:
                        #mf_label = 1.5
                        #print('meizhaodao')
                    #total_list_mf.append(mf_label)

                total_list_mf_idx = sort_list_reverse_with_indices(total_list_mf)
                total_list_mf_idx = total_list_mf_idx[:5]
                total_list_mf_i = [total_list[k_] for k_ in total_list_mf_idx]

                total_list_r = copy.deepcopy(total_list)
                random.shuffle(total_list_r)
                total_list_t = total_list[:5]

                historical_interactions = ', '.join([f'"{movie_name_dict[i]}"' for i in user_like_list])
                neg_interactions = ', '.join([f'"{movie_name_dict[i]}"' for i in user_dislike_list])
                true_answer_items_set = ', '.join([f'"{movie_name_dict[i]}"' for i in total_list_t])
                candidate_item_sets_ = ', '.join([f'"{movie_name_dict[i]}"' for i in total_list_r])
                mf_item_sets_ = ', '.join([f'"{movie_name_dict[i]}"' for i in total_list_mf_i])

                instruct0 = f'''You are a {kws} recommender system. Your task is to rank a given list of candidate {kws}s based on user preferences and return the top five recommendations.\n\n'''

                instruct1 = f'''User's Liked {kws}s: <historical_interactions>. \nUser's Disliked {kws}s: <user_unpre>\n\nQuestion: How would the user rank the candidate item list: <movie_list> based to historical perference?\n'''
                instruct2 = 'Hint: Another recommender model suggests <cm_result>'
                instruct3 = '\n\nPlease only output the top five recommended movies once in the following format:\n1. [Movie Title]\n2. [Movie Title]\n3. [Movie Title]\n4. [Movie Title]\n5. [Movie Title].\n'
                if isHint:
                    instruct1 = instruct1 + instruct2

                instruct1 = instruct1.replace('<historical_interactions>', historical_interactions).replace(
                    '<user_unpre>', neg_interactions).replace('<movie_list>', candidate_item_sets_).replace(
                    '<cm_result>', mf_item_sets_)

                fi = {'messages': [
                    {"role": "system", "content": [{"type": "text", "content": ""}]},
                    {"role": "user", "content": [{"type": "text", "content": instruct0 + instruct1 + instruct3}]},
                    {"role": "assistant",
                     "content": [{"type": "text", "content": 'Answer: ' + str(true_answer_items_set)}]}
                ]}
                mes_list.append(fi)
                mes_list_listwise.append(fi)

            else:
                continue


    with jsonlines.open(f'./pointwisedd.jsonl', mode='w') as writer:
        writer.write_all(mes_list_pointwise)
    with jsonlines.open(f'./pairwisedd.jsonl', mode='w') as writer:
        writer.write_all(mes_list_pairwise)
    #with jsonlines.open(f'./test_ml-1m_lightKG_pairwise_inv-pre-3kg.jsonl', mode='w') as writer:
        #writer.write_all(mes_list_pairwise_inv)
    with jsonlines.open(f'./listwisedd.jsonl', mode='w') as writer:
        writer.write_all(mes_list_listwise)
    with jsonlines.open(f'/data_alldd.jsonl', mode='w') as writer:
        writer.write_all(mes_list)
