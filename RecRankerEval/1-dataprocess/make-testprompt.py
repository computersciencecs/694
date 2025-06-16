import sys
import random
import pandas as pd
import jsonlines
import numpy as np
import copy
import pickle

random.seed(42)
np.random.seed(42)

#dataset_names = ['ml-100k', 'bookcrossing', 'ml-1M']
dataset_names = ['ml-100k']
model_names = ['LightGCN']
#model_names = ['MF', 'LightGCN', 'SGL','SASRec', 'CL4SRec', 'BERT4Rec']
isHint = True


for dataset_name in dataset_names:
    
    with open('./item.pkl', "rb") as file:
        cm_item = pickle.load(file)
    with open('./user.pkl', "rb") as file:
        cm_user = pickle.load(file)
    with open('./pred.pkl', "rb") as file:
        cm_pred = pickle.load(file)    
    with open('./item_id_mapping.pkl', "rb") as file:
        mf_item = pickle.load(file)
    with open('./item_id_mapping-all.pkl', "rb") as file:
        all_item = pickle.load(file)
    with open('./user_id_mapping.pkl', "rb") as file:
        mf_user = pickle.load(file)
    with open('./rating_matrix.pkl', "rb") as file:
        mf_pred = pickle.load(file)
    with open('./user.pkl', "rb") as file:
        cm_user_emb = pickle.load(file)


    def sort_list_reverse_with_indices(lst):
        sorted_indices = sorted(enumerate(lst), key=lambda x: x[1], reverse=True)
        sorted_indices = [index for index, _ in sorted_indices]
        return sorted_indices


    if 'book' in dataset_name:
        kws = 'book'
    else:
        kws = 'movie'

    kk = 5

    for model_name in model_names:

        idd = '1'
        rec_list = pd.read_csv('./LightGCNrec_save_dict2.csv', header=None,
                               names=['v' + str(i) for i in range(11)])
        gt_list = pd.read_csv('./LightGCNgt_save_dict2.csv', header=None,
                              names=['u', 'i'])

        mov_id = gt_list['i'].tolist()


        if dataset_name == 'ml-100k':
            df_like = pd.read_csv('./train_set.txt', names=['u', 'i', 'r', 't'], sep=' ')
            df_dislike = pd.read_csv('./dislike.txt', header=None, names=['u', 'i', 'r', 't'])
            movie_info = pd.read_csv('./movie_info.csv', header=None,
                                     names=['movie_id', 'movie_name', 'url', 'genre', 'genr0', 'genre1', 'genre2',
                                            'genre3', 'genre4', 'genre5', 'genre6', 'genre7', 'genre8', 'genre9',
                                            'genre10', 'genre11', 'genre12', 'genre13', 'genre14', 'genre15', 'genre16',
                                            'genre17', 'genre18', 'genre19'], sep='|', engine='python',
                                     encoding='latin-1')
            df_like_p = pd.read_csv('./train_set.txt', sep=' ', names=['u', 'i', 'r', 't'])  # pointwise
            

            movie_id_list = movie_info['movie_id'].tolist()
            movie_name_list = movie_info['movie_name'].tolist()
            movie_name_dict = {movie_id_list[i]: movie_name_list[i] for i in range(len(movie_name_list))}

        elif dataset_name == 'ml-1M':
            df_like = pd.read_csv(f'./{dataset_name}/train_set.csv')
            df_dislike = pd.read_csv(f'./{dataset_name}/dislike.txt', header=None, names=['u', 'i', 'r', 't'])
            movie_info = pd.read_csv(f'./{dataset_name}/movies.dat', header=None,
                                     names=['movie_id', 'movie_name', 'genre'], sep='::', engine='python',
                                     encoding='latin-1')

            df_like_p = pd.read_csv(f'./{dataset_name}/train_set_prediction.csv', usecols=[0, 1, 2, 3])  # pointwise
            df_like_p.columns = ['u', 'i', 'r', 't']

            movie_id_list = movie_info['movie_id'].tolist()
            movie_name_list = movie_info['movie_name'].tolist()
            movie_name_dict = {movie_id_list[i]: movie_name_list[i] for i in range(len(movie_name_list))}


        elif dataset_name == 'bookcrossing':
            df_like = pd.read_csv(f'./{dataset_name}/train_set.txt', header=None, names=['u', 'i', 'r'], sep=' ')
            df_dislike = pd.read_csv(f'./{dataset_name}/dislike.txt', header=None, names=['u', 'i', 'r'])
            movie_info = pd.read_csv(f'./{dataset_name}/BX-Books.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
            df_like_p = pd.read_csv(f'./{dataset_name}/train_set_prediction.csv', usecols=[0, 1, 2])  # pointwise
            df_like_p.columns = ['u', 'i', 'r']

            movie_id_list = movie_info['ISBN'].tolist()
            movie_name_list = movie_info['Book-Title'].tolist()
            movie_name_dict = {movie_id_list[i]: movie_name_list[i] for i in range(len(movie_name_list))}

        mes_list_pointwise = []
        mes_list_pairwise = []
        mes_list_pairwise_inv = []
        mes_list_listwise = []

        my_ran_list = []

        for idx, row in rec_list.iterrows():

            gt_list = []
            uni = row['v0']

            df = df_like[df_like['u'] == uni]
            df_un = df_dislike[df_dislike['u'] == uni]
            unlikelist = []
            if len(df_un) > 0:
                my_list = df_un['i'].tolist()
                random.shuffle(my_list)
                if not 'book' in dataset_name:
                    my_list = [int(x) for x in my_list]
                # print(my_list)
                unlikelist = my_list[:10]

            '''
            Pairwise ranking
            '''
            my_list = df['i'].tolist()
            random.shuffle(my_list)
            if not 'book' in dataset_name:
                my_list = [int(x) for x in my_list]
            else:
                my_list = [(x) for x in my_list]

            if len(df) > 55:
                topk = 50
            else:
                #topk = len(df) - 3
                topk = max(5, len(df) - 3)  
            trainlist = my_list[:topk]
            
            #2.4
            
            if not trainlist:
                print(f"⚠️ Warning: Trainlist is empty for user {uni} - Skipping user.")
                continue  

            ran_list = [ii for ii in range(0, kk * 2)]
            ran_list_1 = ran_list[:kk]
            ran_list_2 = ran_list[kk:]
            ran_list = ran_list_1 + ran_list_2
            random.shuffle(ran_list)
            my_ran_list.append(ran_list)
            testlist_ = [row['v' + str(ii)] for ii in range(1, 1 + 2 * kk)]  # my_list[-5:]
            my_ground_truth = []
            for ii in range(0, kk, 1):
                testlist = [testlist_[ran_list[ii]]]
                neglist = [testlist_[ran_list[ii + kk]]]
                historical_interactions = [f'"{movie_name_dict[i]}"' for i in trainlist]
                false_items_set = [f'"{movie_name_dict[i]}"' for i in neglist]
                answer_items_set = [f'"{movie_name_dict[i]}"' for i in testlist]

                user_unpre = [f'"{movie_name_dict[i]}"' for i in unlikelist]

                if len(unlikelist) < 3:
                    user_unpre = 'None.'
                else:
                    user_unpre = ', '.join(user_unpre)

                historical_interactions = ', '.join(historical_interactions)

                gt_list.append(movie_name_dict[testlist[0]])

                first_name = answer_items_set[0]
                second_name = false_items_set[0]

                try:
                    xx = cm_item[str(neglist[0])]
                    yy = cm_item[str(testlist[0])]
                    #2.4
                    #uu = cm_user[str(uni)]
                    uu = cm_user.get(str(uni), None)
                    if xx is None:
                        print(f"⚠️ Warning: Item {neglist[0]} not found in neglist cm_item!")
                    if yy is None:
                        print(f"⚠️ Warning: Item {testlist[0]} not found in testlist cm_item!")
                    if uu is None:
                        print(f"⚠️ Warning: User {uni} not found in cm_user!")
                    
                    if xx is not None and yy is not None and uu is not None:
                        if cm_pred.get((str(uu), str(yy)), 0) > cm_pred.get((str(uu), str(xx)), 0):
                            cm_lable = 'Yes.'
                        else:
                            cm_lable = 'No.'
                    else:
                        cm_lable = 'Unknown.'
                        print('pairwiseUnknown')
                except IndexError as e:
                    print(f"⚠️ IndexError: {e}")
                    cm_lable = 'Unknown.'
                except Exception as e:
                    ####print(f"⚠️ Unexpected error: {e} in pairwise")
                    cm_lable = 'Unknown.'
                        

                instruct0 = f'''You are a {kws} recommender system. Based on a user's likes and dislikes, determine if they would prefer one {kws} over another. Respond only with "Yes." or "No.".\n\n'''

                instruct1 = f'''User's Liked {kws}s: <historical interactions>.\n\nUser's Disliked {kws}s: <user_unpre>\n\nQuestion: Would the user prefer <movie1> over <movie2>?'''
                instruct2 = '''Hint: Another recommender system suggests the answer is "<cm_result>".\n'''
                instruct3 = '''\nPlease only output "Yes." or "No.".\n\n'''
                if isHint:
                    instruct1 = instruct1 + instruct2

                instruct1 = instruct1.replace('<historical interactions>', historical_interactions).replace(
                    '<user_unpre>', user_unpre).replace('<movie1>', first_name).replace('<movie2>',
                                                                                        second_name).replace(
                    '<cm_result>', cm_lable)
                instruct2 = '<|endofmessage|><|assistant|>'
                fi = {'inst': instruct0 + instruct1 + instruct3}
                mes_list_pairwise.append(fi)

                instruct0 = f'''You are a {kws} recommender system. Based on a user's likes and dislikes, determine if they would prefer one {kws} over another. Respond only with "Yes." or "No.".\n\n'''

                instruct1 = f'''User's Liked {kws}s: <historical interactions>.\n\nUser's Disliked {kws}s: <user_unpre>\n\nQuestion: Would the user prefer <movie1> over <movie2>?'''
                instruct2 = '''Hint: Another recommender system suggests the answer is "<cm_result>".\n'''
                instruct3 = '''\nPlease only output "Yes." or "No.".\n\n'''
                if isHint:
                    instruct1 = instruct1 + instruct2

                if cm_lable == 'Yes.':
                    cm_lable = 'No'
                elif cm_lable == 'No.':
                    cm_lable = 'Yes'

                instruct1 = instruct1.replace('<historical interactions>', historical_interactions).replace(
                    '<user_unpre>', user_unpre).replace('<movie2>', first_name).replace('<movie1>',
                                                                                        second_name).replace(
                    '<cm_result>', cm_lable)
                instruct2 = '<|endofmessage|><|assistant|>'

                fi = {'inst': instruct0 + instruct1 + instruct3}
                mes_list_pairwise_inv.append(fi)

            df = pd.DataFrame(my_ran_list)

            '''
            Listwise Ranking
            '''
            total_list_mf = []
            for j_ in testlist_:
                try:
                    yy = cm_item[str(j_)]
                    #uu = cm_user[str(uni)]
                    uu = cm_user.get(str(uni), None)
                    if yy is None:
                        print(f"⚠️ Warning: Item {j_} not found in cm_item!")
                    if uu is None:
                        print(f"⚠️ Warning: User {uni} not found in cm_user!")
                    int_yy = int(j_)
                    int_uu = int(uni)
                    if int_uu < mf_pred.shape[0] and int_yy < mf_pred.shape[1]:
                        mf_label = mf_pred[int_uu][int_yy]
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

            total_list_mf_idx = sort_list_reverse_with_indices(total_list_mf)
            total_list_mf_idx = total_list_mf_idx[:5]
            total_list_mf_i = [testlist_[k_] for k_ in total_list_mf_idx] 
            mf_item_sets_ = ', '.join([f'"{movie_name_dict[i]}"' for i in total_list_mf_i])
            candidate_item_sets_ = ', '.join([f'"{movie_name_dict[i]}"' for i in testlist_])

            instruct0 = f'''You are a {kws} recommender system. Your task is to rank a given list of candidate {kws}s based on user preferences and return the top five recommendations.\n\n'''

            instruct1 = f'''User's Liked {kws}s: <historical_interactions>. \nUser's Disliked {kws}s: <user_unpre>\n\nQuestion: How would the user rank the candidate item list: <movie_list> based to historical perference?\n'''
            instruct2 = '''Hint: Another recommender model suggests <cm_result>.\n'''
            instruct3 = '''\nPlease only output the top five recommended movies once in the following format:\n1. [Movie Title]\n2. [Movie Title]\n3. [Movie Title]\n4. [Movie Title]\n5. [Movie Title].\n'''
            if isHint:
                instruct1 = instruct1 + instruct2

            instruct1 = instruct1.replace('<historical_interactions>', historical_interactions).replace('<user_unpre>',
                                                                                                        user_unpre).replace(
                '<movie_list>', candidate_item_sets_).replace('<cm_result>', mf_item_sets_)

            instruct2 = '<|endofmessage|><|assistant|>'


            fi = {'inst': instruct0 + instruct1 + instruct3}
            mes_list_listwise.append(fi)

            '''
            Pointwise Ranking
            '''
            gt_list = []

            dfp = df_like_p[df_like_p['u'] == uni]
            my_list = dfp['i'].tolist()
            my_list_r = dfp['r'].tolist()

            rndl = [i_ for i_ in range(len(my_list))]
            random.shuffle(rndl)

            if not 'book' in dataset_name:
                my_list = [int(my_list[x]) for x in rndl]
                my_list_r = [int(my_list_r[x]) for x in rndl]
            else:
                my_list = [(my_list[x]) for x in rndl]
                my_list_r = [(my_list_r[x]) for x in rndl]

            if len(df) > 50:
                topk = 50
            else:
                #topk = len(df)
                topk = max(5, len(df) - 3)  
            trainlist = my_list[:topk]
            trainlist_r = my_list_r[:topk]


            for testlist in testlist_:
                testlist = [testlist]
                #try:
                    #yy = mf_item[(testlist[0])]
                    #uu = mf_user[(uni)]
                    #mf_lable = mf_pred[uu][yy]

                #except Exception:
                    #mf_lable = 'Unknown.'
                    #print(1)
                #2.4
                yy = mf_item.get(str(testlist[0]), None)
                uu = mf_user.get(str(uni), None)
                
                if yy is None:
                    print(f"⚠️ Warning: Item {testlist[0]} not found in item mapping!")
                if uu is None:
                    print(f"⚠️ Warning: User {uni} not found in user mapping!")
                    
                if yy is not None and uu is not None:
                    try:
                        mf_label = mf_pred[uu][yy]
                        ###print("get-mf_label-=-mf_pred[uu][yy]-pointwise")
                    except IndexError:
                        print(f"⚠️ IndexError: ({uu}, {yy})  `mf_pred`  {mf_pred.shape}!")
                        mf_label = 'Unknown.' 
                    except Exception as e:
                        print(f"⚠️ Unexpected error in fetching mf_label: {e}")
                        mf_label = 'Unknown.'
                else:
                    mf_label = 'Unknown.'
                    print('mf_label')




                historical_interactions = [f'"{movie_name_dict[i]}"' for i in trainlist]
                answer_items_set = [f'"{movie_name_dict[i]}"' for i in testlist]

                historical_interactions = ', '.join(
                    [historical_interactions[i_] + ': ' + str(trainlist_r[i_]) + ';' for i_ in
                     range(len(historical_interactions))])

                if 'book' in dataset_name:
                    highest_score = 10
                else:
                    highest_score = 5
                instruct0 = f'''You are a {kws} recommender system. Your task is to predict the relevance score to a target {kws} based on the user's historical {kws} ratings. The score should be between 1 and {highest_score}, where 1 is the lowest affinity and {highest_score} is the highest. Respond only with a number between 1 to {highest_score}.\n\n'''

                instruct1 = f'''User's Historical {kws} Ratings: <historical interactions>. \n\nQuestion: Based on the user's historical ratings, predict the relavance score of the target {kws} <movie> with the user.\n'''
                instruct2 = '''Hint: Another recommender system suggests the answer is <mf_prediction>.\n'''
                instruct3 = '''\nPlease only output the score.\n'''
                if isHint:
                    instruct1 = instruct1 + instruct2

                instruct1 = instruct1.replace('<historical interactions>', historical_interactions).replace('<movie>',
                                                                                                            answer_items_set[
                                                                                                                0]).replace(
                    '<mf_prediction>', str(mf_label)[:3])

                instruct2 = '<|endofmessage|><|assistant|>'

                fi = {'inst': instruct0 + instruct1 + instruct3}
                mes_list_pointwise.append(fi)

        with jsonlines.open(
                f'./pointwisetest.jsonl',
                mode='w') as writer:
            writer.write_all(mes_list_pointwise)
        with jsonlines.open(
                f'./pairwisetest.jsonl',
                mode='w') as writer:
            writer.write_all(mes_list_pairwise)
        with jsonlines.open(
                f'./pairwise_invtest.jsonl',
                mode='w') as writer:
            writer.write_all(mes_list_pairwise_inv)
        with jsonlines.open(
                f'./listwisetest.jsonl',
                mode='w') as writer:
            writer.write_all(mes_list_listwise)

        df = pd.DataFrame(my_ran_list)
        #df.to_csv(f'./{dataset_name}/my_test_list{model_name}.txt', header=None, index=None)
        print("finished")
