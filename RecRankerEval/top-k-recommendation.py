# Configuration information of the initial recommendation model
from initialRec import initialRec
from util.conf import ModelConf

if __name__ == '__main__':
    graph_baselines = ['LightGCN','MF']
    ssl_graph_models = ['SGL', 'XSimGCL']
    sequential_baselines= ['SASRec']
    ssl_sequential_models = ['CL4SRec','DuoRec','BERT4Rec']
    model = 'LightGCN'
    import time
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='ml-1m', help='')
    parser.add_argument("--model", type=str, default='LightGCN', help='')
    parser.add_argument("--aug_type", type=str, default='0', help='')
    args = parser.parse_args()
    model = args.model
    s = time.time()
    if model in graph_baselines or model in ssl_graph_models or model in sequential_baselines or model in ssl_sequential_models:
        conf = ModelConf('./conf/' + model + '.conf')
    else:
        print('Wrong model name!')
        exit(-1)
    conf.__setitem__('training.set',f'./dataset/{args.dataset}/train_set.txt')
    conf.__setitem__('valid.set',f'./dataset/{args.dataset}/valid_set.txt')
    conf.__setitem__('test.set',f'./dataset/{args.dataset}/test_set.txt')
    conf.__setitem__('dislike.set',f'./dataset/{args.dataset}/dislike_set.txt')
    conf.__setitem__('aug_type',args.aug_type)
    print(conf['training.set'])
    rec = initialRec(conf)
    rec.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
