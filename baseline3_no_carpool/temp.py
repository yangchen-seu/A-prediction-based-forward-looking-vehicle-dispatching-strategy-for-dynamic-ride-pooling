import Seeker
import pickle

with open('output/2017-05-01/history_order.pkl','rb') as tf:
    his_order = pickle.load(tf)
    order_id = []
    print(len(his_order))
    print(len(list(set(his_order.values()))))
    for key in his_order.keys():
        order_id.append(his_order[key].id)
        print('his_order[key].id',his_order[key].id,'his_order[key]',his_order[key])
        
    print(len(order_id))
    print(len(list(set(order_id))))
    # print(his_order)